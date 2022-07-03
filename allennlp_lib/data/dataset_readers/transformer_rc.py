import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable
from collections import defaultdict
from multiprocessing import Value

from overrides import overrides
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
import torch.distributed as dist
import random

from allennlp.common.util import sanitize_wordpiece, is_distributed, get_spacy_model
from allennlp.data.fields import MetadataField, TextField, SpanField, IndexField, ArrayField
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from allennlp_lib.data.dataset_readers.utils import (
    char_span_to_token_span,
    answer_offset_in_context,
    TextClipper
)

from utils.constants import (
    PARAGRAPH_MARKER,
    CONSTITUENT_QUESTION_START, CONSTITUENT_QUESTION_END,
    REPLACEMENT_QUESTION_START, REPLACEMENT_QUESTION_END
)

logger = logging.getLogger(__name__)


@DatasetReader.register("transformer_rc")
class TransformerRCReader(DatasetReader):

    def __init__(
        self,
        transformer_model_name: str = "allenai/longformer-base-4096",
        prefer_question_key_if_present: str = None,
        question_key: str = "question_text",
        min_document_tokens: int = 0,
        max_document_tokens: int = 512,
        max_paragraph_tokens: int = 512,
        max_question_tokens: int = 100,
        global_prefix: str = "",
        take_predicted_topk_contexts: int = None,
        include_paragraph_title: bool = False,
        tokenizer_kwargs: Dict[str, Any] = None,
        **kwargs
    ) -> None:

        # Deprecated arguments.
        kwargs.pop("max_total_paragraphs", None)
        kwargs.pop("max_distractor_paragraphs", None)
        kwargs.pop("per_paragraph_instances", None)
        kwargs.pop("filter_type", None)
        kwargs.pop("topk", None)
        kwargs.pop("max_texts", None)
        kwargs.pop("primary_fraction", None)
        kwargs.pop("fill_in_placeholder_words", None)
        kwargs.pop("truncate_question_text", None)
        kwargs.pop("use_last_question_node_only", None)
        kwargs.pop("sample_any_para_having_answer", None)
        kwargs.pop("use_context_entities_only", None)
        kwargs.pop("add_word_overlap_rank", None)
        kwargs.pop("skip_num_hops_except", None)
        kwargs.pop("add_dataset_info", None)

        super().__init__(
            manual_distributed_sharding=kwargs.pop("manual_distributed_sharding", True),
            manual_multiprocess_sharding=kwargs.pop("manual_multiprocess_sharding", True),
            **kwargs
        )
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name,
            add_special_tokens=False,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                transformer_model_name, tokenizer_kwargs=tokenizer_kwargs
            )
        }

        if "t5" in transformer_model_name:
        # Explicitly set cls_token for T5 since it doesn't have by default.

            assert "cls_token" in tokenizer_kwargs, "CLS token isn't set for T5."

            cls_token = Token(
                self._tokenizer.tokenizer.cls_token,
                text_id=self._tokenizer.tokenizer.cls_token_id,
                type_id=self._tokenizer.single_sequence_end_tokens[0].type_id,
            )
            self._tokenizer.sequence_pair_start_tokens = [cls_token]
            self._tokenizer.single_sequence_start_tokens = self._tokenizer.single_sequence_end_tokens

        self._min_document_tokens = min_document_tokens
        self._max_document_tokens = max_document_tokens
        self._max_paragraph_tokens = max_paragraph_tokens
        self._max_question_tokens = max_question_tokens

        self._spacy = get_spacy_model("en_core_web_sm", True, True, True)

        self._cls_token = self._tokenizer.tokenizer.cls_token

        # Note: Don't change order of these tokens.
        additional_tokens = [PARAGRAPH_MARKER, CONSTITUENT_QUESTION_START, CONSTITUENT_QUESTION_END,
                             REPLACEMENT_QUESTION_START, REPLACEMENT_QUESTION_END]
        self._tokenizer.tokenizer.add_tokens(additional_tokens)

        for added_token in additional_tokens:
            check1 = len(self._tokenizer.tokenize(added_token)) == 1
            check3 = sanitize_wordpiece(self._tokenizer.tokenize(added_token)[0].text) == added_token
            if not all([check1, check3]):
                raise Exception(f"The added token ({added_token}) doesn't tokenize to 1 token.")

        self._paragraph_clipper = TextClipper(self._tokenizer, self._max_paragraph_tokens)

        self._prefer_question_key_if_present = prefer_question_key_if_present
        self._question_key = question_key
        self._include_paragraph_title = include_paragraph_title

        self._global_prefix = global_prefix
        self._take_predicted_topk_contexts = take_predicted_topk_contexts
        self._stopwords_list = set(stopwords.words('english'))

        self._count = defaultdict(lambda : Value("d", 0))


    def json_to_instance(self, instance_dict: Dict) -> Instance:

        clean_ws = lambda text: " ".join(text.split())

        question_id = instance_dict["id"]
        answer_text = instance_dict["answer_text"]
        contexts = instance_dict["contexts"]
        answerability = instance_dict["answerable"]
        dataset = instance_dict["dataset"]

        question_key = self._question_key
        if self._prefer_question_key_if_present and self._prefer_question_key_if_present in instance_dict:
            question_key = self._prefer_question_key_if_present

        question_text = instance_dict[question_key]
        question_text = clean_ws(question_text)

        while PARAGRAPH_MARKER in question_text:
            question_text = question_text.replace(PARAGRAPH_MARKER, " ")

        skipped_support_contexts = []
        if self._take_predicted_topk_contexts is not None:
            contexts = instance_dict["predicted_ordered_contexts"][:self._take_predicted_topk_contexts]
            skipped_support_contexts = [
                context
                for context in
                instance_dict["predicted_ordered_contexts"][self._take_predicted_topk_contexts:]
                if context["is_supporting"]
            ]

        # To make sure the prefixes can be easily removed post-hoc.
        for context in contexts:
            context["original_paragraph_text"] = context["paragraph_text"]

        # This should be kept after take_predicted_topk_contexts
        if self._include_paragraph_title:
            for context in contexts:
                context["paragraph_text"] = " || ".join([context["wikipedia_title"], context["paragraph_text"]])

        assert len({context["paragraph_text"] for context in contexts}) == len(contexts), \
            f"Found duplicate paragraph text in QID: {question_id}."

        answer_offset = None

        context_text = (self._global_prefix+' ').lstrip()
        support_indices = []
        for index, context in enumerate(contexts):

            paragraph_text = context["paragraph_text"]
            is_supporting = context["is_supporting"]

            paragraph_text = clean_ws(paragraph_text)
            paragraph_text = self._paragraph_clipper.clip(paragraph_text)

            if PARAGRAPH_MARKER in paragraph_text:
                logger.info(f"WARNING: sep_token ({PARAGRAPH_MARKER})"
                            f" found in '{paragraph_text}'")
            while PARAGRAPH_MARKER in paragraph_text:
                paragraph_text = paragraph_text.replace(PARAGRAPH_MARKER, " ")
            paragraph_text = PARAGRAPH_MARKER+' '+paragraph_text.strip()+' '

            if is_supporting:
                support_indices.append(index)

                local_answer_offset = answer_offset_in_context(answer_text, paragraph_text)
                if local_answer_offset is not None:
                    answer_offset = len(context_text)+local_answer_offset

            context_text += paragraph_text

        if not context_text.strip():
            # In very very rare (maybe just a 1-2) cases, the unanswerable instance
            # has all the contexts wiped out. In this case, we need to put something in it.
            # TODO: May be I should handle it in combining of answerable+unanswerable datasets?
            logger.info(f"WARNING: Found an instance without any context content.")
            context_text = "<empty context>"

        for context in contexts:
            context["paragraph_text"] = context.pop("original_paragraph_text")

        assert context_text.count(PARAGRAPH_MARKER) == len(contexts), \
            "Inserted para separators don't add up to the actual number of paras."

        if answer_offset is None or answer_text in self._global_prefix:
            answer_offset = answer_offset_in_context(answer_text, context_text)

        if answerability and answer_offset is None:
            self._count["impossible_questions"].value += 1
        self._count["questions"].value += 1

        self._count["paragraphs"].value += context_text.count(PARAGRAPH_MARKER)

        answers = [answer_text]
        if not answerability:
            answers = []
            answer_offset = None

        partition_id = instance_dict.get("partition_index", 0)

        return self.make_instance(
            qid=question_id,
            question=question_text,
            answers=answers,
            context=context_text,
            first_answer_offset=answer_offset,
            support_indices=support_indices,
            answerability=answerability,
            contexts=contexts,
            skipped_support_contexts=skipped_support_contexts,
            partition_id=partition_id
        )


    @overrides
    def _read(self, file_path: str):
        logger.info(f"Reading the dataset: {file_path}")

        with open(file_path, "r") as file:

            lines = file.readlines()

            for line in self.shard_iterable(lines):
                if not line.strip():
                    continue

                instance_dict = json.loads(line.strip())

                yield self.json_to_instance(instance_dict)

        self._warn_impossible_percentages(force=True)
        self._log_num_tokens_statistics()
        self._log_num_paragraphs_statistics()


    def make_instance(
        self,
        qid: str,
        question: str,
        answers: List[str],
        context: str,
        first_answer_offset: Optional[int],
        support_indices: List[int] = None,
        answerability: bool = None,
        contexts: List[str] = None,
        skipped_support_contexts: List[str] = None,
        partition_id: int = None
    ) -> Instance:
        """
        Create training instances from a SQuAD example.
        """
        if bool(answerability) and not bool(answers):
            raise Exception("Instance is answerable but answers are not provided.")
        if not bool(answerability) and bool(answers):
            raise Exception("Instance is unanswerable but answers are still provided.")

        # tokenize context by spaces first, and then with the wordpiece tokenizer
        # For RoBERTa, this produces a bug where every token is marked as beginning-of-sentence. To fix it, we
        # detect whether a space comes before a word, and if so, add "a " in front of the word.
        def tokenize_slice(start: int, end: int) -> Iterable[Token]:
            text_to_tokenize = context[start:end]
            if start - 1 >= 0 and context[start - 1].isspace():
                prefix = "a "  # must end in a space, and be short so we can be sure it becomes only one token
                wordpieces = self._tokenizer.tokenize(prefix + text_to_tokenize)
                for wordpiece in wordpieces:
                    if wordpiece.idx is not None:
                        wordpiece.idx -= len(prefix)
                return wordpieces[1:]
            else:
                return self._tokenizer.tokenize(text_to_tokenize)

        tokenized_context = []
        token_start = 0
        context = context.strip()
        for i, c in enumerate(context):
            if c.isspace():
                for wordpiece in tokenize_slice(token_start, i):
                    if wordpiece.idx is not None:
                        wordpiece.idx += token_start
                    tokenized_context.append(wordpiece)
                token_start = i + 1
        for wordpiece in tokenize_slice(token_start, len(context)):
            if wordpiece.idx is not None:
                wordpiece.idx += token_start
            tokenized_context.append(wordpiece)

        if first_answer_offset is None:
            (token_answer_span_start, token_answer_span_end) = (-1, -1)
        else:
            token_offsets = [
                (t.idx, t.idx + len(sanitize_wordpiece(t.text))) if t.idx is not None else None
                for t in tokenized_context
            ]
            character_span = (first_answer_offset, first_answer_offset + len(answers[0]))
            (token_answer_span_start, token_answer_span_end), _ = char_span_to_token_span(
                token_offsets, character_span
            )

        # Tokenize the question.
        question = question.strip()
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_question = tokenized_question[: self._max_question_tokens]

        # Stride over the context, making instances. [striding disabled]
        num_para_markers = len(self._find_para_sep_indices(tokenized_context))
        space_for_context = (
            self._max_document_tokens
            - len(tokenized_question)
            - len(self._tokenizer.sequence_pair_start_tokens)
            - len(self._tokenizer.sequence_pair_mid_tokens)
            - len(self._tokenizer.sequence_pair_end_tokens)
            - num_para_markers
        )

        # Assure markers aren't clipped away.
        tokenized_context = tokenized_context[:space_for_context] + \
            [t for t in tokenized_context[space_for_context:]
             if sanitize_wordpiece(t.text) == PARAGRAPH_MARKER]

        token_answer_span = (token_answer_span_start, token_answer_span_end)
        if any(i < 0 or i >= len(tokenized_context) for i in token_answer_span):
            # The answer is not contained in the window.
            token_answer_span = None

        assert context.count(PARAGRAPH_MARKER) == \
            len(self._find_para_sep_indices(tokenized_context)), \
            "Looks like some paragraph markers got clipped away."

        if self._min_document_tokens:
            min_padding = self._min_document_tokens - len(tokenized_question+tokenized_context)
            if min_padding > 0:
                has_empty_token = self._tokenizer.tokenize(" ")[0]
                tokenized_context.extend([Token(
                    has_empty_token.text,
                    text_id=tokenized_context[-1].text_id,
                    type_id=tokenized_context[-1].type_id,
                    idx=tokenized_context[-1].idx+i+1
                 )
                 for i in range(min_padding)])

        additional_metadata = {"id": qid}
        if contexts:
            additional_metadata["contexts"] = contexts
        if skipped_support_contexts:
            additional_metadata["skipped_support_contexts"] = skipped_support_contexts

        if partition_id is not None:
            additional_metadata["partition_id"] = partition_id

        return self.text_to_instance(
            question,
            tokenized_question,
            context,
            tokenized_context,
            answers=answers,
            token_answer_span=token_answer_span,
            support_indices=support_indices,
            additional_metadata=additional_metadata,
            answerability=answerability
        )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        tokenized_question: List[Token],
        context: str,
        tokenized_context: List[Token],
        answers: List[str] = None,
        token_answer_span: Optional[Tuple[int, int]] = None,
        support_indices: List[int] = None,
        answerability: bool = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Instance:
        fields = {}

        # make the document (context, question, everything) field
        document_field = TextField(
            self._tokenizer.add_special_tokens(tokenized_question, tokenized_context)
        )
        fields["document"] = document_field

        cls_index = self._find_cls_index(document_field.tokens)
        fields["cls_index"] = ArrayField(np.array(cls_index), dtype=np.long)

        start_of_context = (
            len(self._tokenizer.sequence_pair_start_tokens)
            + len(tokenized_question)
            + len(self._tokenizer.sequence_pair_mid_tokens)
        )

        global_attention_mask = [0] * len(document_field.tokens)
        for i in range(start_of_context):
            global_attention_mask[i] = 1
        for i in self._find_para_sep_indices(document_field.tokens):
            global_attention_mask[i] = 1
        global_attention_mask = ArrayField(np.array(global_attention_mask),
                                           dtype=np.long, padding_value=0)
        fields["global_attention_mask"] = global_attention_mask

        # make the answer span
        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span[0] <= token_answer_span[1]

            fields["answer_span_labels"] = SpanField(
                token_answer_span[0] + start_of_context,
                token_answer_span[1] + start_of_context,
                document_field,
            )
        else:
            fields["answer_span_labels"] = SpanField(-1, -1, document_field)

        if answerability is not None:
            fields["answerability_labels"] = ArrayField(np.array(int(answerability)),
                                                        dtype=np.long)

        # make the context span, i.e., the span of text from which possible answers should be drawn
        fields["context_span"] = SpanField(
            start_of_context, start_of_context + len(tokenized_context) - 1, document_field
        )

        support_positions = self._find_para_sep_indices(document_field.tokens)
        fields["support_positions"] = ArrayField(np.array(support_positions),
                                                 dtype=np.long, padding_value=-1)

        support_labels = [0] * len(support_positions)

        # Note: Currently, I'm NOT supervising support for unanswerables.
        # In case of dire runs, all instances are marked answerables. So all +ves/-ves are supervised for support.
        if answerability is None or answerability:
            # answerability is None when labels/answers are not passed (eg. in test prediction mode)
            # Otherwise, answerability is always true/false. None doesn't have any semantic purpose, it's the default.
            for index in support_indices:
                support_labels[index] = 1
        fields["support_labels"] = ArrayField(np.array(support_labels),
                                              dtype=np.float32, padding_value=-1)

        # make the metadata
        metadata = {
            "question": question,
            "question_tokens": tokenized_question,
            "context": context,
            "context_tokens": tokenized_context,
            "answers": answers or [],
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        # Add some statistics info.
        self._count["document_tokens"].value += len(document_field.tokens)
        self._count["question_tokens"].value += len(tokenized_question)
        self._count["context_tokens"].value += len(tokenized_context)

        return Instance(fields)

    def _find_cls_index(self, tokens: List[Token]) -> int:
        return next(i for i, t in enumerate(tokens) if t.text == self._cls_token)

    def _find_para_sep_indices(self, tokens: List[Token]) -> int:
        return [i for i, t in enumerate(tokens)
                if sanitize_wordpiece(t.text) == PARAGRAPH_MARKER]

    def _should_log(self) -> bool:
        if is_distributed() and dist.get_rank() != 0:
            return False
        if self._worker_info and self._worker_info.id != 0:
            return False
        return True

    def _warn_impossible_percentages(self, force: bool=False) -> None:
        if not self._should_log():
            return None
        numerator = self._count["impossible_questions"].value
        denominator = self._count["questions"].value
        fraction = numerator / denominator
        is_significant = denominator > 1000 and fraction > 0.02
        if is_significant or force:
            percentage = round(100*fraction, 2)
            logger.info(f"Impossible percentage   : {percentage}% ({numerator}/{denominator}) ")

    def _log_num_tokens_statistics(self) -> None:
        if not self._should_log():
            return None
        get_avg = lambda num, denom: round(num / denom, 3)
        avg_num_document_tokens = get_avg(self._count['document_tokens'].value, self._count['questions'].value)
        avg_num_question_tokens = get_avg(self._count['question_tokens'].value, self._count['questions'].value)
        avg_num_context_tokens = get_avg(self._count['context_tokens'].value, self._count['questions'].value)
        logger.info(f"Average document count  : {avg_num_document_tokens}.")
        logger.info(f"Average question count  : {avg_num_question_tokens}.")
        logger.info(f"Average context  count  : {avg_num_context_tokens}.")

    def _log_num_paragraphs_statistics(self) -> None:
        if not self._should_log():
            return None
        average_count = self._count["paragraphs"].value / self._count["questions"].value
        logger.info(f"Average paragraph count : {average_count}.")

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["document"]._token_indexers = self._token_indexers

    def _get_word_overlap(self, question_text: str, context_text: str) -> float:

        question_tokens = self._tokenizer.tokenizer(question_text.lower())
        context_tokens = self._tokenizer.tokenizer(context_text.lower())

        word_overlap = sum([1 for token in question_tokens
                            if token in context_tokens and token not in self._stopwords_list])
        max_word_overlap = sum([1 for token in question_tokens if token not in self._stopwords_list])

        return float(word_overlap/max_word_overlap) if max_word_overlap != 0 else 0.0
