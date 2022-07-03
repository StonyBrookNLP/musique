from typing import Dict, List, Union, Any
import logging
import json
import random

import numpy as np
from overrides import overrides
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


logger = logging.getLogger(__name__)


@DatasetReader.register("text_ranker")
class TextRankerReader(DatasetReader):

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        max_tokens: int = None,
        add_question_info: bool = False,
        add_word_overlap_rank: bool = False,
        question_key: str = "question_text",
        balance: bool = False,
        **kwargs,
    ) -> None:

        # Deprecated arguments.
        kwargs.pop("filter_type")
        kwargs.pop("topk")
        kwargs.pop("max_texts")
        kwargs.pop("primary_fraction")
        kwargs.pop("overwrite_support_by_answer")
        kwargs.pop("add_dataset_info")
        kwargs.pop("skip_word_content")

        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._max_tokens = max_tokens
        self._add_question_info = add_question_info
        self._prefix_separator = " SEP "
        self._question_key = question_key

        self._is_transformer = isinstance(tokenizer, PretrainedTransformerTokenizer)
        self._stopwords_list = set(stopwords.words('english'))
        self._add_word_overlap_rank = add_word_overlap_rank
        self._balance = balance

    def json_to_instance(self, instance_dict: Dict) -> Instance:

        contexts = instance_dict["contexts"]

        dataset = instance_dict["dataset"]
        texts, labels = [], []

        question_key = self._question_key

        clean_ws = lambda text: " ".join(text.split())
        question_text = clean_ws(instance_dict[question_key])

        if self._add_question_info:
            for context in contexts:
                if self._prefix_separator in context["paragraph_text"]:
                    logger.info(f"WARNING: prefix-separator ({self._prefix_separator}) "
                                f"found in paragraph_text: '{context['paragraph_text']}'")
                context["prefix_text"] = (" " + question_text + " " + context.get("prefix_text", ""))


        if self._add_word_overlap_rank:
            sorting_func = lambda context: self._get_word_overlap(question_text, context["paragraph_text"])
            sorted_contexts = sorted(contexts, key=sorting_func, reverse=True)
            for rank, context in enumerate(sorted_contexts):
                context["prefix_text"] = (" " + str(rank) + " " + context.get("prefix_text", ""))

        if self._balance:
            supporting_contexts = [context for context in contexts if context["is_supporting"]]
            non_supporting_contexts = [context for context in contexts if not context["is_supporting"]]
            non_supporting_contexts = random.sample(
                non_supporting_contexts, min(len(supporting_contexts), len(non_supporting_contexts))
            )
            contexts = supporting_contexts+non_supporting_contexts
            random.shuffle(contexts)

        metadata = {"id": instance_dict["id"], "contexts": contexts}

        for context in contexts:
            text = self._prefix_separator.join(
                [context["prefix_text"].strip(), context["paragraph_text"].strip()]
            )
            tokens = text.split(" ")[:self._max_tokens]
            text = " ".join(tokens)
            texts.append(text)

            labels.append(context["is_supporting"])

        return self.text_to_instance(texts, labels, metadata)

    @overrides
    def _read(self, file_path):

        with open(file_path, "r") as file:

            lines = file.readlines()
            for line in self.shard_iterable(lines):
                if not line.strip():
                    continue

                instance_dict = json.loads(line.strip())
                yield self.json_to_instance(instance_dict)

    @overrides
    def text_to_instance(
            self,
            texts: List[str],
            labels: List[bool],
            metadata: Dict[str, Any] = None
        ) -> Instance:

        fields: Dict[str, Field] = {}

        cls_indices = []
        text_fields = []
        for text in texts:
            text = text.strip()
            tokens = self._tokenizer.tokenize(text)[: self._max_tokens]
            if self._is_transformer:
                tokens = self._tokenizer.add_special_tokens(tokens)
            text_field = TextField(tokens)
            text_fields.append(text_field)

            if self._is_transformer:
                cls_index = self._find_cls_index(text_field.tokens)
                cls_indices.append(cls_index)

        if not text_fields: # In mixed-answerable set there can be some with no context.
            text_fields = [TextField(self._tokenizer.tokenize("dummy")).empty_field()]

        fields["texts"] = ListField(text_fields)

        if self._is_transformer:
            fields["cls_indices"] = ArrayField(np.array(cls_indices), dtype=np.long, padding_value=-1)

        fields["labels"] = ArrayField(np.array(labels), dtype=np.float32, padding_value=-1)

        if metadata:
            fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        for text_field in instance.fields["texts"]:  # type: ignore
            text_field._token_indexers = self._token_indexers

    def _find_cls_index(self, tokens: List[Token]) -> int:
        return next(i for i, t in enumerate(tokens) if t.text == self._tokenizer.tokenizer._cls_token)

    def _get_word_overlap(self, question_text: str, context_text: str) -> float:

        question_tokens = [token.text for token in self._tokenizer.tokenize(question_text.lower())]
        context_tokens = [token.text for token in self._tokenizer.tokenize(context_text.lower())]

        word_overlap = sum([1 for token in question_tokens
                            if token in context_tokens and token not in self._stopwords_list])
        max_word_overlap = sum([1 for token in question_tokens if token not in self._stopwords_list])

        return float(word_overlap/max_word_overlap) if max_word_overlap != 0 else 0.0
