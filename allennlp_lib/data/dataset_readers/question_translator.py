import json
import logging
from typing import Any, Dict, Iterable

from overrides import overrides
import numpy as np
import torch.distributed as dist
import random

from allennlp.common.util import sanitize_wordpiece, is_distributed
from allennlp.data.fields import TextField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from utils.constants import (
    PARAGRAPH_MARKER,
    CONSTITUENT_QUESTION_START, CONSTITUENT_QUESTION_END,
    REPLACEMENT_QUESTION_START, REPLACEMENT_QUESTION_END
)

logger = logging.getLogger(__name__)


@DatasetReader.register("question_translator")
class QuestionTranslatorReader(DatasetReader):


    def __init__(
        self,
        translation_type: str,
        transformer_model_name: str = "facebook/bart-base",
        max_source_tokens: int = 100,
        max_target_tokens: int = 100,
        composed_question_key: str = "composed_question_text",
        tokenizer_kwargs: Dict[str, Any] = None,
        **kwargs
    ) -> None:

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

        self._translation_type = translation_type
        assert translation_type in ("compose", "decompose"), \
            f"{translation_type} should be one of compose or decompose."

        self._source_max_tokens = max_source_tokens
        self._target_max_tokens = max_target_tokens

        # # Note: Don't change order of these tokens.
        # additional_tokens = [PARAGRAPH_MARKER, CONSTITUENT_QUESTION_START, CONSTITUENT_QUESTION_END,
        #                      REPLACEMENT_QUESTION_START, REPLACEMENT_QUESTION_END]
        # self._tokenizer.tokenizer.add_tokens(additional_tokens)

        additional_tokens = []
        for index in range(5):
            additional_tokens.append(f"{CONSTITUENT_QUESTION_START} {index} {CONSTITUENT_QUESTION_END}")
            additional_tokens.append(f"{REPLACEMENT_QUESTION_START} {index} {REPLACEMENT_QUESTION_END}")
        self._tokenizer.tokenizer.add_tokens(additional_tokens)
        self._additional_tokens = additional_tokens

        for added_token in additional_tokens:
            check1 = len(self._tokenizer.tokenize(added_token)) == 1
            check3 = sanitize_wordpiece(self._tokenizer.tokenize(added_token)[0].text) == added_token
            if not all([check1, check3]):
                raise Exception(f"The added token ({added_token}) doesn't tokenize to 1 token.")

        self._composed_question_key = composed_question_key


    def json_to_instance(self, instance_dict: Dict) -> Instance:

        decomposed_question_text = " ".join(
            [sub_question_text for sub_question_text in instance_dict["component_question_texts"]]
        )
        assert any([token in decomposed_question_text for token in self._additional_tokens])

        if self._translation_type == "compose":
            composed_question_text = instance_dict.get(self._composed_question_key, None)
            source_text, target_text = decomposed_question_text, composed_question_text
        else:
            composed_question_text = instance_dict[self._composed_question_key]
            source_text, target_text = composed_question_text, decomposed_question_text

        source_text = source_text.strip()

        # This shouldn't be required. But it was for overfitting fixture.
        if target_text is not None:
            target_text = " ".join([
                self._tokenizer.tokenizer.bos_token or "",
                target_text.strip(),
                self._tokenizer.tokenizer.eos_token
            ]).strip()

        return self.text_to_instance(source_text, target_text)


    @overrides
    def _read(self, file_path: str):
        logger.info(f"Reading the dataset: {file_path}")

        with open(file_path, "r") as file:
            for line in self.shard_iterable(file.readlines()):
                if not line.strip():
                    continue

                instance_dict = json.loads(line.strip())
                yield self.json_to_instance(instance_dict)


    @overrides
    def text_to_instance(
        self, source_text: str, target_text: str = None
    ) -> Instance:  # type: ignore
        tokenized_source_text = self._tokenizer.tokenize(source_text)
        tokenized_source_text = tokenized_source_text[: self._source_max_tokens]

        source_field = TextField(tokenized_source_text)
        if target_text is not None:
            tokenized_target_text = self._tokenizer.tokenize(target_text)[: self._target_max_tokens]
            target_field = TextField(tokenized_target_text)

            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:

            return Instance({"source_tokens": source_field})


    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._token_indexers  # type: ignore
