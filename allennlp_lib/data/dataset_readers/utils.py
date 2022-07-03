"""
Utilities for reading comprehension dataset readers.
"""

import logging
import string
from typing import List, Tuple, Optional

from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.common.util import sanitize_wordpiece

logger = logging.getLogger(__name__)

# These are tokens and characters that are stripped by the standard SQuAD and TriviaQA evaluation
# scripts.
IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])


def normalize_text(text: str) -> str:
    """
    Performs a normalization that is very similar to that done by the normalization functions in
    SQuAD and TriviaQA.

    This involves splitting and rejoining the text, and could be a somewhat expensive operation.
    """
    return " ".join(
        [
            token
            for token in text.lower().strip(STRIPPED_CHARACTERS).split()
            if token not in IGNORED_TOKENS
        ]
    )


def char_span_to_token_span(
    token_offsets: List[Optional[Tuple[int, int]]], character_span: Tuple[int, int]
) -> Tuple[Tuple[int, int], bool]:
    """
    Converts a character span from a passage into the corresponding token span in the tokenized
    version of the passage.  If you pass in a character span that does not correspond to complete
    tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
    We return an error flag in this case, and have some debug logging so you can figure out the
    cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
    problems; there's a fair amount of both).

    The basic outline of this method is to find the token span that has the same offsets as the
    input character span.  If the tokenizer tokenized the passage correctly and has matching
    offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
    but mostly just find the closest thing we can.

    The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
    So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
    two-word span beginning at token index 3, and so on.

    Returns
    -------
    token_span : ``Tuple[int, int]``
        `Inclusive` span start and end token indices that match as closely as possible to the input
        character spans.
    error : ``bool``
        Whether there was an error while matching the token spans exactly. If this is ``True``, it
        means there was an error in either the tokenization or the annotated character span. If this
        is ``False``, it means that we found tokens that match the character span exactly.
    """
    error = False
    start_index = 0
    while start_index < len(token_offsets) and (
        token_offsets[start_index] is None or token_offsets[start_index][0] < character_span[0]
    ):
        start_index += 1

    # If we overshot and the token prior to start_index ends after the first character, back up.
    if (
        start_index > 0
        and token_offsets[start_index - 1] is not None
        and token_offsets[start_index - 1][1] > character_span[0]
    ) or (
        start_index <= len(token_offsets)
        and token_offsets[start_index] is not None
        and token_offsets[start_index][0] > character_span[0]
    ):
        start_index -= 1

    if start_index >= len(token_offsets):
        raise ValueError("Could not find the start token given the offsets.")

    if token_offsets[start_index] is None or token_offsets[start_index][0] != character_span[0]:
        error = True

    end_index = start_index
    while end_index < len(token_offsets) and (
        token_offsets[end_index] is None or token_offsets[end_index][1] < character_span[1]
    ):
        end_index += 1
    if end_index == len(token_offsets):
        # We want a character span that goes beyond the last token. Let's see if this is salvageable.
        # We consider this salvageable if the span we're looking for starts before the last token ends.
        # In other words, we don't salvage if the whole span comes after the tokens end.
        if character_span[0] < token_offsets[-1][1]:
            # We also want to make sure we aren't way off. We need to be within 8 characters to salvage.
            if character_span[1] - 8 < token_offsets[-1][1]:
                end_index -= 1

    if end_index >= len(token_offsets):
        raise ValueError("Character span %r outside the range of the given tokens.")
    if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
        # Looks like there was a token that should have been split, like "1854-1855", where the
        # answer is "1854".  We can't do much in this case, except keep the answer as the whole
        # token.
        logger.debug("Bad tokenization - end offset doesn't match")
    elif token_offsets[end_index][1] > character_span[1]:
        # This is a case where the given answer span is more than one token, and the last token is
        # cut off for some reason, like "split with Luckett and Rober", when the original passage
        # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
        # where it is, and assume the intent was to mark the whole token.
        logger.debug("Bad labelling or tokenization - end offset doesn't match")
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


def answer_offset_in_context(answer_text: str, context_text: str) -> int:
    answer_text = answer_text.strip()
    context_text = context_text.strip()
    if f" {answer_text} " in context_text:
        return context_text.index(f" {answer_text} ")+1
    elif f" {answer_text}" in context_text:
        return context_text.index(f" {answer_text}")+1
    elif f"{answer_text} " in context_text:
        return context_text.index(f"{answer_text} ")
    elif answer_text in context_text:
        return context_text.index(answer_text)

    # If can't find case-specific occurrence, find first lower cased.
    answer_text = answer_text.lower()
    context_text = context_text.lower()
    if f" {answer_text} " in context_text:
        return context_text.index(f" {answer_text} ")+1
    elif f" {answer_text}" in context_text:
        return context_text.index(f" {answer_text}")+1
    elif f"{answer_text} " in context_text:
        return context_text.index(f"{answer_text} ")
    elif answer_text in context_text:
        return context_text.index(answer_text)

    return None


class TextClipper:

    def __init__(self,
                 tokenizer: Tokenizer,
                 max_num_tokens: int) -> None:
        self._max_num_tokens = max_num_tokens
        self._tokenizer = tokenizer

    def clip(self, text: str) -> str:

        text = ' '.join(text.split(' ')[:self._max_num_tokens])

        all_tokens = self._tokenizer.tokenize(text)
        tokens = all_tokens[:self._max_num_tokens]

        if not tokens:
            return text

        last_non_empty_token = None
        for token in tokens[::-1]:
            if sanitize_wordpiece(token.text).strip():
                last_non_empty_token = token
                break

        if last_non_empty_token is None:
            return ""

        start = tokens[0].idx
        end = last_non_empty_token.idx + len(
            sanitize_wordpiece(last_non_empty_token.text)
        )

        text = text[start:end]

        return text
