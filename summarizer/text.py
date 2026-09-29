from __future__ import annotations

import re
from collections.abc import Callable

from nltk.tokenize import PunktSentenceTokenizer

from summarizer.providers.base import GenerationRequest


SYSTEM_INSTRUCTIONS = (
    "You are a writing assistant, skilled in revising and summarizing "
    "complex technical writing with accuracy and precision."
)
USER_PROMPT_PREFIX = (
    "Provide an executive summary of the following text (delimited by "
    "triple quotes). Present the key ideas and findings directly, without "
    "bullet points, as if for a busy professional who needs to grasp the "
    "essential points quickly. Ignore complete sentences and grammatical "
    "correctness. Abbreviate long and repetitive words. "
)
DELIMITER = '\n"""\n'
_SENTENCE_TOKENIZER = PunktSentenceTokenizer()
# Keep the transitional, resource-free tokenizer useful for common prose. Issue
# #4 replaces this legacy chunker with the generalized segmentation pipeline.
_SENTENCE_TOKENIZER._params.abbrev_types.update(
    {
        "co",
        "dept",
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "u.k",
        "u.s",
        "vs",
    }
)


def default_sentence_tokenizer(text: str) -> list[str]:
    return _SENTENCE_TOKENIZER.tokenize(text)


# A complete sentence ends in terminal punctuation, optionally followed by
# closing quotes, brackets or emphasis markers.
_SENTENCE_END = re.compile(r"[.!?…。！？][\"'”’»)\]}*_]*\Z")


def original_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """`(start, end, stripped text)` of each tokenizer sentence in `text`."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in default_sentence_tokenizer(text):
        index = text.find(sentence, cursor)
        if index < 0:
            return []
        spans.append((index, index + len(sentence), sentence.strip()))
        cursor = index + len(sentence)
    return spans


def split_unfinished_ending(draft: str) -> tuple[str, str | None]:
    """Split off the draft's last sentence when it stops without ending.

    A model that stops mid-sentence leaves a fragment that makes no complete
    claim. Only the last sentence is checked: earlier ones without terminal
    punctuation may be list items or headings, and the tokenizer joins a
    mid-text fragment to what follows. The draft's own ending is tested,
    because the tokenizer splits a closing marker such as ``**`` into a piece
    of its own.
    """
    spans = original_sentence_spans(draft)
    if not spans or _SENTENCE_END.search(draft.rstrip()):
        return draft, None
    start, _, text = spans[-1]
    if not any(character.isalnum() for character in text):
        return draft, None
    return draft[:start].rstrip(), text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip()


def chunk_text_by_sentences(
    text: str,
    max_chunk_size: int,
    sentence_tokenizer: Callable[[str], list[str]] = default_sentence_tokenizer,
) -> list[str]:
    chunks: list[str] = []
    current_chunk = ""
    for sentence in sentence_tokenizer(text):
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def build_generation_request(
    chunk: str,
    *,
    model: str,
    timeout_seconds: float,
    operation_id: str | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input_text=f"{USER_PROMPT_PREFIX}{DELIMITER}{chunk}{DELIMITER}",
        timeout_seconds=timeout_seconds,
        operation_id=operation_id,
    )
