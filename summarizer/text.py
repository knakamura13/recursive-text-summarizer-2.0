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


def default_sentence_tokenizer(text: str) -> list[str]:
    return _SENTENCE_TOKENIZER.tokenize(text)


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
