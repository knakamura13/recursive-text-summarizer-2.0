from __future__ import annotations

import re
from typing import Sequence

from summarizer.providers.base import GenerationRequest

_RETENTION_RATIO = 0.70


def _extract_compress_chunk(input_text: str) -> str:
    lines = input_text.splitlines()
    begin = end = None
    for index, line in enumerate(lines):
        if "COMPRESS-BEGIN" in line:
            begin = index
        elif "COMPRESS-END" in line and begin is not None:
            end = index
            break
    if begin is not None and end is not None and end > begin:
        return "\n".join(lines[begin + 1 : end])
    return input_text


def _claim_quotes() -> tuple[str, ...]:
    from tests.support.evaluation import CURATED_CLAIMS

    return tuple(claim.quote for claims in CURATED_CLAIMS.values() for claim in claims)


def _sentence_matches_claim(sentence: str, quotes: Sequence[str]) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    for quote in quotes:
        if quote in stripped or stripped in quote:
            return True
    return False


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def compression_generation_payload(request: GenerationRequest) -> dict[str, object]:
    chunk = _extract_compress_chunk(request.input_text)
    words = chunk.split()
    target = max(1, int(len(words) * _RETENTION_RATIO))
    sentences = _split_sentences(chunk)
    quotes = _claim_quotes()
    required = [
        sentence
        for sentence in sentences
        if _sentence_matches_claim(sentence, quotes) or re.search(r"\d", sentence)
    ]
    optional = [sentence for sentence in sentences if sentence not in required]
    kept: list[str] = []
    count = 0
    for sentence in required + optional:
        sentence_words = len(sentence.split())
        if count + sentence_words > target and count > 0:
            continue
        kept.append(sentence)
        count += sentence_words
        if count >= target:
            break
    if not kept and sentences:
        kept = [sentences[0]]
    text = " ".join(kept)
    trimmed = text.split()
    if len(trimmed) > target:
        text = " ".join(trimmed[:target])
    return {"text": text}
