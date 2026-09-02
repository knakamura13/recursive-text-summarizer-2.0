from summarizer.text import (
    build_generation_request,
    chunk_text_by_sentences,
    normalize_whitespace,
)


def test_normalize_whitespace_collapses_all_whitespace() -> None:
    assert normalize_whitespace("  Alpha  \n\t Beta     Gamma  ") == (
        "Alpha Beta Gamma"
    )


def test_chunk_text_preserves_characterized_sentence_packing() -> None:
    chunks = chunk_text_by_sentences(
        "ignored",
        12,
        sentence_tokenizer=lambda _text: ["Alpha.", "Beta.", "Gamma."],
    )

    assert chunks == [" Alpha. Beta.", "Gamma."]


def test_chunk_text_keeps_oversized_first_sentence_unsplit() -> None:
    chunks = chunk_text_by_sentences(
        "ignored",
        4,
        sentence_tokenizer=lambda _text: ["Oversized sentence."],
    )

    assert chunks == [" Oversized sentence."]


def test_default_tokenizer_requires_no_downloaded_nltk_data(
    monkeypatch,
) -> None:
    import nltk.data

    monkeypatch.setattr(nltk.data, "path", [])

    assert chunk_text_by_sentences("First. Second.", 100) == [
        " First. Second."
    ]


def test_build_generation_request_preserves_characterized_prompt() -> None:
    request = build_generation_request(
        "Source text",
        model="gpt-4o-mini",
        timeout_seconds=45,
        operation_id="chunk-1",
    )

    assert request.model == "gpt-4o-mini"
    assert request.timeout_seconds == 45
    assert request.operation_id == "chunk-1"
    assert request.instructions == (
        "You are a writing assistant, skilled in revising and summarizing "
        "complex technical writing with accuracy and precision."
    )
    assert request.input_text == (
        "Provide an executive summary of the following text (delimited by "
        "triple quotes). Present the key ideas and findings directly, without "
        "bullet points, as if for a busy professional who needs to grasp the "
        "essential points quickly. Ignore complete sentences and grammatical "
        "correctness. Abbreviate long and repetitive words. "
        '\n"""\nSource text\n"""\n'
    )
