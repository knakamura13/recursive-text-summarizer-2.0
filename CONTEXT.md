# Recursive Text Summarizer

A local tool that produces source-grounded, claim-verified summaries of documents of any length.

## Language

### Documents

**Document**:
A file or pasted text in the library, held as the text extracted from it.
_Avoid_: File, source

**Import**:
The process that turns an uploaded file or pasted text into a Document, including OCR of scanned pages and images.
_Avoid_: Upload, ingestion

**Import report**:
The result of an Import shown on its Document: page count, OCR'd pages, blank pages, and a text preview.
_Avoid_: Extraction review, confirmation

### Runs

**Run**:
One summarization of one Document with one configuration.
_Avoid_: Task, job, summary

**Attempt**:
One execution of a Run. Resuming a Run starts a new Attempt of the same Run.
_Avoid_: Retry, rerun

**Stop**:
Halting an active Run while keeping its finished work.
_Avoid_: Cancel, abort, pause

**Resume**:
Continuing a stopped, failed, or interrupted Run as a new Attempt that reuses its finished work.
_Avoid_: Retry, restart
