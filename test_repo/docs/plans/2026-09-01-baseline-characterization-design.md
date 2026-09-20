# Baseline Characterization Harness Design

Issue: [#2: Characterize the existing summarizer and add a baseline test harness](https://github.com/knakamura13/recursive-text-summarizer-2.0/issues/2)

## Context

The current application keeps file handling, sentence chunking, OpenAI access,
retry behavior, progress reporting, and executable orchestration in main.py.
Importing the module downloads NLTK data and constructs an OpenAI client. The
executable splits text by an approximate character limit, summarizes each chunk,
and joins the results without a final synthesis step.

Before changing those behaviors, the project needs a deterministic offline
baseline. The baseline must expose both useful compatibility constraints and
legacy defects that later issues are expected to replace.

## Decision

Build a no-production-change characterization harness. Tests will load the
existing application with deterministic replacements for its external modules
and services. This preserves the current executable as the subject under test
while preventing network calls, credential requirements, tokenizer downloads,
and nondeterministic model responses.

This approach is preferred over adding injection seams to production code
because provider and CLI refactoring belongs to #3. It is preferred over a fake
HTTP service because the current import-time NLTK and OpenAI behavior would
still need interception, and transport fidelity is not the purpose of this
baseline.

## Components

### Legacy module loader

A test helper will install controlled module doubles before importing
main.py, then restore interpreter state afterward. Each test receives a fresh
module instance so mutable globals and fake-provider call history cannot leak
between cases.

The doubles cover:

- OpenAI client construction and chat-completion requests
- NLTK download requests and sentence tokenization
- dotenv loading
- progress iteration

The real requests exception types may remain available for retry
characterization, but no request is allowed to reach the network.

### Scripted fake model

The fake OpenAI client records the requested model, messages, timeout, and call
order. A test can queue successful responses or exceptions. Exhausting the
script is an explicit test failure rather than an implicit default response.

### Network guard

An automatic test fixture will reject socket connection attempts. This makes
the offline requirement executable and prevents a future import or test path
from silently using the network.

### Fixture corpus

Compact source fixtures will cover an article, report, transcript, structured
Markdown document, and narrative prose. They are designed to exercise genre
variation rather than evaluate summary quality. Existing historical lecture
data remains unchanged.

### Baseline record

A short baseline document will describe the observed workflow and separate
compatibility constraints from known limitations. It will include the command
for running the offline suite.

## Data Flow

For unit characterization, a test loads a fresh legacy module, scripts sentence
or model results, invokes one public function, and asserts its return value and
recorded side effects.

For executable characterization, a test changes into a temporary directory,
writes input.txt, loads the external-module doubles, and executes main.py
as the main module. The fake tokenizer supplies deterministic sentences, the fake
model returns one response per chunk, and the test verifies call order,
prompt/model/timeout values, logging side effects, and the joined
\`output.txt\`.

Tests of retry behavior replace sleeping with a recorder. Scripted exceptions
therefore exercise retry count and exponential delays without slowing the
suite.

## Error Handling

The loader fails clearly if a required external-module access was not stubbed.
The fake provider fails clearly when calls differ from its configured script.
The network guard fails on attempted connections.

Characterization tests record current application behavior even when that
behavior is undesirable, including returning provider failures as summary text
and catching the executable's fatal exception without re-raising it. These
expectations will be labeled as legacy limitations so later issues can change
them deliberately.

## Test Coverage

The offline suite will characterize:

- UTF-8 file reads and writes, including invalid output paths
- whitespace collapsing
- sentence packing at, below, and above the character limit
- an oversized single sentence
- default and aliased model names
- prompt content, delimiter handling, timeout, and provider call order
- successful completion and logged prompt/response files
- retry count and exponential delays
- current provider-error return values
- MAX_CHUNKS truncation
- default input.txt to output.txt execution
- per-chunk response concatenation
- representative fixture loading
- absence of external network access

## Non-goals

This slice will not modernize the provider API, introduce production dependency
injection, change prompts, replace character chunking, add recursive merging,
or correct legacy error handling. Those changes begin with #3 and are judged
against the accepted baseline rather than being mixed into its creation.
