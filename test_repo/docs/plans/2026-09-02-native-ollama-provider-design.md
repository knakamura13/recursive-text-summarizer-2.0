# Native Ollama Provider Design

## Context

Issue #3 now includes local inference through Ollama. The existing provider
boundary already keeps orchestration independent of OpenAI, so this addition
should extend provider composition without changing the transitional workflow.
The parent goal remains efficient, low-signal-loss summarization of extremely
long artifacts. Local models make repeated hierarchical calls practical without
per-call API cost, while provider metadata keeps later throughput and
compression evaluation possible.

## Goals

- Support Ollama through its native API as a first-class model provider.
- Keep the workflow and retry decorator provider-neutral.
- Allow explicit provider, model, and Ollama host selection from the CLI.
- Preserve the existing no-argument OpenAI workflow.
- Capture Ollama token counts and completion metadata when available.
- Fail with actionable, sanitized application exceptions.
- Keep imports, dry runs, and automated tests free of network activity.

## Non-goals

- Streaming generation.
- Model installation, pulling, or lifecycle management.
- Automatic model discovery or selection.
- Provider-specific prompting or context-budget policy.
- Structured intermediate summaries or hierarchical traversal.
- Remote Ollama authentication and cloud account management.

## Alternatives considered

### Ollama's OpenAI-compatible endpoint

This would reuse portions of the OpenAI adapter, but it would hide native token,
duration, thinking, and completion fields behind a compatibility layer. It also
would make provider errors harder to classify accurately.

### Handwritten native HTTP transport

Calling `/api/chat` directly would avoid another package, but the application
would own request serialization, response decoding, connection behavior, and
client compatibility. That transport code would add maintenance without
improving the summarization boundary.

### Official native Python client

The selected approach uses Ollama's official Python client and native chat API.
It provides typed client construction and native response values while keeping
the adapter narrow and injectable for tests.

## Configuration and composition

Provider selection is a validated value with `openai` as the default. The CLI
adds `--provider {openai,ollama}` and `--ollama-host`, whose default is
`http://localhost:11434`. The existing unrestricted `--model` value supports
installed tags such as `qwen3.8`, `gemma3:4b`, and custom models.

The CLI composition layer selects and constructs the provider. It then applies
the existing `RetryingProvider` exactly once. `LegacyWorkflow` continues to
receive only a `ModelProvider` and contains no Ollama branch.

Tests retain provider-factory injection. Provider selection is represented by a
small factory function so CLI tests can verify routing without constructing a
real client.

## Native request mapping

`OllamaProvider` constructs `ollama.Client(host=...)` lazily on its first
generation call. It invokes the native chat API with:

- the request model;
- a system message containing `GenerationRequest.instructions`;
- a user message containing `GenerationRequest.input_text`;
- `stream=False` so each application request has one terminal response;
- `think=False` to avoid unnecessary reasoning output and latency for the
  current plain-text summarization contract.

The request timeout is enforced by the client transport configuration rather
than added to the native chat payload. Client construction remains injectable
so unit tests do not contact a local service.

## Response mapping

The adapter accepts only a terminal response with non-empty assistant content.
It normalizes that response into `GenerationResult`:

- `message.content` becomes `text`;
- response `model` becomes the resolved model, falling back to the request;
- `prompt_eval_count` becomes `input_tokens`;
- `eval_count` becomes `output_tokens`;
- `done_reason` becomes `finish_status`;
- `request_id` remains absent because the local native response has no request
  identifier.

Ollama duration fields remain native adapter details for now because
`GenerationResult` has no provider-neutral latency fields. A later metrics
issue can add duration data consistently across providers.

## Failure policy

The adapter translates failures into the existing exception hierarchy:

- transport timeouts become `ProviderTimeoutError`;
- connection failures become `ProviderConnectionError`;
- HTTP 404 or a model-not-found response becomes a non-retryable
  `ProviderRequestError` with an actionable model message;
- HTTP 429 becomes `ProviderRateLimitError`;
- HTTP 500 and 502 become `ProviderServerError`;
- other rejected requests become `ProviderRequestError`;
- missing terminal state, missing content, or malformed response values become
  `ProviderResponseError`.

Messages identify Ollama and the relevant category without including prompts or
source text. The existing retry decorator retries only the translated transient
types.

## Testing and documentation

Tests use fake client factories and response objects to cover exact request
mapping, lazy construction, metadata extraction, malformed responses, and error
translation. CLI tests cover defaults, overrides, provider selection, dry-run
isolation, and unavailable local-service reporting. Import-safety tests include
the new adapter. All automated tests remain offline.

The README documents installation as an external prerequisite, the standard
host, model selection, an Ollama command example, and equivalent OpenAI and
Ollama summarizer invocations. A manual local smoke test is documented but is
not required by the deterministic suite.

## Future compatibility

The provider protocol and workflow remain unchanged. Later hierarchical stages
can choose OpenAI or Ollama without transport branches, and later model-profile
work can attach provider-specific context limits and capability metadata outside
this adapter.
