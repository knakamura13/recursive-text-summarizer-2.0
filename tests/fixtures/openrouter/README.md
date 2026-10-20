# OpenRouter fixtures

These files were written by hand from OpenRouter's published documentation, not
recorded from live traffic. No OpenRouter request was made to produce them.

Sources, read 2026-10-19:

- https://openrouter.ai/docs/api-reference/overview (response shape, finish reasons)
- https://openrouter.ai/docs/api-reference/errors (error shape, status codes, errors inside a 200)
- https://openrouter.ai/docs/guides/features/structured-outputs (response_format)
- https://openrouter.ai/docs/guides/routing/provider-selection (provider object)
- https://openrouter.ai/docs/api/reference/limits (402 `limit_source` values)
- https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties (models listing)

Tests that load these assert the provider's request and its error mapping. They
cannot prove OpenRouter still sends this shape. Replace a file with a recorded
response, with secrets and prompt text removed, as soon as one is available.

`error_404_no_endpoint.json` is a guess. The documentation says a request with
`require_parameters: true` that no endpoint can serve fails, but not with which
status. The provider maps every 4xx other than 401, 402, 408 and 429 to a
non-retryable request error, so the test holds for any 4xx value.
