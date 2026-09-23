# Verified content-unit fallback

## Problem

The final editor can add details that the source excerpt does not establish. Verification correctly withholds those drafts, but repeated short-prompt runs did not produce a publishable summary of the Snow Fall sample.

## Considered approaches

- Prompt changes alone: small implementation, but the measured 20-, 50-, and 80-word drafts still contained unsupported claims.
- Verify root content units, then ask the editor to rewrite only accepted units: preserves fluent prose but gives the editor another opportunity to add claims.
- Verify root content units and assemble accepted prose directly: less editorial polish, but each unit and the complete output can be checked before publication.

The third approach is used as a fallback after the normal editorial draft fails verification.

## Flow and bounds

Screen at most eight root content units, omitting those marked uncertain. A verifier failure on one unit does not stop the screen. Keep a unit when every claim in it is supported. When a unit mixes supported and unsupported fragments and none are contradicted, keep an original sentence only when every claim in that sentence is supported. Do not join leftover fragments into a new sentence. A shorter sentence is published only if that sentence passes the same all-claims check. Adding a candidate must leave the assembled draft supported. An insufficient fragment of a sentence already kept does not block a new sentence whose own claims are supported. A new insufficient fragment, a fallback failure, or a contradiction still rejects the addition. Stop when the assembled draft reaches the requested word target. Verify the selected draft again through the standard final verification path before publication. If that check fails, drop each sentence that contains an unsupported or contradicted claim and check the remaining sentences again. Publish only a remainder that passes. If no unit survives or no remainder passes, withhold output and retain an audit artifact. Mark a successful fallback in audit warnings.

The fallback retains citation behavior from the final verification pass when enabled, using verifier evidence segments instead of whole-document provenance when those segments exist. It can produce a shorter result than requested; the UI marks completed runs whose published word count is below the configured target.

## Validation

The Snow Fall frontend run `09c4d9c0-b478-4c2e-8c04-691bf08de729` completed with local Qwen 3.6 and claim verification enabled. Its published one-sentence summary passed final verification and was manually checked against the sample excerpt. The audit records `verified_content_unit_fallback`. The result demonstrates a grounded short summary, not comprehensive coverage of the article.
