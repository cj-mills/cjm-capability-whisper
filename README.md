# cjm-capability-whisper

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

An OpenAI Whisper speech-to-text capability for the cjm-substrate runtime that provides local transcription with configurable model selection and parameter control.

## Modules

- **`cjm_capability_whisper.capability`**

## API

### `cjm_capability_whisper.capability`

- `WhisperCapabilityConfig` _class_ — Configuration for Whisper transcription capability.
- `WhisperLocalCapability` _class_ — OpenAI Whisper transcription capability (stage 8: pure-compute tool capability).

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-substrate`, `torch-utils`
