# NEO controlled experiential learning lifecycle — architecture decision

Status: **design decision; implementation NOT STARTED** (2026-11-10).

## Owner decision
- During ordinary conversation, capture durable experience candidates without changing model parameters.
- Accumulate, deduplicate, prioritize and qualify experiences.
- Run controlled learning in a distinct phase, including a best-effort bounded phase before **planned** shutdown.
- Preserve a stable active checkpoint until candidate training, validation, atomic checkpoint commit and versioned activation succeed.
- Preserve experiences across crashes; never promise training on abrupt termination.

## State machine
`AWAKE -> QUEUED -> LEARNING -> VALIDATING -> COMMITTING -> READY_TO_SHUTDOWN`
Failure/cancellation: `LEARNING|VALIDATING|COMMITTING -> ROLLBACK -> READY_TO_SHUTDOWN`.
On power loss or forced termination, recover the last validated checkpoint and replayable experience queue. Checkpoint commit must not depend on partial output directories.

## Contracts to freeze before implementation
1. **Experience envelope**: immutable ID, timestamp, provenance, trust classification, scope/consent, content digest, retention choice, and revision.
2. **Learning candidate**: exactly named base checkpoint/version, tokenizer/codec identifiers, replay sample provenance, optimizer state, budget, model/adapter target, and reversible candidate artifacts.
3. **Validation gate**: held-out new-knowledge evaluation, previous-knowledge regression, general chat benchmark, safety/malicious-input controls, persistence/restart verification, trainable parameter checksum change, and rollback drill.
4. **Commit protocol**: candidate stored separately, manifest and hashes checked, atomic active-pointer switch only after validation, prior checkpoint retained.
5. **Shutdown contract**: bounded by a user-configured resource/time budget; poweroff may proceed with pending experiences untrained, and must never block indefinitely.
6. **Boundary**: SQLite notes/external retrieval are not neural weight training; LoRA updates are trainable adapter weights, not frozen base weights. Any proposed in-model memory or base-weight mutation must be independently qualified.
7. **Governance**: host controls training policy and can disable or cancel; user content is low-trust evidence, never system instructions.

## Required tests (not yet run)
- Capture/deduplication and queue durability across restart.
- No weight change during chat; changed trainable tensor checksum after controlled learning.
- Crash before, during and after checkpoint atomic commit; deterministic rollback.
- Fault-injected resource exhaustion, OOM, timeout and user cancellation.
- Tokenizer/codec mismatch must fail closed without corrupting old memories.
- Evaluation leakage prevention and held-out retention/regression acceptance.
- End-to-end restart demonstrates memory retrieval versus parameter-learned knowledge separately.

## Implementation boundary
This document changes **architecture only**. Existing pretrained chat experiment and the original recurrent implementation must not be presented as already implementing lifecycle hooks or pre-shutdown training. Do not merge as production-ready until demonstrated on actual checkpoints.
