# AURYVETH-NEO architecture review + experimental pretrained chat

**Status: experimental / not production qualified.** This add-on does not replace NEO's original recurrent language model. It does not merge model weights. Its only enabled pretrained generation backbone is `Qwen/Qwen3-0.6B`. RWKV-7, Mamba-2, xLSTM and Titans are architectural comparisons, not included runtime binaries. Downloaded weights are *not* bundled.

## Verdict

- AURYVETH-NEO recurrent mixer: **promising** fixed-width per-layer decode cache, with PyTorch chunked scan. **Weak** relative to pretrained quality: its weights are independently trained and cannot directly accept RWKV/Mamba checkpoint tensors.
- Memory bank: **strong** separation of provenance, memory records, versions, provisional/consolidated states and graph association metadata. **Weak** linear-scan retrieval, bounded capacity, encoder fingerprints and lack of demonstrated continual-chat reliability.
- RWKV-7: **strong** mature pretrained recurrent architecture; **weak** direct compatibility with original NEO layer state and checkpoints.
- Mamba-2: **strong** selective state space efficiency; **weak** released small models are base language models and integrating their kernels/weights is non-trivial.
- xLSTM: **strong** matrix memory recurrence; **weak** 7B model hardware/kernels.
- Qwen3-0.6B: **strong** immediately usable pretrained conversational starting point and multilingual coverage; **weak** transformer KV behavior, not recurrent, and weights do not automatically update from conversation.
- Titans: **strong** research inspiration for test-time neural adaptation; **weak** no validated plug-and-play compatibility or equivalent persistent-memory semantics.

## What the experiment can and cannot prove

**Persistent memory:** `/remember` saves an explicit user-provided fact in SQLite. A fresh Python process on the same DB can recover it and include it in context for Qwen. This is genuine cross-session text-memory persistence, **not** latent neural memory.

**Neural weight changes:** `train-adapter` opt-in supervised fine-tuning modifies Qwen3 LoRA tensors, validates that at least one trainable tensor changed, and saves an adapter plus receipt. These are **adapter neural weights**, not the frozen base-model weights. Updates are not performed implicitly during chat, and successful gradient steps do **not** prove durable, generalizable learning. Evaluate with held-out examples before relying on the adapter.

**Security:** data from saved memories is supplied as lower-trust content, not as executable system instructions. The memory is nevertheless concatenated into prompt text and may be susceptible to prompt injection. Do not save secrets. **No production security qualification.**

## Setup (Windows PowerShell)

From repository root, with Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch transformers accelerate peft safetensors
.\.venv\Scripts\python.exe -m unittest -v test_neo_chat_experimental
.\.venv\Scripts\python.exe neo_chat_experimental.py chat --model Qwen/Qwen3-0.6B
```

Requires network and sufficient disk/RAM to download Qwen weights on first run. This environment has **not** downloaded or tested Qwen weights.

Test memory:

1. Type `/remember My test phrase is silver lantern 847`.
2. Type `/new` to clear dialogue only.
3. Ask `What is my test phrase?`.
4. `/quit`, restart the Python process using the same `--db`, ask again. Expect the model to find the fact in its system-context evidence; verify what it actually says.

Create a private `examples.jsonl` file:

```jsonl
{"prompt":"What greeting should you use for me?","answer":"Hello, silver lantern."}
```

Explicitly update trainable adapter weights:

```powershell
.\.venv\Scripts\python.exe neo_chat_experimental.py train-adapter --data examples.jsonl --out runs/neo_adapter --consent-training
.\.venv\Scripts\python.exe neo_chat_experimental.py chat --adapter runs/neo_adapter
```

Never conclude general learning from one memorized training example. Compare before/after answers using held-out prompts, ensure the adapter survives restart, and measure regressions.

## Architecture gate before real integration

Do not copy RWKV, Mamba, xLSTM, and Qwen parameters into one tensor space. First freeze an explicit interchangeable backbone interface (tokenizer, shape, causal state, checkpoint provenance, adapter training contract). Qualify each implementation independently. Preserve NEO's native neural memory in its own codec namespace until adapters are trained and memory migration is validated. The current experiment is a separate chat harness, not a replacement for that architecture work.