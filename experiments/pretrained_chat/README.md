# AURYVETH-NEO — optional pretrained chat experiment

**Experimental; not a merged foundation model.** This directory provides a Qwen3 chat harness with durable explicit memories and a separate, opt-in LoRA fine-tuning command. No model weights are bundled. RWKV-7, Mamba-2, and xLSTM remain separate, unintegrated research candidates.

## On Windows

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch transformers accelerate peft safetensors
cd experiments/pretrained_chat
..\..\.venv\Scripts\python.exe -m unittest -v test_neo_chat_experimental
..\..\.venv\Scripts\python.exe neo_chat_experimental.py chat
```

During chat, enter `/remember My project code is violet 739`. Use `/new` to clear conversational history but keep memories. Exit and restart chat from the same directory to check persistence. Data is written under `data/neo_memories.sqlite3` and should not be committed. The persistence test passes locally (5/5); actual pretrained generation has **not** been run in the review environment.

To train a neural LoRA adapter explicitly, create a private JSONL file with `{"prompt":"...","answer":"..."}` rows and run:

```powershell
..\..\.venv\Scripts\python.exe neo_chat_experimental.py train-adapter --data examples.jsonl --consent-training
..\..\.venv\Scripts\python.exe neo_chat_experimental.py chat --adapter runs/neo_adapter
```

This changes adapter weights rather than base-model weights. It has not been validated on pretrained Qwen here. Saving a fact in SQLite **does not** train weights. The original AURYVETH-NEO neural memory architecture is unchanged and there is no automatic background training.

See [design and evidence assessment](../../docs/reviews/NEO-PRETRAINED-BACKBONE-ASSESSMENT-20261110.md).
