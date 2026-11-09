# AURYVETH-NEO

A trainable recurrent language model with persistent neural memory and an optional switch planner. Ordinary chat returns the model’s decoded output. The Ollama CLI connects to a local PyTorch service for inference.

## Install

From the repository root, using Python 3.11 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,data]"
.\.venv\Scripts\python.exe -m pytest -q
```

The `data` extra is needed for the OpenAssistant builder. The trainer supports CPU and CUDA through `--device auto`; CUDA requires a CUDA-enabled PyTorch installation.

## Train a recurrent chat model

Start with a UTF-8 training-only text corpus at `data/train.txt`. Train a tokenizer once and use the same file when preparing dialogue, training, and serving. The dialogue builder downloads pinned OpenAssistant data and holds out separate conversation trees. You can also supply your own train and validation JSONL files, one object per line, with `prompt`, `answer`, and `context` fields. Dialogue prompts use alternating `User:` and `Assistant:` lines and end with `Assistant: `. Use an empty `context` array for ordinary chat.

```powershell
.\.venv\Scripts\python.exe -m scripts.train_tokenizer --input data/train.txt --prefix data/tokenizer_50k
.\.venv\Scripts\python.exe -m scripts.build_conversation_data --tokenizer data/tokenizer_50k.model --out data/dialogue
.\.venv\Scripts\python.exe -m scripts.train --backbone recurrent --width 256 --layers 4 --heads 8 --kv-heads 2 --block-size 1024 --train-seq-len 512 --tokenizer data/tokenizer_50k.model --train-jsonl data/dialogue/train.jsonl --validation-jsonl data/dialogue/validation.jsonl --steps 2000 --out runs/chat
```

The trainer can mix pretraining text and supervised data. Add `--corpus data/train.txt --validation-corpus data/validation.txt` with a separate UTF-8 validation corpus. Each checkpoint includes the exact SentencePiece tokenizer and a hash manifest. `model.pt` is for inference, `best_model.pt` records the lowest validation loss, and `training_state.pt` stores the optimizer and random-number state.

Use `--stop-after` to pause a planned run. To continue it, repeat the same command and add `--resume-state runs/chat/training_state.pt` in place of `--stop-after`, with the original total `--steps`. To start a new fine-tuning run, use `--resume-checkpoint path/to/model.pt`, its matching tokenizer, and a new output directory.

Training from scratch at this size is experimental. A successful run verifies the training pipeline. Generated replies can be blank, repetitive, or inaccurate.

## Chat and Ollama

Run directly in Python:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.chat --checkpoint runs/chat/model.pt --tokenizer runs/chat/tokenizer.model --manifest runs/chat/manifest.json
```

With Ollama installed, build a local bundle and launch:

```powershell
.\.venv\Scripts\python.exe -m scripts.neo_ollama --build --chat-manifest runs/chat/manifest.json
.\Run-AURYVETH-NEO.ps1
```

The launcher connects the Ollama CLI to the PyTorch service on loopback port 11435. The custom checkpoint requires this service. For a manually managed service, run `python -m scripts.neo_ollama` in the project environment and set `OLLAMA_HOST=http://127.0.0.1:11435` in the client shell.

To train the optional structured switch planner:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_switch_reasoning --baseline runs/chat/manifest.json --out runs/switch --steps 660
```

The planner requires distinct single-token representations for `off`, `on`, `关`, and `开`. If its manifest reports `cli_qualified: true`, build a new bundle using the same chat manifest and `--think-manifest runs/switch/manifest.json`. Then `think: off toggle keep` displays the model’s predicted switch states. This capability is scoped to structured switch commands.

The repository contains source and tests. Build checkpoints and prepare datasets locally.

## License

Apache 2.0. See [LICENSE](LICENSE).
