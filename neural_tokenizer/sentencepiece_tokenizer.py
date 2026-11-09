"""Lossless, version-pinned 50k BPE tokenization backed by SentencePiece.

A new vocabulary means a new model/checkpoint contract. Never silently expand
existing embeddings or overwrite an established tokenizer artifact.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class SentencePieceTokenizer:
    """UTF-8 BPE with byte fallback and immutable vocabulary fingerprint."""

    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError("Install sentencepiece to use the 50k BPE tokenizer") from exc
        self.path = Path(model_path)
        self._bytes = self.path.read_bytes()
        self._processor = spm.SentencePieceProcessor(model_proto=self._bytes)
        self.BOS_ID = self._processor.bos_id()
        self.EOS_ID = self._processor.eos_id()
        if self.BOS_ID < 0 or self.EOS_ID < 0:
            raise ValueError("SentencePiece model must define BOS and EOS")

    @property
    def vocab_size(self) -> int:
        return int(self._processor.vocab_size())

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text must be str")
        ids = self._processor.encode(text, out_type=int)
        return ([self.BOS_ID] if add_bos else []) + ids + ([self.EOS_ID] if add_eos else [])

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids if int(i) not in (self.BOS_ID, self.EOS_ID)]
        return self._processor.decode(ids)

    def model_contract(self) -> dict:
        return {"format": "sentencepiece-bpe-byte-fallback-v1",
                "vocab_size": self.vocab_size,
                "vocabulary_sha256": hashlib.sha256(self._bytes).hexdigest()}


def train_sentencepiece_bpe(corpus_paths: list[str | Path], output_prefix: str | Path,
                            *, vocab_size: int = 50_000) -> Path:
    """Train a real corpus-derived vocabulary, never pad with dummy tokens."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install sentencepiece to train the BPE tokenizer") from exc
    paths = [Path(p) for p in corpus_paths]
    if not paths or any(not p.is_file() or p.stat().st_size == 0 for p in paths):
        raise ValueError("Provide nonempty corpus files")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Disable the lossy default normalization. Byte fallback keeps uncommon
    # Unicode characters encodable; no artificial 50k placeholders.
    spm.SentencePieceTrainer.train(
        input=','.join(str(p.resolve()) for p in paths),
        model_prefix=str(prefix.resolve()), vocab_size=vocab_size, model_type='bpe',
        character_coverage=1.0, byte_fallback=True,
        normalization_rule_name='identity', add_dummy_prefix=False,
        remove_extra_whitespaces=False, split_by_whitespace=False,
        bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
        hard_vocab_limit=True, max_sentence_length=16384,
    )
    result = SentencePieceTokenizer(prefix.with_suffix('.model'))
    if result.vocab_size != vocab_size:
        raise RuntimeError(f"Expected {vocab_size} real tokens, got {result.vocab_size}")
    for sample in ('Hello 你好 🍎!', ' A  B\n', '中文和English', '👩\u200d💻'):
        if result.decode(result.encode(sample)) != sample:
            raise RuntimeError("Trained tokenizer failed exact Unicode round trip")
    return prefix.with_suffix('.model')
