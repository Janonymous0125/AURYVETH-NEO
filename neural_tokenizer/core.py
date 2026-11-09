"""Layered tokenizer research prototype (pure Python, no neural network dependencies).

Base byte IDs 0..255, BOS 256 and EOS 257 match Modular LLM v0.2.
Higher-level word/concept/sentence IDs are namespaced and never mixed into
next-token model IDs. All learned registries are append-only except for
explicit candidate-status changes; model weights are never touched here.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Iterable


class RegistryError(ValueError):
    """Saved registry is corrupt, incompatible or unsupported."""


class ContractMismatch(ValueError):
    """Tokenizer LM vocabulary differs from a model's recorded contract."""


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(obj: object) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


# Word recognition is deliberately lexical, not semantic word segmentation.
# Keeps all input characters including punctuation, whitespace, emoji ZWJ and
# combining codepoints; recognized units are annotations, never lossy text edits.
_UNIT_PATTERN = re.compile(
    r"\s+|[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[\u3400-\u9fff]+|[^\w\s]+|[^\W_]+|_",
    re.UNICODE,
)
_EN_RELATION = re.compile(
    r"\b(?P<subject>[A-Za-z][A-Za-z0-9_\-]{0,63})\s+is\s+an?\s+"
    r"(?P<object>[A-Za-z][A-Za-z0-9_\-]{0,63})\b", re.IGNORECASE,
)
_ZH_RELATION = re.compile(
    r"(?P<subject>[\u3400-\u9fff]{1,12}?)是(?:一[种個个]|[种個个])?"
    r"(?P<object>[\u3400-\u9fff]{1,12})",
)
_SENTENCE_END = set(".!?。！？\n")


def _new_state() -> dict:
    return {
        "schema": 1,
        "next_token_id": 258,
        "tokens": [],
        "merges": [],
        "literals": [],
        "next_word_id": 0,
        "words": [],
        "next_sentence_id": 0,
        "sentences": [],
        "next_concept_id": 0,
        "concepts": [],
        "next_relation_id": 0,
        "relations": [],
        "candidates": [],
    }


class LayeredTokenizer:
    """Byte/BPE tokens + independent word/concept/sentence annotation registries.

    `encode` and `analyze` are read-only; `observe`, `train_bpe`, and explicit
    registration methods are controlled learning operations. Model vocabulary
    changes require migration, NOT an in-place change to an existing checkpoint.
    """

    BOS_ID = 256
    EOS_ID = 257
    MAX_TEXT_BYTES = 4_000_000
    MAX_LITERAL_BYTES = 256

    def __init__(self, state: dict | None = None) -> None:
        self._state = copy.deepcopy(state) if state is not None else _new_state()
        self._reindex_and_validate()

    def _reindex_and_validate(self) -> None:
        s = self._state
        try:
            if not isinstance(s, dict) or s["schema"] != 1:
                raise RegistryError("Unsupported registry schema")
            if set(s) != set(_new_state()):
                raise RegistryError("Registry has missing or unexpected fields")
            self._token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
            for expected, entry in enumerate(s["tokens"], start=258):
                if entry["id"] != expected or entry["kind"] not in ("bpe", "literal"):
                    raise RegistryError("Noncontiguous token ID or invalid kind")
                v = bytes.fromhex(entry["bytes_hex"])
                if not v or len(v) > self.MAX_LITERAL_BYTES * 4:
                    raise RegistryError("Invalid token bytes")
                self._token_bytes[expected] = v
            if s["next_token_id"] != 258 + len(s["tokens"]):
                raise RegistryError("Invalid next_token_id")
            self._bytes_to_id: dict[bytes, int] = {}
            for k, v in self._token_bytes.items():
                if v in self._bytes_to_id:
                    raise RegistryError("Duplicate token byte sequences")
                self._bytes_to_id[v] = k
            self._merge_pairs: set[tuple[int, int]] = set()
            for m in s["merges"]:
                a, b, out = m["left"], m["right"], m["out"]
                if not all(type(i) is int and i in self._token_bytes for i in (a, b, out)):
                    raise RegistryError("Invalid merge references")
                pair = (a, b)
                if pair in self._merge_pairs or self._token_bytes[a] + self._token_bytes[b] != self._token_bytes[out]:
                    raise RegistryError("Invalid or duplicate merge")
                self._merge_pairs.add(pair)
            self._literals = {}
            for i in s["literals"]:
                if type(i) is not int or i < 258 or i not in self._token_bytes:
                    raise RegistryError("Invalid literal token ID")
                self._literals[self._token_bytes[i]] = i
            if len(self._literals) != len(s["literals"]):
                raise RegistryError("Duplicate literals")
            self._sorted_literals = sorted(self._literals, key=lambda x: (-len(x), x))
            for collection, next_key, identity in (
                ("words", "next_word_id", "surface"),
                ("sentences", "next_sentence_id", "text"),
                ("concepts", "next_concept_id", "key"),
                ("relations", "next_relation_id", "key"),
            ):
                entries = s[collection]
                if s[next_key] != len(entries):
                    raise RegistryError(f"Invalid {next_key}")
                if [e["id"] for e in entries] != list(range(len(entries))):
                    raise RegistryError(f"Invalid IDs in {collection}")
                if len({e[identity] for e in entries}) != len(entries):
                    raise RegistryError(f"Duplicate entries in {collection}")
            if len({e["key"] for e in s["candidates"]}) != len(s["candidates"]):
                raise RegistryError("Duplicate candidate keys")
            if any(e["status"] not in ("pending", "approved", "rejected") for e in s["candidates"]):
                raise RegistryError("Invalid candidate status")
            self._word_index = {e["surface"]: e for e in s["words"]}
            self._sentence_index = {e["text"]: e for e in s["sentences"]}
            self._concept_index = {e["key"]: e for e in s["concepts"]}
            self._relation_index = {e["key"]: e for e in s["relations"]}
            self._candidate_index = {e["key"]: e for e in s["candidates"]}
        except (KeyError, TypeError, AttributeError, IndexError, ValueError) as e:
            raise RegistryError(f"Invalid registry: {e}") from e

    @property
    def vocab_size(self) -> int:
        return self._state["next_token_id"]

    def _text_bytes(self, text: str) -> bytes:
        if not isinstance(text, str):
            raise TypeError("Text must be str, not bytes")
        raw = text.encode("utf-8", errors="strict")
        if len(raw) > self.MAX_TEXT_BYTES:
            raise ValueError("Text exceeds per-operation size limit")
        return raw

    def _apply_merges(self, ids: list[int]) -> list[int]:
        for m in self._state["merges"]:
            if len(ids) < 2:
                break
            left, right, out = m["left"], m["right"], m["out"]
            merged = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == left and ids[i + 1] == right:
                    merged.append(out)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def _encode_data(self, raw: bytes) -> list[int]:
        if not self._sorted_literals:
            return self._apply_merges(list(raw))
        out: list[int] = []
        start = cursor = 0
        while cursor < len(raw):
            match = next((v for v in self._sorted_literals if raw.startswith(v, cursor)), None)
            if match is None:
                cursor += 1
                continue
            if start < cursor:
                out.extend(self._apply_merges(list(raw[start:cursor])))
            out.append(self._literals[match])
            cursor += len(match)
            start = cursor
        if start < len(raw):
            out.extend(self._apply_merges(list(raw[start:])))
        return out

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._encode_data(self._text_bytes(text))
        return ([self.BOS_ID] if add_bos else []) + ids + ([self.EOS_ID] if add_eos else [])

    def decode(self, ids: Iterable[int], *, errors: str = "strict") -> str:
        chunks: list[bytes] = []
        for i in ids:
            if type(i) is not int:
                raise ValueError("Token IDs must be integers")
            if i in (self.BOS_ID, self.EOS_ID):
                continue
            if i not in self._token_bytes:
                raise ValueError(f"Unknown model token ID {i}")
            chunks.append(self._token_bytes[i])
        return b"".join(chunks).decode("utf-8", errors=errors)

    def register_literal_token(self, text: str) -> int:
        """EXPLICIT model-vocabulary growth. Requires downstream model migration."""
        raw = self._text_bytes(text)
        if not raw or len(raw) > self.MAX_LITERAL_BYTES:
            raise ValueError("Literal token length must be 1..256 UTF-8 bytes")
        id_ = self._bytes_to_id.get(raw)
        if id_ is None:
            id_ = self._state["next_token_id"]
            self._state["next_token_id"] += 1
            self._state["tokens"].append({"id": id_, "kind": "literal", "bytes_hex": raw.hex()})
            self._token_bytes[id_] = raw
            self._bytes_to_id[raw] = id_
        if id_ >= 258 and id_ not in self._state["literals"]:
            self._state["literals"].append(id_)
            self._literals[raw] = id_
            self._sorted_literals = sorted(self._literals, key=lambda x: (-len(x), x))
        return id_

    def train_bpe(self, texts: Iterable[str], *, max_merges: int = 100,
                  min_frequency: int = 2) -> int:
        """Learn append-only BPE pairs on supplied corpus; never change existing IDs.

        Note: new merges CAN change the tokenization of previously encoded text.
        Do not modify an active model's tokenizer without model-data migration.
        """
        if type(max_merges) is not int or not 0 <= max_merges <= 10_000:
            raise ValueError("max_merges must be an integer in 0..10000")
        if type(min_frequency) is not int or min_frequency < 2:
            raise ValueError("min_frequency must be >=2")
        sequences = [self._encode_data(self._text_bytes(t)) for t in texts]
        if sum(len(x) for x in sequences) > 2_000_000:
            raise ValueError("Training text exceeds size limit")
        added = 0
        for _ in range(max_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            eligible = [(pair, count) for pair, count in counts.items()
                        if count >= min_frequency and pair not in self._merge_pairs]
            if not eligible:
                break
            (left, right), freq = min(eligible, key=lambda e: (-e[1], e[0]))
            raw = self._token_bytes[left] + self._token_bytes[right]
            if len(raw) > self.MAX_LITERAL_BYTES * 4:
                break
            out = self._bytes_to_id.get(raw)
            if out is None:
                out = self._state["next_token_id"]
                self._state["next_token_id"] += 1
                self._state["tokens"].append({"id": out, "kind": "bpe", "bytes_hex": raw.hex()})
                self._token_bytes[out] = raw
                self._bytes_to_id[raw] = out
            self._state["merges"].append({"left": left, "right": right, "out": out})
            self._merge_pairs.add((left, right))
            added += 1
            for j, seq in enumerate(sequences):
                merged = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
                        merged.append(out)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                sequences[j] = merged
        return added

    @staticmethod
    def _words(text: str) -> list[str]:
        units = [m.group() for m in _UNIT_PATTERN.finditer(text)]
        if "".join(units) != text:
            raise RuntimeError("Lexical partition must be lossless")
        return units

    @staticmethod
    def _sentences(text: str) -> list[str]:
        chunks = []
        start = 0
        for i, ch in enumerate(text):
            if ch in _SENTENCE_END:
                chunks.append(text[start:i + 1])
                start = i + 1
        if start < len(text):
            chunks.append(text[start:])
        return chunks

    @staticmethod
    def _is_memorable_unit(unit: str) -> bool:
        return any(unicodedata.category(ch)[0] in ("L", "N")
                   or unicodedata.category(ch) == "So" for ch in unit)

    @staticmethod
    def _relation_hypotheses(text: str) -> list[dict]:
        found = []
        for pattern in (_EN_RELATION, _ZH_RELATION):
            for m in pattern.finditer(text):
                subj, obj = m.group("subject"), m.group("object")
                key = _digest([subj.casefold(), "is_a", obj.casefold()])
                found.append({"key": key, "subject": subj, "relation": "is_a", "object": obj})
        return found

    def analyze(self, text: str) -> dict:
        """Lossless four-level analysis; annotations are not LM input IDs.

        BPE tokens can cross word/sentence boundaries, so alignments are
        explicit many-to-many byte-span overlaps, not assumed tree edges.
        """
        raw = self._text_bytes(text)
        tokens = []
        position = 0
        ids = self._encode_data(raw)
        for i in ids:
            b = self._token_bytes[i]
            tokens.append({"id": i, "start_byte": position, "end_byte": position + len(b),
                           "bytes_hex": b.hex()})
            position += len(b)
        if position != len(raw):
            raise AssertionError("Token byte spans are not lossless")

        def annotate(units: list[str], index: dict, key: str) -> list[dict]:
            result = []
            start_char = start_byte = 0
            for unit in units:
                end_char = start_char + len(unit)
                end_byte = start_byte + len(unit.encode("utf-8"))
                result.append({
                    "text": unit,
                    "id": index[unit]["id"] if unit in index else None,
                    "start_char": start_char, "end_char": end_char,
                    "start_byte": start_byte, "end_byte": end_byte,
                    "token_indices": [j for j, t in enumerate(tokens)
                                      if t["start_byte"] < end_byte and t["end_byte"] > start_byte],
                })
                start_char, start_byte = end_char, end_byte
            return result

        words = annotate(self._words(text), self._word_index, "surface")
        sentences = annotate(self._sentences(text), self._sentence_index, "text")
        hypotheses = self._relation_hypotheses(text)
        for h in hypotheses:
            c = self._candidate_index.get(h["key"])
            h["status"] = c["status"] if c else "unregistered"
            if c is not None and c["status"] == "approved":
                rel = self._relation_index[h["key"]]
                h["relation_id"] = rel["id"]
                h["subject_concept_id"] = rel["subject_concept_id"]
                h["object_concept_id"] = rel["object_concept_id"]
        return {"token_ids": ids, "tokens": tokens, "words": words,
                "sentences": sentences, "concept_hypotheses": hypotheses}

    def observe(self, text: str, *, source: str, remember_sentences: bool = False) -> dict:
        """Learn lexical units; discover, but DO NOT assert, relational concepts."""
        self._text_bytes(text)
        if not isinstance(source, str) or not source.strip() or len(source) > 256:
            raise ValueError("Attributable source ID required (1..256 chars)")
        for unit in self._words(text):
            if not self._is_memorable_unit(unit):
                continue
            existing = self._word_index.get(unit)
            if existing is None:
                existing = {"id": self._state["next_word_id"], "surface": unit,
                            "count": 0, "sources": []}
                self._state["next_word_id"] += 1
                self._state["words"].append(existing)
                self._word_index[unit] = existing
            existing["count"] += 1
            if source not in existing["sources"]:
                existing["sources"].append(source)
        if remember_sentences:
            for sentence in self._sentences(text):
                if sentence.strip():
                    self.register_sentence(sentence)
        for h in self._relation_hypotheses(text):
            c = self._candidate_index.get(h["key"])
            if c is None:
                c = {**h, "sources": [], "status": "pending", "reviewer": None,
                     "decision_reason": None}
                self._state["candidates"].append(c)
                self._candidate_index[c["key"]] = c
            if source not in c["sources"]:
                c["sources"].append(source)
        return self.analyze(text)

    def register_sentence(self, text: str) -> int:
        self._text_bytes(text)
        if not text:
            raise ValueError("Cannot register an empty sentence")
        entry = self._sentence_index.get(text)
        if entry is not None:
            return entry["id"]
        id_ = self._state["next_sentence_id"]
        self._state["next_sentence_id"] += 1
        entry = {"id": id_, "text": text}
        self._state["sentences"].append(entry)
        self._sentence_index[text] = entry
        return id_

    @staticmethod
    def _concept_key(term: str) -> str:
        return term.casefold()

    def _ensure_concept(self, term: str) -> int:
        key = self._concept_key(term)
        existing = self._concept_index.get(key)
        if existing is not None:
            return existing["id"]
        id_ = self._state["next_concept_id"]
        self._state["next_concept_id"] += 1
        entry = {"id": id_, "key": key, "label": term}
        self._state["concepts"].append(entry)
        self._concept_index[key] = entry
        return id_

    def approve_candidate(self, key: str, *, approver: str = "") -> int:
        c = self._candidate_index.get(key)
        if c is None:
            raise KeyError("Unknown concept candidate")
        if c["status"] == "rejected":
            raise ValueError("Rejected candidates cannot be approved implicitly")
        if not isinstance(approver, str) or not approver.strip():
            raise ValueError("Explicit named approval is required")
        if len(c["sources"]) < 2:
            raise ValueError("At least two distinct attributed sources are required")
        subj = self._ensure_concept(c["subject"])
        obj = self._ensure_concept(c["object"])
        entry = self._relation_index.get(key)
        if entry is None:
            id_ = self._state["next_relation_id"]
            self._state["next_relation_id"] += 1
            entry = {"id": id_, "key": key, "subject_concept_id": subj,
                     "relation": c["relation"], "object_concept_id": obj,
                     "sources": list(c["sources"]), "approved_by": approver}
            self._state["relations"].append(entry)
            self._relation_index[key] = entry
        c["status"] = "approved"
        c["reviewer"] = approver
        return entry["id"]

    def reject_candidate(self, key: str, *, reviewer: str, reason: str) -> None:
        c = self._candidate_index.get(key)
        if c is None:
            raise KeyError("Unknown concept candidate")
        if c["status"] == "approved":
            raise ValueError("Cannot silently reject a registered relation")
        if not reviewer or not reason:
            raise ValueError("Reviewer and rejection reason required")
        c["status"] = "rejected"
        c["reviewer"] = reviewer
        c["decision_reason"] = reason

    def word_entries(self) -> list[dict]:
        return copy.deepcopy(self._state["words"])

    def sentence_entries(self) -> list[dict]:
        return copy.deepcopy(self._state["sentences"])

    def concept_candidates(self) -> list[dict]:
        return copy.deepcopy(self._state["candidates"])

    def concept_entries(self) -> list[dict]:
        return copy.deepcopy(self._state["concepts"])

    def relation_entries(self) -> list[dict]:
        return copy.deepcopy(self._state["relations"])

    def model_contract(self) -> dict:
        """Manifest to pin a tokenizer LM-vocabulary to the model's tensors."""
        relevant = {k: self._state[k] for k in ("tokens", "merges", "literals")}
        return {"format": "neural-tokenizer-v1", "vocab_size": self.vocab_size,
                "vocabulary_sha256": _digest(relevant)}

    def assert_model_contract(self, contract: dict) -> None:
        if contract != self.model_contract():
            raise ContractMismatch("Tokenizer changed; model embeddings/output head and data must be migrated")

    def save(self, path: str | Path) -> None:
        """Integrity-checked atomic snapshot; no concurrent writer coordination."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._reindex_and_validate()
        document = {"data": self._state, "integrity_sha256": _digest(self._state)}
        data = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2).encode("utf8")
        fd, temporary = tempfile.mkstemp(prefix=".tmp-tokenizer-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> "LayeredTokenizer":
        path = Path(path)
        if path.stat().st_size > 100_000_000:
            raise RegistryError("Registry exceeds size limit")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if set(doc) != {"data", "integrity_sha256"} or doc["integrity_sha256"] != _digest(doc["data"]):
                raise RegistryError("Registry integrity mismatch")
            return cls(doc["data"])
        except (UnicodeError, ValueError, KeyError, TypeError) as e:
            raise RegistryError(f"Cannot load registry: {e}") from e
