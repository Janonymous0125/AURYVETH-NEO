"""Lossless, graph-indexed document archive for selective context recovery.

No canned answers. Retrieval supplies verbatim evidence which a neural LM
must interpret. BM25-ish lexical search plus explicit graph hops; this is a
CPU reference implementation with O(document count) lookup, not ANN search.
The archive is kept separate from the model checkpoint to avoid silently
baking raw private conversation text into a distributable weight file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

# English words, numerals and individual CJK ideographs; index is Unicode safe.
WORDS = re.compile(r'[a-zA-Z][a-zA-Z0-9_-]*|[0-9]+|[\u3400-\u9fff]')


def terms(text: str):
    return WORDS.findall(text.lower())


class ContextArchive:
    """Versioned source text chunks with stable IDs and explicit links.

    `add` does not invent entities or factual relationships; graph edges must
    be supplied by the caller. Linked nodes enable multi-hop source expansion.
    """

    SCHEMA = 1

    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.edges: list[dict] = []

    def add(self, text: str, *, source: str, doc_id: str | None = None) -> str:
        if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
            raise ValueError('Text must be a nonempty chunk under 100k characters')
        if not isinstance(source, str) or not source.strip() or len(source) > 256:
            raise ValueError('Source provenance required')
        if doc_id is None:
            doc_id = hashlib.sha256((source+'\0'+text).encode('utf-8')).hexdigest()[:24]
        if not isinstance(doc_id, str) or not re.fullmatch(r'[\w.-]{1,100}', doc_id):
            raise ValueError('Invalid document ID')
        existing = self.documents.get(doc_id)
        data = {'id':doc_id, 'text':text, 'source':source,
                'sha256':hashlib.sha256(text.encode('utf-8')).hexdigest()}
        if existing is not None and existing != data:
            raise ValueError('Document ID already refers to different content; remove first')
        self.documents[doc_id] = data
        return doc_id

    def link(self, left: str, right: str, *, relation: str):
        if (left == right or left not in self.documents or right not in self.documents
            or not isinstance(relation, str) or not 1 <= len(relation.strip()) <= 64):
            raise ValueError('Graph edge needs two existing different documents and a relation')
        edge = {'left':left, 'right':right, 'relation':relation}
        if edge not in self.edges:
            self.edges.append(edge)

    def delete(self, doc_id: str):
        if doc_id not in self.documents:
            raise KeyError(doc_id)
        del self.documents[doc_id]
        self.edges = [e for e in self.edges if e['left'] != doc_id and e['right'] != doc_id]

    def retrieve(self, query: str, *, top_k: int = 3, max_hops: int = 1,
                 max_total: int = 5, min_score: float = 0.01,
                 policy: str = 'legacy'):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('Query required')
        if not 1 <= top_k <= 100 or not 0 <= max_hops <= 8 or not 1 <= max_total <= 100:
            raise ValueError('Invalid retrieval budget')
        if policy not in ('legacy', 'balanced'):
            raise ValueError('Retrieval policy must be legacy or balanced')
        if not self.documents:
            return []
        q = Counter(terms(query))
        if not q:
            return []
        docs = list(self.documents.values())
        tfs = {d['id']:Counter(terms(d['text'])) for d in docs}
        frequencies = Counter(term for tf in tfs.values() for term in tf)
        avg_len = sum(sum(tf.values()) for tf in tfs.values()) / len(docs)
        scores = []
        for doc in docs:
            tf = tfs[doc['id']]
            length = sum(tf.values())
            score = 0.0
            for term in q:
                if tf[term]:
                    idf = math.log(1 + (len(docs) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                    score += idf * (tf[term] * 2.2) / (tf[term] + 1.2 * (0.25 + 0.75 * length / max(1, avg_len)))
            if score >= min_score:
                scores.append((score, doc['id']))
        scores.sort(key=lambda x: (-x[0], x[1]))
        seeds = scores[:top_k]
        adj = defaultdict(list)
        for edge in self.edges:
            adj[edge['left']].append((edge['right'],edge['relation']))
            adj[edge['right']].append((edge['left'],edge['relation']))
        if policy == 'balanced':
            # Alternate lexical seeds and linked discoveries. Equal seed and
            # result budgets must not starve every graph-only neighbor.
            lexical = deque((doc_id, 0, score) for score, doc_id in seeds)
            seed_scores = {doc_id:score for score, doc_id in seeds}
            graph = deque()
            result, visited = [], set()
            use_graph = False
            while (lexical or graph) and len(result) < max_total:
                queue = graph if graph and (use_graph or not lexical) else lexical
                doc_id, depth, score = queue.popleft()
                if doc_id in visited:
                    continue
                # A lexical seed reached through another seed is still a
                # depth-zero root with its full hop allowance.
                if doc_id in seed_scores:
                    depth, score = 0, seed_scores[doc_id]
                visited.add(doc_id)
                result.append({**self.documents[doc_id], 'score':float(score), 'hop':depth})
                if depth < max_hops:
                    for neighbor, _ in sorted(adj[doc_id]):
                        if neighbor not in visited:
                            graph.append((neighbor, depth + 1, 0.0))
                use_graph = not use_graph
            return result
        result = []
        visited = set()
        queue = deque((doc_id, 0, score) for score, doc_id in seeds)
        while queue and len(result) < max_total:
            doc_id, depth, score = queue.popleft()
            if doc_id in visited:
                continue
            visited.add(doc_id)
            result.append({**self.documents[doc_id], 'score':float(score), 'hop':depth})
            if depth < max_hops:
                for neighbor, relation in sorted(adj[doc_id]):
                    if neighbor not in visited:
                        queue.append((neighbor, depth+1, 0.0))
        return result

    def save(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {'schema':self.SCHEMA, 'documents':list(self.documents.values()), 'edges':self.edges}
        fd, temp = tempfile.mkstemp(prefix='.archive-', dir=target.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as out:
                json.dump(payload,out,ensure_ascii=False,sort_keys=True)
                out.flush();os.fsync(out.fileno())
            os.replace(temp,target)
        finally:
            if os.path.exists(temp):os.unlink(temp)

    @classmethod
    def load(cls, path: str | Path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data,dict) or set(data)!={'schema','documents','edges'} or data['schema']!=cls.SCHEMA:
            raise ValueError('Invalid archive schema')
        archive=cls()
        if not isinstance(data['documents'],list) or not isinstance(data['edges'],list):
            raise ValueError('Invalid archive records')
        for d in data['documents']:
            if not isinstance(d,dict) or set(d)!={'id','text','source','sha256'}:
                raise ValueError('Invalid document record')
            if hashlib.sha256(d['text'].encode('utf8')).hexdigest()!=d['sha256']:
                raise ValueError('Original text integrity check failed')
            archive.add(d['text'],source=d['source'],doc_id=d['id'])
        for e in data['edges']:
            if not isinstance(e,dict) or set(e)!={'left','right','relation'}:
                raise ValueError('Invalid graph link')
            archive.link(e['left'],e['right'],relation=e['relation'])
        if len(archive.edges)!=len(data['edges']):
            raise ValueError('Duplicate graph links')
        return archive
