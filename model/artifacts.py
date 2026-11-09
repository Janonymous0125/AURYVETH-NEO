"""Hash-bound release manifests, separate from legacy model checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _local_path(root, name):
    relative = Path(name)
    target = (root / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(root):
        raise ValueError('Artifact path must stay inside its bundle')
    return target


def write_manifest(path, artifacts, *, metadata=None):
    """Publish last, after every named artifact has been completely written."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not artifacts:
        raise ValueError('At least one artifact required')
    records = {}
    for name, relative in artifacts.items():
        target = _local_path(path.parent, relative)
        records[name] = {'path':str(relative), 'bytes':target.stat().st_size,
                         'sha256':file_sha256(target)}
    payload = {'schema':1, 'complete':True, 'artifacts':records,
               'metadata':metadata or {}}
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.manifest-', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload


def verify_manifest(path):
    """Fail closed for partial, tampered or mixed artifact releases."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding='utf8'))
    if (payload.get('schema') != 1 or payload.get('complete') is not True
            or not isinstance(payload.get('artifacts'), dict) or not payload['artifacts']):
        raise ValueError('Incomplete or invalid artifact manifest')
    verified = {}
    for name, record in payload['artifacts'].items():
        if not isinstance(record, dict) or set(record) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Invalid artifact record')
        target = _local_path(path.parent, record['path'])
        if (not target.is_file() or target.stat().st_size != record['bytes']
                or file_sha256(target) != record['sha256']):
            raise ValueError(f'Artifact integrity mismatch: {name}')
        verified[name] = target
    return verified, payload['metadata']
