"""Pinned OpenAssistant preparation with assistant-only targets and tree splits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

from neural_tokenizer import SentencePieceTokenizer
from model.artifacts import file_sha256

DATASET = 'OpenAssistant/oasst1'
REVISION = 'fdf72ae0827c1cda404aff25b6603abec9e3399b'
FILES = {'train':'data/train-00000-of-00001-b42a775f407cee45.parquet',
         'validation':'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'}


def format_prompt(messages, *, allow_empty_assistant=False):
    """No new vocabulary: ordinary text role labels plus EOS in the trainer."""
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('Conversation prompt must end with a user turn')
    lines = []
    for i, message in enumerate(messages):
        expected = 'user' if i % 2 == 0 else 'assistant'
        content = message['content']
        if (message['role'] != expected or not isinstance(content, str)
                or (not content.strip() and not (allow_empty_assistant and expected == 'assistant'))):
            raise ValueError('Expected alternating user/assistant messages with nonempty user turns')
        lines.append(('User: ' if expected == 'user' else 'Assistant: ') + content.strip())
    return '\n'.join(lines) + '\nAssistant: '


def conversation_examples(rows, tokenizer, *, languages=('en', 'zh'), max_tokens=512):
    """Reconstruct paths; never train on rejected, deleted or missing ancestors."""
    by_id = {r['message_id']:r for r in rows}
    examples, stats = [], Counter()
    for reply in sorted(rows, key=lambda r:r['message_id']):
        if reply['role'] != 'assistant':
            continue
        if reply.get('rank') not in (None, 0):
            stats['nonpreferred_reply'] += 1
            continue
        chain, visited, node = [], set(), reply
        valid = True
        while node is not None:
            if (node['message_id'] in visited or node.get('deleted')
                    or node.get('review_result') is not True
                    or node.get('lang') not in languages
                    or not isinstance(node.get('text'), str) or not node['text'].strip()
                    or node.get('message_tree_id') != reply.get('message_tree_id')
                    or (node['role'] == 'assistant' and node.get('rank') not in (None, 0))):
                valid = False
                break
            visited.add(node['message_id'])
            chain.append(node)
            parent = node.get('parent_id')
            if parent is not None and parent not in by_id:
                valid = False
                break
            node = by_id.get(parent)
        if not valid:
            stats['invalid_or_filtered_path'] += 1
            continue
        chain.reverse()
        messages = [{'role':'user' if n['role'] == 'prompter' else n['role'],
                     'content':n['text']} for n in chain[:-1]]
        try:
            prompt = format_prompt(messages)
        except ValueError:
            stats['invalid_roles'] += 1
            continue
        answer = reply['text'].strip()
        length = len(tokenizer.encode(prompt)) + len(tokenizer.encode(answer, add_eos=True)) - 1
        if length > max_tokens:
            stats['over_token_budget'] += 1
            continue
        examples.append({'context':[], 'prompt':prompt, 'answer':answer,
                         '_tree_id':reply['message_tree_id'], '_message_id':reply['message_id'],
                         '_lang':reply['lang'], '_turns':len(chain), '_tokens':length})
    return examples, dict(stats)


def normalized_digest(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode('utf8')).hexdigest()


def separate(examples):
    """Tree-disjoint 90/5/5 split, plus exact normalized cross-split dedup."""
    output = {'train':[], 'validation':[], 'test':[]}
    seen = set()
    # Reserve held-out content first; drop duplicate prompts/answers from train.
    assigned = {k:[] for k in output}
    for row in examples:
        bucket = int(hashlib.sha256(row['_tree_id'].encode()).hexdigest()[:8], 16) % 100
        split = 'test' if bucket < 5 else ('validation' if bucket < 10 else 'train')
        assigned[split].append(row)
    for split in ('test', 'validation', 'train'):
        split_seen = set()
        for row in assigned[split]:
            prompt_key = normalized_digest(row['prompt'])
            answer_key = normalized_digest(row['answer'])
            if prompt_key in seen or answer_key in seen or prompt_key in split_seen:
                continue
            output[split].append(row)
            split_seen.add(prompt_key)
        for row in output[split]:
            seen.add(normalized_digest(row['prompt']))
            seen.add(normalized_digest(row['answer']))
    return output


def download(url, destination):
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        with urllib.request.urlopen(url, timeout=90) as source, temporary.open('wb') as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', default='data/tokenizer_50k.model')
    p.add_argument('--out', default='data/conversation_oasst1_v1')
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--languages', nargs='+', default=['en', 'zh'])
    args = p.parse_args()
    if args.max_tokens < 2:
        p.error('max-tokens must be at least two')
    import pyarrow.parquet as pq
    out = Path(args.out)
    if (out/'manifest.json').exists():
        raise FileExistsError('Dataset output already exists; use a new --out')
    raw = out/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    base = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/'
    sources, rows = {}, []
    for split, relative in FILES.items():
        path = raw/Path(relative).name
        download(base + relative, path)
        sources[split] = {'url':base + relative, 'sha256':file_sha256(path)}
        rows.extend(pq.read_table(path).to_pylist())
    for name in ('LICENSE', 'README.md'):
        download(base + name, raw/name)
    tok = SentencePieceTokenizer(args.tokenizer)
    examples, filters = conversation_examples(rows, tok, languages=args.languages, max_tokens=args.max_tokens)
    splits = separate(examples)
    counts = {}
    for split, selected in splits.items():
        if not selected:
            raise ValueError(f'No usable {split} examples')
        path = out/f'{split}.jsonl'
        with path.open('w', encoding='utf8') as stream:
            for row in selected:
                stream.write(json.dumps({k:row[k] for k in ('context','prompt','answer')}, ensure_ascii=False) + '\n')
        with (out/f'{split}_provenance.jsonl').open('w', encoding='utf8') as stream:
            for row in selected:
                stream.write(json.dumps({k:v for k,v in row.items() if k.startswith('_')}) + '\n')
        counts[split] = {'examples':len(selected), 'languages':dict(Counter(r['_lang'] for r in selected)),
                         'multi_turn':sum(r['_turns'] > 2 for r in selected),
                         'tokens':sum(r['_tokens'] for r in selected),
                         'sha256':file_sha256(path)}
    manifest = {'dataset':DATASET, 'revision':REVISION, 'license':'Apache-2.0',
                'source_card':'https://huggingface.co/datasets/OpenAssistant/oasst1',
                'sources':sources, 'license_sha256':file_sha256(raw/'LICENSE'),
                'tokenizer_contract':tok.model_contract(), 'max_tokens':args.max_tokens,
                'splits':counts, 'filters':filters,
                'split_policy':'SHA256 conversation tree id: 90/5/5, normalized prompt/answer cross-split exact dedup',
                'limitations':['No semantic near-duplicate detection', 'Human replies are not guaranteed factually correct',
                                'Long conversations are skipped, never silently truncated',
                                'Language counts reflect available filtered source data, not balanced bilingual coverage']}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
