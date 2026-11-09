"""Validate physical JSONL records without splitting on Unicode text separators.

Usage: python -m scripts.check_jsonl data/cognitive_v1/train_supervised.jsonl data/cognitive_v1/validation_supervised.jsonl
This tool never modifies source data and needs only the Python standard library.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def check(path: str | Path) -> dict:
    path=Path(path)
    count=0
    kind_count={'memory':0,'non_memory':0}
    with path.open('r',encoding='utf-8',newline='') as source:
        for number,line in enumerate(source,1):
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}, physical line {number}, column {exc.colno}: {exc.msg}') from exc
            if not isinstance(row,dict) or set(row)!={'context','prompt','answer'}:
                raise ValueError(f'{path}, physical line {number}: missing/extra fields (expect context, prompt, answer)')
            if (not isinstance(row['context'],list) or
                any(not isinstance(x,str) or not x for x in row['context']) or
                not isinstance(row['prompt'],str) or
                not isinstance(row['answer'],str) or
                not row['prompt'] or not row['answer']):
                raise ValueError(f'{path}, physical line {number}: invalid data types or empty prompt/answer')
            count+=1
            kind_count['memory' if row['context'] else 'non_memory']+=1
    if count==0:
        raise ValueError(f'{path}: no examples')
    return {'path':str(path),'status':'valid_jsonl','examples':count,**kind_count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths',nargs='+',help='One or more supervised JSONL files')
    args=p.parse_args()
    for path in args.paths:
        print(json.dumps(check(path),ensure_ascii=False))

if __name__=='__main__':
    main()
