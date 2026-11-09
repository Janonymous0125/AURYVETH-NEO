"""CLI for the isolated LayeredTokenizer research prototype."""
import argparse
import json
from pathlib import Path
from .core import LayeredTokenizer


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description='Persistent layered tokenizer research prototype')
    sub = p.add_subparsers(dest='command', required=True)

    a = sub.add_parser('init', help='Create a new empty registry with legacy byte IDs')
    a.add_argument('registry', type=Path)

    a = sub.add_parser('analyze', help='Read-only four-level analysis')
    a.add_argument('registry', type=Path)
    a.add_argument('text')

    a = sub.add_parser('observe', help='Learn word units and discover unverified concept candidates')
    a.add_argument('registry', type=Path)
    a.add_argument('text')
    a.add_argument('--source', required=True, help='Attributable source identifier')
    a.add_argument('--sentences', action='store_true', help='Persist exact sentence units')

    a = sub.add_parser('train-bpe', help='Explicitly train new BPE vocabulary tokens from UTF-8 text')
    a.add_argument('registry', type=Path)
    a.add_argument('corpus', type=Path)
    a.add_argument('--merges', type=int, default=50)
    a.add_argument('--min-frequency', type=int, default=2)

    a = sub.add_parser('add-token', help='Explicitly append a literal LM token (requires model migration)')
    a.add_argument('registry', type=Path)
    a.add_argument('text')

    a = sub.add_parser('approve', help='Approve discovered candidate after two attributed sources')
    a.add_argument('registry', type=Path)
    a.add_argument('key')
    a.add_argument('--approver', required=True)

    a = sub.add_parser('reject', help='Reject an unverified candidate')
    a.add_argument('registry', type=Path)
    a.add_argument('key')
    a.add_argument('--reviewer', required=True)
    a.add_argument('--reason', required=True)

    a = sub.add_parser('inspect', help='Print registry counts and verification states')
    a.add_argument('registry', type=Path)

    a = sub.add_parser('contract', help='Print pinned LM token vocabulary fingerprint')
    a.add_argument('registry', type=Path)

    args = p.parse_args()
    if args.command == 'init':
        if args.registry.exists():
            p.error('Will not overwrite an existing registry')
        t = LayeredTokenizer()
        t.save(args.registry)
        output({'registry': str(args.registry), 'contract': t.model_contract()})
        return

    t = LayeredTokenizer.load(args.registry)
    if args.command == 'analyze':
        output(t.analyze(args.text))
        return
    if args.command == 'contract':
        output(t.model_contract())
        return
    if args.command == 'inspect':
        c = t.concept_candidates()
        output({'vocab_size': t.vocab_size,
                'bpe_merges': len(t._state['merges']),
                'literal_tokens': len(t._state['literals']),
                'words': len(t.word_entries()),
                'sentences': len(t.sentence_entries()),
                'registered_concepts': len(t.concept_entries()),
                'registered_relations': len(t.relation_entries()),
                'concept_candidates': len(c),
                'pending': sum(x['status'] == 'pending' for x in c),
                'rejected': sum(x['status'] == 'rejected' for x in c),
                'contract': t.model_contract()})
        return
    if args.command == 'observe':
        result = t.observe(args.text, source=args.source,
                           remember_sentences=args.sentences)
        output({'word_units': result['words'], 'concept_hypotheses': result['concept_hypotheses'],
                'vocab_size': t.vocab_size})
    elif args.command == 'train-bpe':
        corpus = args.corpus.read_text(encoding='utf-8')
        n = t.train_bpe(corpus.splitlines(keepends=True), max_merges=args.merges,
                        min_frequency=args.min_frequency)
        output({'new_bpe_merges': n, 'vocab_size': t.vocab_size,
                'warning': 'This changes the LM vocabulary; migrate model embeddings/output and retokenize data.'})
    elif args.command == 'add-token':
        i = t.register_literal_token(args.text)
        output({'new_or_existing_model_token_id': i, 'vocab_size': t.vocab_size,
                'warning': 'Model checkpoint migration required before using new token IDs.'})
    elif args.command == 'approve':
        i = t.approve_candidate(args.key, approver=args.approver)
        output({'registered_relation_id': i})
    elif args.command == 'reject':
        t.reject_candidate(args.key, reviewer=args.reviewer, reason=args.reason)
        output({'candidate': args.key, 'status': 'rejected'})
    t.save(args.registry)


if __name__ == '__main__':
    main()
