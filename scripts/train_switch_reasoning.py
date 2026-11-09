"""Train the switch planner from a recurrent checkpoint and test composition."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import shutil

import torch
from torch.nn import functional as F

from model import IntegratedLM
from model.artifacts import file_sha256, verify_manifest, write_manifest
from neural_tokenizer import SentencePieceTokenizer
from scripts.switch_reasoning import COMMANDS, STATES, deliberate_switch, transition_prompt


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def expected_state(state, command, language):
    off, on = STATES[language]
    return {'keep': state, 'toggle': off if state == on else on,
            'set-off': off, 'set-on': on}[command]


def measure(model, tok, cases, *, variant):
    rows = []
    for language, initial, commands in cases:
        predicted, trace = deliberate_switch(model, tok, initial, commands, language=language, variant=variant)
        expected = []
        state = initial
        for command in commands:
            state = expected_state(state, command, language)
            expected.append(state)
        rows.append({'language': language, 'initial': initial, 'commands': commands,
                     'predicted': predicted, 'expected': state, 'trace': trace, 'expected_trace': expected,
                     'trace_correct': trace == expected, 'final_correct': predicted == state})
    return {'languages': {lang: {'total': sum(r['language'] == lang for r in rows),
                                'trace_correct': sum(r['language'] == lang and r['trace_correct'] for r in rows),
                                'final_correct': sum(r['language'] == lang and r['final_correct'] for r in rows)}
                          for lang in STATES},
            'trace_correct': sum(r['trace_correct'] for r in rows), 'total': len(rows), 'examples': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=360)
    parser.add_argument('--baseline', required=True, help='Recurrent chat training manifest')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    files, _ = verify_manifest(args.baseline)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(603)
    rng = random.Random(603)
    tok = SentencePieceTokenizer(files['tokenizer'])
    model = IntegratedLM.load(files['model'], tok).eval()
    if model.memory.count or model.config.backbone != 'recurrent' or model.config.planner_steps:
        raise ValueError('Training requires an empty-bank recurrent checkpoint without a planner.')
    model.add_planner_to_existing_model(3)
    model.requires_grad_(False)
    for block in model.blocks:
        block.planner.requires_grad_(True)
        block.mixer.requires_grad_(True)
        block.mixer_norm.requires_grad_(True)
    frozen = frozen_digest(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': [p for name, p in model.named_parameters() if p.requires_grad and '.planner.' in name],
         'lr': 8e-4},
        {'params': [p for name, p in model.named_parameters() if p.requires_grad and '.planner.' not in name],
         'lr': 3e-4},
    ])
    singles = [(lang, state, (command,)) for lang, states in STATES.items()
               for state, command in product(states, COMMANDS)]
    canonical, validation, held_out = 5, 6, 7
    variants = (canonical, 0, 1)
    compact_rows = {}
    for lang, states in STATES.items():
        encoded = [tok.encode(label) for label in states]
        if any(len(ids) != 1 for ids in encoded) or encoded[0] == encoded[1]:
            raise ValueError(f'{lang} requires two distinct single-token switch labels.')
        labels = [ids[0] for ids in encoded]
        for state, command, variant in product(states, COMMANDS, variants):
            tokens = torch.tensor([tok.encode(transition_prompt(state, command, lang, variant))])
            target = states.index(expected_state(state, command, lang))
            compact_rows[lang, state, command, variant] = tokens, labels, target
    history = []
    best = (-1, -1)
    path = args.out/'model.pt'
    for step in range(1, args.steps + 1):
        if (step - 1) % 4 == 0:
            rng.shuffle(singles)
        cases = singles[4*((step-1) % 4):4*((step-1) % 4 + 1)]
        keys = [(lang, state, commands[0], canonical if step % 3 else rng.choice(variants))
                for lang, state, commands in cases]
        model.train()
        opt.zero_grad(set_to_none=True)
        scores, targets = [], []
        for key in keys:
            ids, labels, target = compact_rows[key]
            logits, _ = model._recurrent_prefill(ids)
            scores.append(logits[0, labels])
            targets.append(target)
        loss = F.cross_entropy(torch.stack(scores), torch.tensor(targets))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        opt.step()
        if step % 60 == 0 or step == args.steps:
            model.eval()
            scores = [measure(model, tok, singles, variant=i)['trace_correct'] for i in (canonical, validation)]
            item = {'step': step, 'loss': float(loss.detach()), 'canonical': scores[0], 'validation': scores[1]}
            history.append(item)
            print(json.dumps(item), flush=True)
            key = tuple(scores)
            if key > best:
                model.save(path, tok)
                best = key
                selected = step
            if step >= 180 and min(scores) == len(singles):
                break
    assert frozen_digest(model) == frozen
    model = IntegratedLM.load(path, tok).eval()
    sequences = [(lang, state, commands) for lang, states in STATES.items()
                 for state in states for commands in product(COMMANDS, repeat=3)]
    lengths = random.Random(220)
    sequences.extend((lang, state, tuple(lengths.choices(COMMANDS, k=n))) for lang, states in STATES.items()
                     for state in states for n in (5, 8, 16) for _ in range(8))
    report = {'baseline_sha256': file_sha256(files['model']), 'updates': step, 'selected': selected,
              'train_mixers': True, 'canonical_variant': canonical,
              'history': history, 'frozen_weights_verified': True,
              'canonical': measure(model, tok, sequences, variant=canonical),
              'unseen_phrasing': measure(model, tok, sequences, variant=held_out)}
    for block in model.blocks:
        block.planner.steps = 1
    report['same_weights_one_pass'] = measure(model, tok, sequences, variant=held_out)
    shutil.copyfile(files['tokenizer'], args.out/'tokenizer.model')
    (args.out/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    qualified = report['canonical']['trace_correct'] == report['canonical']['total']
    write_manifest(args.out/'manifest.json',
                   {'model': 'model.pt', 'tokenizer': 'tokenizer.model', 'comparison': 'comparison.json'},
                   metadata={'task': 'switch-transitions-v1', 'baseline_sha256': report['baseline_sha256'],
                             'cli_qualified': qualified, 'prompt_variant': canonical})
    print({key: (report[key]['trace_correct'], report[key]['total'])
           for key in ('canonical', 'unseen_phrasing', 'same_weights_one_pass')}, flush=True)


if __name__ == '__main__':
    main()
