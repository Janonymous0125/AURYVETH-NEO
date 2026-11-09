"""Multi-turn CLI for checkpoints trained with the shared conversation format."""
import argparse
from pathlib import Path
import torch
from model import IntegratedLM
from model.artifacts import verify_manifest
from neural_tokenizer import SentencePieceTokenizer
from scripts.build_conversation_data import format_prompt
from scripts.switch_reasoning import STATES, deliberate_switch


def reply(model, tokenizer, messages, *, max_new_tokens=80, temperature=0.7):
    if not 1 <= max_new_tokens < model.config.block_size:
        raise ValueError('Invalid answer budget')
    history = list(messages)
    while True:
        prompt = format_prompt(history, allow_empty_assistant=True)
        ids = tokenizer.encode(prompt)
        if len(ids) + max_new_tokens <= model.config.block_size:
            break
        if len(history) <= 1:
            raise ValueError('User message exceeds context budget')
        history = history[2:]  # Drop oldest complete exchange, never half a turn.
    tensor = torch.tensor([ids], dtype=torch.long, device=model.token_embedding.weight.device)
    output = model.generate(tensor, max_new_tokens=max_new_tokens, temperature=temperature,
                            eos_id=tokenizer.EOS_ID)
    return tokenizer.decode(output[0,len(ids):].tolist()).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', default='data/tokenizer_50k.model')
    p.add_argument('--manifest', help='Verify a release bundle and require its model/best_model artifact')
    p.add_argument('--think-manifest', help='Enable /think for structured switch puzzles')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max-new-tokens', type=int, default=80)
    args = p.parse_args()
    if args.manifest:
        artifacts, _ = verify_manifest(args.manifest)
        if Path(args.checkpoint).resolve() not in [v for k,v in artifacts.items() if k in ('model','best_model')]:
            p.error('Checkpoint is not a model artifact in this manifest')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tok = SentencePieceTokenizer(args.tokenizer)
    model = IntegratedLM.load(args.checkpoint, tok).eval()
    planner = None
    if args.think_manifest:
        files, metadata = verify_manifest(args.think_manifest)
        variant = metadata.get('prompt_variant', 0)
        if (metadata.get('task') != 'switch-transitions-v1'
                or metadata.get('cli_qualified') is not True or 'model' not in files
                or type(variant) is not int or variant not in (0, 5)):
            p.error('--think-manifest must identify a qualified switch planner bundle')
        planner = IntegratedLM.load(files['model'], tok).eval()
    messages = []
    print('Enter /reset for a new conversation or /quit to exit.')
    if planner is not None:
        print('Switch puzzles: /think off toggle keep')
    while True:
        try:
            text = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == '/quit':
            break
        if text == '/reset':
            messages.clear()
            continue
        if not text:
            continue
        if planner is not None:
            command, *arguments = text.split()
            if command == '/think':
                if len(arguments) < 2:
                    print('Use /think followed by on, off, 开, or 关, then commands such as toggle or keep.')
                    continue
                language = 'zh' if arguments[0] in STATES['zh'] else 'en'
                try:
                    answer, _ = deliberate_switch(planner, tok, arguments[0], arguments[1:],
                                                  language=language, variant=variant)
                except ValueError as exc:
                    print(str(exc))
                    continue
                print('Assistant:', answer)
                continue
        messages.append({'role':'user', 'content':text})
        try:
            answer = reply(model, tok, messages, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature)
        except ValueError as exc:
            messages.pop()
            print(str(exc))
            continue
        print('Assistant:', answer)
        # Empty generation still represents a completed assistant turn.
        messages.append({'role':'assistant', 'content':answer})


if __name__ == '__main__':
    main()
