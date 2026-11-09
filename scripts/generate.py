"""Generate from a trained checkpoint with optional persistent neural memory writes."""

import argparse

import torch

from model import IntegratedLM
from neural_tokenizer import LayeredTokenizer, SentencePieceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", help="Required matching .model for 50k SentencePiece checkpoints")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--remember", action="append", default=[])
    parser.add_argument("--save", help="Save the updated model and memories to this path")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.75)
    args = parser.parse_args()

    tokenizer = (SentencePieceTokenizer(args.tokenizer) if args.tokenizer and args.tokenizer.endswith(".model")
                 else LayeredTokenizer.load(args.tokenizer) if args.tokenizer
                 else LayeredTokenizer())
    model = IntegratedLM.load(args.checkpoint, tokenizer)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model.to(device)
    prefix = tokenizer.encode(args.prompt)
    if not prefix:
        parser.error("Prompt cannot be empty")

    for memory_text in args.remember:
        memory_id = model.remember(memory_text, tokenizer, source="cli-user")
        print(f"Neural memory {memory_id} written")
    if args.save:
        model.save(args.save, tokenizer)

    with torch.inference_mode():
        output = model.generate(
            torch.tensor([prefix], dtype=torch.long, device=device),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_id=tokenizer.EOS_ID,
        )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
