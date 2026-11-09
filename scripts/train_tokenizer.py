"""Train a new exact-size byte-fallback BPE vocabulary on genuine corpus files."""
import argparse
from neural_tokenizer import train_sentencepiece_bpe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',nargs='+',required=True,help='Training-only .txt files; NEVER include validation')
    parser.add_argument('--prefix',required=True,help='Output without .model suffix')
    parser.add_argument('--vocab-size',type=int,default=50000)
    args=parser.parse_args()
    path=train_sentencepiece_bpe(args.input,args.prefix,vocab_size=args.vocab_size)
    print(path)

if __name__=='__main__': main()
