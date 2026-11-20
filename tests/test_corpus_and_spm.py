import json
from pathlib import Path
import pytest
import torch
from neural_tokenizer import SentencePieceTokenizer, train_sentencepiece_bpe
from scripts.train import TokenStream


def test_train_real_bpe_and_unicode_roundtrip(tmp_path):
    source = tmp_path/'train.txt'
    source.write_text(''.join(f'alpha{i} beta{i} 你好 🍎  English and 中文.\n' for i in range(180)),encoding='utf8')
    path = train_sentencepiece_bpe([source],tmp_path/'vocab',vocab_size=512)
    tk = SentencePieceTokenizer(path)
    assert tk.vocab_size == 512
    for s in ['hello world!', 'Hello 你好 🍎!', ' A  B\n', '👩\u200d💻', 'cat 中文', '\t\n']:
        assert tk.decode(tk.encode(s)) == s
    assert tk.model_contract() == SentencePieceTokenizer(path).model_contract()
    streams=TokenStream(source,tk,tmp_path/'cache')
    assert streams.count>500
    assert streams.row(0,8)[0] and streams.row(0,8)[1]
    assert len(streams.validation_rows(16,4))==4


def test_tokenizer_fails_on_insufficient_corpus_for_50k(tmp_path):
    path=tmp_path/'small.txt'; path.write_text('apple orange banana'*5)
    with pytest.raises(Exception):
        train_sentencepiece_bpe([path],tmp_path/'small_vocab',vocab_size=50000)


def test_dotted_tokenizer_prefix_is_preserved(tmp_path):
    source = tmp_path / 'train.txt'
    source.write_text(''.join(f'alpha{i} 你好 🍎 beta.\n' for i in range(150)), encoding='utf8')

    model = train_sentencepiece_bpe([source], tmp_path / 'tokenizer.v2', vocab_size=400)

    assert model == tmp_path / 'tokenizer.v2.model'
    assert model.is_file()
    assert (tmp_path / 'tokenizer.v2.vocab').is_file()
    assert SentencePieceTokenizer(model).vocab_size == 400


@pytest.mark.parametrize('existing_suffixes', [('.model', '.vocab'), ('.model',), ('.vocab',)])
def test_tokenizer_training_does_not_replace_existing_artifacts(tmp_path, existing_suffixes):
    source = tmp_path / 'train.txt'
    source.write_text(''.join(f'alpha{i} 你好 🍎 beta.\n' for i in range(150)), encoding='utf8')
    prefix = tmp_path / 'tokenizer'
    train_sentencepiece_bpe([source], prefix, vocab_size=400)
    artifacts = {suffix: prefix.with_suffix(suffix) for suffix in ('.model', '.vocab')}
    original = {suffix: artifacts[suffix].read_bytes() for suffix in existing_suffixes}
    for suffix in artifacts.keys() - original.keys():
        artifacts[suffix].unlink()
    source.write_text(''.join(f'gamma{i} 中文 😀 delta.\n' for i in range(150)), encoding='utf8')

    with pytest.raises(FileExistsError):
        train_sentencepiece_bpe([source], prefix, vocab_size=400)

    assert {suffix: path.read_bytes() for suffix, path in artifacts.items() if path.exists()} == original


def test_tokenizer_publication_rolls_back_its_sidecar_on_model_conflict(tmp_path, monkeypatch):
    source = tmp_path / 'train.txt'
    source.write_text(''.join(f'alpha{i} 你好 🍎 beta.\n' for i in range(150)), encoding='utf8')
    prefix = tmp_path / 'tokenizer'
    model_path = prefix.with_suffix('.model')
    hardlink_to = Path.hardlink_to

    def publish_with_conflict(path, target):
        if path == model_path:
            path.write_bytes(b'existing model')
        return hardlink_to(path, target)

    monkeypatch.setattr(Path, 'hardlink_to', publish_with_conflict)
    with pytest.raises(FileExistsError):
        train_sentencepiece_bpe([source], prefix, vocab_size=400)

    assert model_path.read_bytes() == b'existing model'
    assert not prefix.with_suffix('.vocab').exists()


def test_sentencepiece_checkpoint_training_pipeline(tmp_path,monkeypatch):
    """Real tokenizer, trainer, checkpoint, loader and Unicode prompts together."""
    import sys
    from scripts import train
    from model import IntegratedLM
    source=tmp_path/'train.txt';val=tmp_path/'val.txt'
    source.write_text(''.join(f'alpha{i} 你好 🍎 beta.\n' for i in range(150)),encoding='utf8')
    val.write_text(''.join(f'gamma{i} 不同的文本 🍎.\n' for i in range(40)),encoding='utf8')
    tok_file=train_sentencepiece_bpe([source],tmp_path/'model_tokenizer',vocab_size=400)
    dest=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['train','--corpus',str(source),'--validation-corpus',str(val),
      '--tokenizer',str(tok_file),'--allow-small-vocab','--allow-small-dataset',
      '--out',str(dest),'--width','32','--layers','1','--heads','4','--kv-heads','2',
      '--block-size','64','--train-seq-len','32','--steps','2',
      '--warmup-steps','1','--grad-accum','1','--eval-every','1','--save-every','1'])
    train.main()
    tok=SentencePieceTokenizer(tok_file)
    lm=IntegratedLM.load(dest/'model.pt',tok)
    assert lm.config.vocab_size==400
    assert lm.config.block_size==64
    with pytest.raises(ValueError,match='tokenizer mismatch'):
        IntegratedLM.load(dest/'model.pt',__import__('neural_tokenizer').LayeredTokenizer())
    ids=torch.tensor([tok.encode('你好 🍎')])
    with torch.no_grad():
        output=lm(ids)
    assert output.logits.shape==(1,ids.shape[1],400)
