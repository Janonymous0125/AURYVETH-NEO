"""Regression gates for expanded capacity and memory-bounded training."""
import json
import pytest
import torch
from model import IntegratedConfig, IntegratedLM
from neural_tokenizer import LayeredTokenizer
from scripts import train


def tiny_model(vocab=258):
    return IntegratedLM(IntegratedConfig(vocab_size=vocab, block_size=48, d_model=32,
        n_layers=1, n_heads=4, n_kv_heads=2, ffn_hidden=64,
        memory_slots=4, memory_tokens=16, regions=2, active_regions=1, active_memories=2))


def test_full_capacity_default_only_not_allocated():
    cfg = IntegratedConfig()
    assert (cfg.vocab_size, cfg.block_size, cfg.d_model) == (50_000, 8192, 1024)
    assert cfg.d_model % cfg.n_heads == 0


def test_chunked_loss_matches_full_with_gradient():
    torch.manual_seed(22)
    m = tiny_model()
    m.eval()
    ids = torch.randint(0, 250, (1, 19))
    targets = ids.roll(-1, 1)
    targets[:, :4] = -100
    full = m(ids, targets=targets).loss
    cheap = m(ids, targets=targets, return_logits=False, loss_chunk_size=3)
    assert cheap.logits is None
    torch.testing.assert_close(full, cheap.loss, rtol=1e-5, atol=1e-5)
    cheap.loss.backward()
    assert m.lm_head.weight.grad is not None


def test_checkpointed_block_matches_normal_and_has_gradients():
    torch.manual_seed(10)
    a = tiny_model(); b = tiny_model(); b.load_state_dict(a.state_dict())
    ids = torch.randint(0, 250, (1, 13)); gold=ids.roll(-1, 1)
    a.train(); b.train(); b.gradient_checkpointing=True
    aa=a(ids, targets=gold, return_logits=False, loss_chunk_size=4).loss
    bb=b(ids, targets=gold, return_logits=False, loss_chunk_size=4).loss
    torch.testing.assert_close(aa,bb,rtol=1e-5,atol=1e-5)
    aa.backward();bb.backward()
    torch.testing.assert_close(a.token_embedding.weight.grad,b.token_embedding.weight.grad,rtol=1e-4,atol=1e-5)


def test_resume_training_state_and_validation(tmp_path, monkeypatch):
    from pathlib import Path
    import sys
    tr = tmp_path/'train.txt'; val = tmp_path/'val.txt'
    tr.write_text('Real language and Unicode 中文 text.\n' * 9, encoding='utf-8')
    val.write_text('Different validation words 香蕉 world.\n' * 9, encoding='utf-8')
    out=tmp_path/'out'
    common = ['--corpus',str(tr),'--validation-corpus',str(val),
        '--allow-small-dataset','--allow-byte-tokenizer','--width','32',
        '--layers','1','--heads','4','--kv-heads','2','--block-size','48',
        '--train-seq-len','24','--grad-accum','1','--batch-size','1',
        '--device','cpu','--precision','fp32','--log-every','1',
        '--eval-every','1','--save-every','1','--validation-max-rows','2']
    monkeypatch.setattr(sys,'argv',['train']+common+['--out',str(out),'--steps','3','--stop-after','2'])
    train.main()
    assert (out/'training_state.pt').exists()
    first = json.loads((out/'train_metrics.json').read_text())
    assert first['completed_steps'] == 2
    assert first['tokens_seen'] > 0
    monkeypatch.setattr(sys,'argv',['train']+common+['--out',str(out),'--steps','3',
                                  '--resume-state',str(out/'training_state.pt')])
    train.main()
    last=json.loads((out/'train_metrics.json').read_text())
    assert last['completed_steps'] == 3
    assert last['tokens_seen'] > first['tokens_seen']
    assert last['validation_loss'] is not None


def test_no_unvalidated_training_by_default(tmp_path,monkeypatch):
    import sys
    corpus=tmp_path/'corpus.txt'; corpus.write_text('One line of text')
    monkeypatch.setattr(sys,'argv',['train','--corpus',str(corpus),'--out',str(tmp_path/'out'),
       '--allow-byte-tokenizer','--allow-small-dataset','--width','32','--layers','1',
       '--heads','4','--kv-heads','2','--block-size','48','--steps','1'])
    with pytest.raises(SystemExit):
        train.main()


def test_exact_resume_matches_uninterrupted_cpu(tmp_path,monkeypatch):
    import sys
    corpus=tmp_path/'corpus.txt';val=tmp_path/'validation.txt'
    corpus.write_text(('An actual text used for tests only, 中文!\n')*15,encoding='utf8')
    val.write_text(('A separate held-out input for validation.\n')*12,encoding='utf8')
    opts=['--corpus',str(corpus),'--validation-corpus',str(val),'--allow-small-dataset',
       '--allow-byte-tokenizer','--width','32','--layers','1','--heads','4','--kv-heads','2',
       '--block-size','40','--train-seq-len','24','--steps','3', '--grad-accum','2',
       '--batch-size','1','--device','cpu','--precision','fp32','--warmup-steps','1',
       '--save-every','2','--eval-every','2','--log-every','3']
    continuous=tmp_path/'continuous';paused=tmp_path/'paused'
    monkeypatch.setattr(sys,'argv',['train']+opts+['--out',str(continuous)])
    train.main()
    monkeypatch.setattr(sys,'argv',['train']+opts+['--out',str(paused),'--stop-after','2'])
    train.main()
    monkeypatch.setattr(sys,'argv',['train']+opts+['--out',str(paused),
                                  '--resume-state',str(paused/'training_state.pt')])
    train.main()
    def load(path):
        return torch.load(path/'training_state.pt',weights_only=True,map_location='cpu')
    a=load(continuous);b=load(paused)
    assert a['completed_steps']==b['completed_steps']==3
    assert a['tokens_seen']==b['tokens_seen']
    for key in a['model']:
        assert torch.equal(a['model'][key],b['model'][key]),key


def test_real_50k_head_in_memory_bounded_train_mode():
    torch.manual_seed(42)
    model=tiny_model(vocab=50_000)
    model.train()
    inputs=torch.randint(0,50000,(1,17))
    output=model(inputs,targets=inputs.roll(-1,1),return_logits=False,loss_chunk_size=4)
    assert output.logits is None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_train_validation_same_file_rejected(tmp_path,monkeypatch):
    import sys
    path=tmp_path/'text.txt';path.write_text('hello world 中文!\n'*5,encoding='utf8')
    cmd=['train','--corpus',str(path),'--validation-corpus',str(path),
         '--allow-small-dataset','--allow-byte-tokenizer',
         '--width','32','--layers','1','--heads','4','--kv-heads','2',
         '--block-size','40','--steps','1','--out',str(tmp_path/'out')]
    monkeypatch.setattr(sys,'argv',cmd)
    with pytest.raises(ValueError,match='identical'):
        train.main()


def test_true_batch_loss_padding_and_memory_gradient():
    from scripts.train import batch_loss
    t=LayeredTokenizer()
    m=tiny_model();m.train()
    rows=[(t.encode('Hello.'),t.encode('World!'),[]) ]
    # Correctly paired teacher-forced token spans in a variable-length batch.
    texts=['a cat slept.','a very big cat slept.']
    textrows=[]
    for txt in texts:
        ids=t.encode(txt,add_eos=True)
        textrows.append((ids[:-1],ids[1:],[]))
    result=batch_loss(m,textrows,'cpu',loss_chunk_size=6)
    assert torch.isfinite(result)
    result.backward()
    assert m.token_embedding.weight.grad is not None

    # Memory-conditioned batching uses different contexts in the same forward.
    m.zero_grad(set_to_none=True)
    rows=[]
    for context,prompt,answer in [('A is red.','Color? ','red.'),
                                  ('B is blue.','Color B? ','blue.')]:
        question=t.encode(prompt);ans=t.encode(answer,add_eos=True)
        seq=question+ans
        rows.append((seq[:-1],[-100]*(len(question)-1)+seq[len(question):],
                     [t.encode(context)]))
    memory_loss=batch_loss(m,rows,'cpu',loss_chunk_size=6)
    memory_loss.backward()
    assert m.memory_encoder.weight_ih_l0.grad is not None


def test_persistent_memory_remains_after_optimizer_resume(tmp_path,monkeypatch):
    from model import IntegratedConfig
    from neural_tokenizer import LayeredTokenizer
    import sys
    t=LayeredTokenizer()
    cfg=IntegratedConfig(vocab_size=258,block_size=48,d_model=32,n_layers=1,
        n_heads=4,n_kv_heads=2,ffn_hidden=64,memory_slots=8,memory_tokens=48,
        regions=2,active_regions=1)
    m=IntegratedLM(cfg)
    m.remember('The ocean is blue.',t,source='test')
    previous=m.memory.states.clone()
    source=tmp_path/'model.pt';m.save(source,t)
    corpus=tmp_path/'input.txt';val=tmp_path/'valid.txt'
    corpus.write_text('Some words to process.\n'*8)
    val.write_text('A disjoint evaluation.\n'*8)
    output=tmp_path/'output'
    command=['train','--resume-checkpoint',str(source),'--corpus',str(corpus),
       '--validation-corpus',str(val),'--allow-small-dataset',
       '--out',str(output),'--steps','2','--stop-after','1',
       '--train-seq-len','24','--grad-accum','1','--batch-size','1',
       '--eval-every','1','--save-every','1','--log-every','1']
    monkeypatch.setattr(sys,'argv',command);train.main()
    checkpoint=output/'training_state.pt'
    command=command.copy()
    del command[1:3]
    stop_index=command.index('--stop-after')
    del command[stop_index:stop_index+2]
    command.extend(['--resume-state',str(checkpoint)])
    monkeypatch.setattr(sys,'argv',command);train.main()
    final=IntegratedLM.load(output/'model.pt',t)
    assert final.memory.count==1
    assert torch.equal(final.memory.states,previous)
    assert final.memory.provenance[0]['source']=='test'
