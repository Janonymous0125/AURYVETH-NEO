"""v0.6.1: optional trained retrieval + recurrent scan correctness."""
import copy
import torch
import pytest
from model import (IntegratedConfig, IntegratedLM, EvidenceRanker, ContextArchive,
                   evidence_prompt, grounded_training_row)
from model.recurrent import GatedRecurrentMixer
from neural_tokenizer import LayeredTokenizer


def tiny():
    return IntegratedConfig(vocab_size=258, d_model=24, ffn_hidden=48,
                            n_heads=4, n_kv_heads=2, n_layers=1,
                            block_size=256, backbone='recurrent',
                            memory_slots=8, memory_tokens=64,
                            regions=2, active_regions=2)


@pytest.mark.parametrize('length', [1, 4, 17, 64, 65, 130])
def test_scan_forward_and_gradients_match_reference(length):
    torch.manual_seed(length)
    mixer=GatedRecurrentMixer(12)
    other=copy.deepcopy(mixer)
    x=torch.randn(2,length,12,requires_grad=True)
    x2=x.detach().clone().requires_grad_()
    initial=torch.randn(2,12)
    a,b=mixer(x,initial)
    c,d=other.forward_reference(x2,initial)
    assert torch.allclose(a,c,atol=2e-6,rtol=2e-5)
    assert torch.allclose(b,d,atol=2e-6,rtol=2e-5)
    (a.square().sum()+b.square().sum()).backward()
    (c.square().sum()+d.square().sum()).backward()
    assert torch.allclose(x.grad,x2.grad,atol=3e-5,rtol=2e-4)
    for (k,p),(k2,q) in zip(mixer.named_parameters(),other.named_parameters()):
        assert k==k2
        assert torch.allclose(p.grad,q.grad,atol=3e-5,rtol=3e-4),k


def test_scan_extreme_decay_no_nan():
    torch.manual_seed(100)
    mixer=GatedRecurrentMixer(16)
    with torch.no_grad():
        mixer.decay_bias.fill_(-22)
    x=torch.randn(1,139,16)
    y,z=mixer(x)
    a,b=mixer.forward_reference(x)
    assert torch.isfinite(y).all() and torch.isfinite(z).all()
    assert torch.allclose(y,a,atol=1e-6)
    assert torch.allclose(z,b,atol=1e-6)


def test_training_mask_only_supervises_answer():
    tok=LayeredTokenizer()
    docs=[{'id':'source1','text':'Mira owns a violet drone.'}]
    q='Who owns the violet drone?'
    answer='Mira.'
    inputs, targets, contexts=grounded_training_row(q,answer,docs,tok,256,prompt_style='compact')
    assert contexts==[]
    prefix=tok.encode(evidence_prompt(q,docs,style='compact'))
    assert targets[:len(prefix)-1] == [-100]*(len(prefix)-1)
    assert targets[len(prefix)-1:] == tok.encode(answer,add_eos=True)
    assert len(targets)==len(inputs)


def test_optional_reranker_keeps_checkpoint_and_legacy_behavior(tmp_path):
    tok=LayeredTokenizer()
    torch.manual_seed(10)
    model=IntegratedLM(tiny()).eval()
    old_keys=set(model.state_dict())
    archive=ContextArchive()
    archive.add('Name LENA owns copper.',source='x',doc_id='a')
    archive.add('Name GANA owns copper.',source='y',doc_id='b')
    class FakeRanker:
        def rerank(self, question,hits,tokenizer,*,top_k):
            return sorted(hits,key=lambda d:d['id'],reverse=True)[:top_k]
    _,hits=model.generate_with_archive('copper name',tok,archive,top_k=1,max_hops=0,
              candidate_pool=2,evidence_ranker=FakeRanker(),max_new_tokens=1,temperature=0,
              prompt_style='compact')
    assert hits[0]['id']=='b'
    assert old_keys==set(model.state_dict())
    file=tmp_path/'old.pt'
    model.save(file,tok)
    restored=IntegratedLM.load(file,tok)
    assert set(restored.state_dict())==old_keys


def test_trainable_ranker_contrastive_gradients_and_unseen_text():
    tok=LayeredTokenizer()
    torch.manual_seed(7)
    ranker=EvidenceRanker(tok.vocab_size,width=12,key_dim=16)
    q,ql=ranker._padded(['Color of EMMA?'],tok,'cpu',48)
    d,dl=ranker._padded(['EMMA is blue.','NORA is red.','LORA is blue.'],tok,'cpu',48)
    loss,scores=ranker.contrastive_loss(q,ql,d[None],dl[None],
                                        torch.tensor([0],dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss) and scores.shape==(1,3)
    assert ranker.embedding.weight.grad.abs().sum().item()>0
    hits=[{'id':str(i),'text':txt} for i,txt in enumerate(['EMMA is blue.','NORA is red.'])]
    result=ranker.rerank('Color of EMMA?',hits,tok,top_k=2)
    assert {r['id'] for r in result}=={'0','1'}
    assert all('learned_relevance' in r for r in result)


def test_recurrent_parallel_prefill_matches_serial_with_real_neural_memory():
    tok=LayeredTokenizer()
    torch.manual_seed(42)
    m=IntegratedLM(tiny()).eval()
    m.remember('A short fact.',tok,source='test')
    inp=torch.tensor([tok.encode('Color of MIRA?')],dtype=torch.long)
    with torch.no_grad():
        fast,states=m._recurrent_prefill(inp)
        old=[None]*len(m.blocks)
        for token in inp.unbind(1):
            serial,old=m._recurrent_step(token[:,None],old)
    assert torch.allclose(fast,serial,atol=2e-5,rtol=2e-5)
    for x,y in zip(states,old):
        assert torch.allclose(x,y,atol=2e-5,rtol=2e-5)


def test_ranker_checkpoint_is_independent_and_tokenizer_verified(tmp_path):
    tok=LayeredTokenizer()
    ranker=EvidenceRanker(tok.vocab_size,width=12,key_dim=16)
    p=tmp_path/'ranker.pt'
    ranker.save(p,tok)
    loaded=EvidenceRanker.load(p,tok)
    assert set(loaded.state_dict())==set(ranker.state_dict())
    for name,weight in ranker.state_dict().items():
        assert torch.equal(loaded.state_dict()[name],weight)
    class BadTokenizer:
        vocab_size=258
        def model_contract(self):return {'different':'tokenizer'}
    with pytest.raises(ValueError,match='tokenizer'):
        EvidenceRanker.load(p,BadTokenizer())
