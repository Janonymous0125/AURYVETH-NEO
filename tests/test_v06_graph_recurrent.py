"""v0.6 regression coverage: causality, stateful decoding and lossless archive."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from model import IntegratedLM, IntegratedConfig, ContextArchive
from neural_tokenizer import LayeredTokenizer


def tiny():
    return IntegratedConfig(vocab_size=258, d_model=24, ffn_hidden=48,
                            n_heads=4, n_kv_heads=2, n_layers=2, block_size=256,
                            memory_slots=8, regions=2, active_regions=2,
                            backbone='recurrent', memory_tokens=64)


@pytest.mark.parametrize('planner_steps', [0, 3])
def test_recurrent_incremental_parity_and_no_future_leakage(planner_steps):
    torch.manual_seed(6)
    m=IntegratedLM(tiny()).eval()
    if planner_steps:
        m.add_planner_to_existing_model(planner_steps)
        m.eval()
        with torch.no_grad():
            for block in m.blocks:
                block.planner.out.weight.normal_(std=0.02)
    ids=torch.randint(0,258,(1,10))
    with torch.no_grad():
        logits=m(ids).logits
        states=[None]*len(m.blocks)
        step_logits=[]
        for token in ids.unbind(1):
            one,states=m._recurrent_step(token[:,None],states)
            step_logits.append(one[:,None])
        assert torch.allclose(logits,torch.cat(step_logits,dim=1),atol=1e-5,rtol=1e-5)
        alternate=ids.clone()
        alternate[0,-1]= (alternate[0,-1]+1)%258
        assert torch.allclose(logits[:,:-1],m(alternate).logits[:,:-1],atol=1e-6)
        output=m.generate(ids,max_new_tokens=3,temperature=0,eos_id=None)
        manual=ids.clone()
        for _ in range(3):
            next_token=m(manual).logits[:,-1].argmax(-1,keepdim=True)
            manual=torch.cat([manual,next_token],dim=1)
        assert torch.equal(output,manual)


def test_graph_hops_original_text_integrity_deletion_and_reload(tmp_path):
    archive=ContextArchive()
    a=archive.add('Lab Blue owns Drone-7.',source='note-a',doc_id='lab')
    b=archive.add('Drone-7 was repaired by Mira on Monday.',source='note-b',doc_id='repair')
    c=archive.add('Mira changed the battery during the repair.',source='note-c',doc_id='battery')
    archive.link(a,b,relation='owns-device-repaired')
    archive.link(b,c,relation='same-repair')
    hits=archive.retrieve('Lab Blue',top_k=1,max_hops=2,max_total=3)
    assert [h['id'] for h in hits]==['lab','repair','battery']
    assert hits[1]['text']=='Drone-7 was repaired by Mira on Monday.'
    target=tmp_path/'archive.json'
    archive.save(target)
    restored=ContextArchive.load(target)
    assert restored.retrieve('Lab Blue',top_k=1,max_hops=2,max_total=3)==hits
    data=json.loads(target.read_text())
    data['documents'][0]['text']='tampered'
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='integrity'):
        ContextArchive.load(target)
    restored.delete('repair')
    assert not restored.edges


def test_generate_with_archive_yields_evidence_not_canned_answers():
    torch.manual_seed(7)
    m=IntegratedLM(tiny()).eval()
    tok=LayeredTokenizer()
    a=ContextArchive()
    a.add('Mira repaired Drone-7 on Monday.',source='test')
    answer,hits=m.generate_with_archive('When did Mira repair Drone-7?',tok,a,
                                        max_new_tokens=2,temperature=0,max_evidence_tokens=60)
    assert isinstance(answer,str)
    assert len(hits)==1 and hits[0]['text']=='Mira repaired Drone-7 on Monday.'
    # No assertion of correctness: an untrained LM has not learned evidence QA.


def test_codec_fingerprint_is_cached_but_detects_mutation():
    m=IntegratedLM(tiny())
    tok=LayeredTokenizer()
    m.remember('An archived fact.',tok,source='test')
    value=m._encoder_fingerprint()
    assert m._fingerprint_cache[1]==value
    before=m._fingerprint_cache
    m(torch.tensor([[2,3,4]],dtype=torch.long))
    assert m._fingerprint_cache is before
    with torch.no_grad():
        m.memory_state_proj.weight.add_(0.01)
    with pytest.raises(RuntimeError, match='codec changed'):
        m(torch.tensor([[2,3,4]],dtype=torch.long))


def test_recurrent_training_step_gradients():
    m=IntegratedLM(tiny())
    m.train()
    x=torch.tensor([[10,11,12,13]],dtype=torch.long)
    y=torch.tensor([[11,12,13,14]],dtype=torch.long)
    loss=m(x,targets=y,return_logits=False,loss_chunk_size=8).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert m.blocks[0].mixer.input_proj.weight.grad is not None
    assert torch.isfinite(m.blocks[0].mixer.input_proj.weight.grad).all()


def test_untrained_retrieval_controller_fails_closed():
    m=IntegratedLM(tiny())
    with pytest.raises(RuntimeError,match='trained'):
        m.retrieval_probability(torch.tensor([[1,2,3]],dtype=torch.long))
