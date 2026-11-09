"""v0.6.2 direct neural graph state integration regressions."""
import copy

import pytest
import torch

from model import (IntegratedConfig, IntegratedLM, ContextArchive,
                   GraphStateBridge, GraphRecurrentEngine)
from neural_tokenizer import LayeredTokenizer


def setup_model():
    tok = LayeredTokenizer()
    cfg = IntegratedConfig(vocab_size=258,d_model=24,ffn_hidden=48,
        n_heads=4,n_kv_heads=2,n_layers=2,block_size=256,
        backbone='recurrent',memory_slots=8,memory_tokens=64,
        regions=2,active_regions=2)
    torch.manual_seed(62)
    model=IntegratedLM(cfg).eval()
    archive=ContextArchive()
    archive.add('Mira repaired the cobalt drone on Tuesday.',source='repair',doc_id='repair')
    archive.add('The cobalt drone belongs to Lab Blue.',source='inventory',doc_id='inventory')
    archive.link('repair','inventory',relation='same_device')
    bridge=GraphStateBridge(cfg.d_model,cfg.n_layers)
    return model,tok,archive,bridge


def test_no_evidence_keeps_original_recurrent_state():
    model,tok,archive,bridge=setup_model()
    archive.delete('repair'); archive.delete('inventory')
    engine=GraphRecurrentEngine(model,tok,archive,bridge)
    question='What day was the drone repaired?'
    prepared=engine.prepare(question)
    original,states=model._recurrent_prefill(prepared.query_ids)
    assert prepared.hits==[] and prepared.scores is None
    assert all(torch.allclose(a,b) for a,b in zip(prepared.states,states))


def test_recovered_context_directly_changes_hidden_states_without_prompt_text():
    model,tok,archive,bridge=setup_model()
    engine=GraphRecurrentEngine(model,tok,archive,bridge)
    question='When was the cobalt drone repaired?'
    with torch.no_grad():
        x=engine.prepare(question,top_k=2)
        _,plain_states=model._recurrent_prefill(x.query_ids)
        assert all(torch.norm(a-b)>0 for a,b in zip(x.states,plain_states))
        # Replacing raw evidence changes the bridge output, but NOT the question input.
        different=[{**h,'text':'Cobalt drone was repaired on Saturday.'}
                   if h['id']=='repair' else h for h in x.hits]
        y=engine.prepare(question,top_k=2,explicit_hits=different)
        assert torch.equal(x.query_ids,y.query_ids)
        assert any(torch.norm(a-b)>1e-7 for a,b in zip(x.states,y.states))
        t=torch.tensor([[tok.encode(' ')[0]]])
        logit_x,_=model._recurrent_step(t,x.states)
        logit_y,_=model._recurrent_step(t,y.states)
        assert not torch.allclose(logit_x,logit_y,atol=1e-9,rtol=1e-9)


def test_graph_edge_changes_neural_messages():
    model,tok,archive,bridge=setup_model()
    engine=GraphRecurrentEngine(model,tok,archive,bridge)
    question='When was the cobalt drone repaired?'
    with torch.no_grad():
        linked=engine.prepare(question,top_k=2)
        archive.edges=[]
        unlinked=engine.prepare(question,top_k=2,explicit_hits=linked.hits)
    assert not torch.allclose(linked.states[0],unlinked.states[0],atol=1e-9,rtol=1e-9)


def test_training_answer_loss_reaches_neural_graph_and_recurrent_weights():
    model,tok,archive,bridge=setup_model()
    model.train();bridge.train()
    engine=GraphRecurrentEngine(model,tok,archive,bridge)
    loss,prepared=engine.answer_loss('When was the cobalt drone repaired?',
             'Tuesday.',top_k=2,gold_index=0)
    assert torch.isfinite(loss) and len(prepared.hits)==2
    loss.backward()
    assert bridge.encoder.weight_ih_l0.grad.abs().sum()>0
    assert bridge.state_values.weight.grad.abs().sum()>0
    assert model.blocks[0].mixer.input_proj.weight.grad.abs().sum()>0


def test_existing_model_state_dict_and_checkpoint_stay_identical(tmp_path):
    model,tok,archive,bridge=setup_model()
    before={k:v.clone() for k,v in model.state_dict().items()}
    eng=GraphRecurrentEngine(model,tok,archive,bridge)
    eng.generate('When was cobalt drone repaired?',max_new_tokens=2,top_k=2,eos_id=tok.EOS_ID)
    answer,hits=model.generate_with_graph('When was cobalt drone repaired?',tok,archive,bridge,
                                          max_new_tokens=2,top_k=2,eos_id=tok.EOS_ID)
    assert isinstance(answer,str) and len(hits)==2
    assert set(before)==set(model.state_dict())
    assert all(torch.equal(t,model.state_dict()[k]) for k,t in before.items())
    lm_ckpt=tmp_path/'base.pt'; bridge_ckpt=tmp_path/'bridge.pt'
    model.save(lm_ckpt,tok)
    bridge.save(bridge_ckpt,tok,model.config)
    reloaded=IntegratedLM.load(lm_ckpt,tok)
    restored=GraphStateBridge.load(bridge_ckpt,tok,reloaded.config)
    assert set(reloaded.state_dict())==set(before)
    assert all(torch.equal(a,restored.state_dict()[k]) for k,a in bridge.state_dict().items())
    class WrongTokenizer:
        vocab_size=258
        def model_contract(self):return {'bad':'contract'}
    with pytest.raises(ValueError,match='incompatible'):
        GraphStateBridge.load(bridge_ckpt,WrongTokenizer(),reloaded.config)


def test_no_transformer_or_graph_bridge_on_legacy_backbone():
    model,tok,archive,bridge=setup_model()
    model.config.backbone='transformer'
    with pytest.raises(ValueError,match='recurrent'):
        GraphRecurrentEngine(model,tok,archive,bridge)


def test_recurrent_continue_agrees_with_step_reference_and_uses_initial_state():
    model,tok,archive,bridge=setup_model()
    model.eval()
    ids=torch.tensor([[22,31,87,102,33,18,43]],dtype=torch.long)
    zero=[torch.randn(1,model.config.d_model) for _ in model.blocks]
    with torch.no_grad():
        out,final=model.recurrent_continue(ids,zero)
        reference=[]; states=zero
        for t in ids.unbind(dim=1):
            logits,states=model._recurrent_step(t[:,None],states)
            reference.append(logits)
        comparison=torch.stack(reference,dim=1)
        assert torch.allclose(out.logits,comparison,atol=1e-5,rtol=1e-5)
        assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(final,states))
