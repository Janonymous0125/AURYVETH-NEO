import copy
import json
import pytest
import torch
from model import IntegratedConfig,IntegratedLM
from model.entity_memory import EntityMemoryRouter
from model.memory import NeuralMemoryBank
from neural_tokenizer import LayeredTokenizer


def setup_router():
    tok=LayeredTokenizer()
    m=IntegratedLM(IntegratedConfig(vocab_size=258,d_model=16,n_layers=2,n_heads=2,n_kv_heads=1,
        ffn_hidden=32,block_size=128,memory_slots=8,memory_tokens=64,regions=4,active_regions=4,
        active_memories=1,backbone='recurrent')).eval()
    router=EntityMemoryRouter(m,tok)
    router.remember('QX7000 is violet.',entity='QX7000',source='test',region=0)
    router.remember('QX7001 is red.',entity='QX7001',source='test',region=1)
    router.remember('QX7000 has a drone.',entity='QX7000',source='test',region=2)
    return router,m,tok


def test_entity_view_preserves_bodies_metadata_edges_and_original_bank():
    router,m,tok=setup_router();original=m.memory;before=copy.deepcopy(m.state_dict())
    original.link(1,3,relation='same_entity');original.link(1,2,relation='different_entity')
    original.consolidate(1,reviewer='test',evidence='fixture')
    assert router.selected_ids('QX7000')==(1,3)
    with router.scope('QX7000') as selected:
        assert selected==(1,3) and m.memory.count==2
        assert torch.equal(m.memory.states[0],original.states[0])
        assert torch.equal(m.memory.states[1],original.states[2])
        assert m.memory.edges==[{'source_id':1,'target_id':2,'relation':'same_entity'}]
        assert m.memory.provenance[0]['status']=='consolidated'
        assert m.memory.snapshot()['provenance'][1]['id']==2
    assert m.memory is original and original.count==3
    # Only the explicit consolidation/links above changed metadata, never weights.
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items() if not k.startswith('memory.'))


def test_missing_exact_entity_matches_memory_off_without_answer_lookup():
    router,m,tok=setup_router();ids=torch.tensor([tok.encode('Question? Answer: ')])
    original=m.memory
    with original.suspended():expected=m.generate(ids,max_new_tokens=3,temperature=0)
    actual=router.generate(ids,entity='QX7002',max_new_tokens=3,temperature=0)
    assert torch.equal(actual,expected) and m.memory is original and not original._suspended
    assert router.selected_ids('qx7000')==()  # Entity identity is explicitly case-sensitive.
    assert set(router.entities.values())=={'QX7000','QX7001'}  # No answer text in the index.


def test_unique_recall_abstains_before_decoding_when_missing_or_ambiguous(monkeypatch):
    router,m,tok=setup_router();original=m.memory
    ids=torch.tensor([tok.encode('Question? Answer: ')])
    kwargs={'max_new_tokens':2,'temperature':0}
    expected=router.generate(ids,entity='QX7001',**kwargs)
    assert torch.equal(router.generate_unique(ids,entity='QX7001',**kwargs),expected)

    def unexpected_generation(*args,**kwargs):
        pytest.fail('Lookup must abstain before model generation')

    monkeypatch.setattr(m,'generate',unexpected_generation)
    with pytest.raises(LookupError,match='No memory record'):
        router.generate_unique(ids,entity='QX7002',**kwargs)
    with pytest.raises(LookupError,match='Multiple memory records'):
        router.generate_unique(ids,entity='QX7000',**kwargs)
    with original.suspended(), pytest.raises(LookupError,match='Memory is disabled'):
        router.generate_unique(ids,entity='QX7001',**kwargs)
    assert m.memory is original and not original._suspended and not router._scoping


def test_scope_restores_bank_on_error_and_respects_external_ablation():
    router,m,tok=setup_router();original=m.memory
    with pytest.raises(ValueError,match='deliberate'):
        with router.scope('QX7000'):
            raise ValueError('deliberate')
    assert m.memory is original
    with original.suspended():
        with router.scope('QX7000'):
            assert m.memory._suspended
    assert not original._suspended


def test_scoped_writes_saves_retagging_and_nested_views_are_rejected(tmp_path):
    router,m,tok=setup_router()
    with router.scope('QX7000'):
        with pytest.raises(RuntimeError):router.remember('New fact.',entity='QX7000',source='test')
        with pytest.raises(RuntimeError):router.save(tmp_path/'bad')
        with pytest.raises(RuntimeError):
            with router.scope('QX7001'):pass
    with pytest.raises(ValueError):router.tag(1,'other-entity')
    m.memory=NeuralMemoryBank(m.config)
    with pytest.raises(RuntimeError,match='replaced'):router.selected_ids('QX7000')
    with pytest.raises(RuntimeError,match='replaced'):router.remember('New fact.',entity='new',source='test')
    with pytest.raises(RuntimeError,match='replaced'):router.save(tmp_path/'stale')
    assert m.memory.count==0 and not (tmp_path/'stale').exists()


def test_entity_bundle_roundtrip_binds_index_and_model(tmp_path):
    router,m,tok=setup_router();directory=tmp_path/'bundle';router.save(directory)
    loaded=EntityMemoryRouter.load(directory,tok)
    assert loaded.entities==router.entities and loaded.model.memory.count==3
    ids=torch.tensor([tok.encode('Question? Answer: ')])
    a=router.generate(ids,entity='QX7000',max_new_tokens=2,temperature=0)
    b=loaded.generate(ids,entity='QX7000',max_new_tokens=2,temperature=0)
    assert torch.equal(a,b)
    payload=json.loads((directory/'entities.json').read_text());payload['entities']['1']='QX7001'
    (directory/'entities.json').write_text(json.dumps(payload),encoding='utf8')
    with pytest.raises(ValueError,match='integrity'):EntityMemoryRouter.load(directory,tok)
