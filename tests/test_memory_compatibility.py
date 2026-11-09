"""End-to-end neural memory compatibility gates; no synthetic answers supplied at inference."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import pytest
import torch

from model import IntegratedConfig, IntegratedLM
from neural_tokenizer import LayeredTokenizer


def setup():
    torch.manual_seed(31)
    torch.set_num_threads(2)
    t = LayeredTokenizer()
    cfg = IntegratedConfig(vocab_size=t.vocab_size, d_model=24, n_layers=1,
        n_heads=4, n_kv_heads=2, ffn_hidden=48, block_size=48,
        memory_tokens=48, memory_slots=8, regions=2, active_regions=2)
    return IntegratedLM(cfg), t


def sample_query(t):
    return torch.tensor([t.encode('Which color?')], dtype=torch.long)


def test_stored_memory_survives_ordinary_training_and_restart():
    model, tok = setup()
    first_id = model.remember('The secret color is cyan.', tok, source='user', region=1)
    assert first_id == 1
    model.memory.link(1, model.remember('The bird is quiet.', tok, source='user'), relation='related')
    saved_bodies = model.memory.states.detach().clone()
    saved_keys = model.memory.keys.detach().clone()
    initial_codec = model._encoder_fingerprint()
    before = model(sample_query(tok)).logits.detach().clone()
    # AdamW is created AFTER writing memories: frozen codec must stay frozen.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):
        loss = model(sample_query(tok), targets=sample_query(tok)).loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert model._encoder_fingerprint() == initial_codec
    assert torch.equal(saved_bodies, model.memory.states)
    assert torch.equal(saved_keys, model.memory.keys)
    assert not torch.equal(before, model(sample_query(tok)).logits.detach())
    model.remember('This is a third fact.', tok, source='user')
    model.memory.consolidate(1, reviewer='test', evidence='source-attested record')
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'model.pt'; model.save(p, tok)
        fresh = IntegratedLM.load(p, tok)
        assert fresh.memory.count == 3
        assert fresh.memory.provenance[0]['status'] == 'consolidated'
        assert fresh.memory.edges == [{'source_id': 1, 'target_id': 2, 'relation': 'related'}]
        assert torch.equal(fresh(sample_query(tok)).logits, model(sample_query(tok)).logits)
        assert fresh._encoder_fingerprint() == initial_codec


def test_unapproved_codec_mutation_fails_without_destroying_memory():
    model, tok = setup()
    model.remember('Persistent information.', tok, source='user')
    with torch.no_grad(): model.memory_state_proj.weight.add_(0.2)
    with pytest.raises(RuntimeError, match='compatib|migrat|changed'):
        model(sample_query(tok))


def test_validated_codec_upgrade_retains_ids_graph_and_changes_version():
    model, tok = setup()
    model.remember('The color is cyan.', tok, source='user', region=1)
    model.remember('The color is teal.', tok, source='user', region=0)
    model.memory.link(1, 2, relation='similar-but-distinct')
    model.memory.consolidate(1, reviewer='reviewer', evidence='checked')
    old = model.memory.states[:model.memory.count].detach().clone()
    logits_before = model(sample_query(tok)).logits.detach().clone()
    embedding = copy.deepcopy(model.memory_token_embedding)
    encoder = copy.deepcopy(model.memory_encoder)
    projection = copy.deepcopy(model.memory_state_proj)
    # A known invertible representation change allows objective migration validation.
    q, _ = torch.linalg.qr(torch.randn(model.config.d_model, model.config.d_model))
    with torch.no_grad():
        projection.weight.copy_(q @ projection.weight)
        projection.bias.copy_(q @ projection.bias)
    anchors = ['Cats sleep on cushions.', 'Engineers study systems.', '红色的苹果。',
      'The bright bird flies.', 'Three boats cross the river.', '今天的天气不错。',
      'Memory is part of the model.', 'Different words carry meaning.',
      'Time moves forward.', 'Two trains reach the station.', 'A book on the table.',
      'Night follows the afternoon.', 'A student reads a map.']
    result = model.upgrade_memory_codec(
        tok, embedding, encoder, projection,
        train_anchors=anchors[:10], validation_anchors=anchors[10:],
        min_validation_cosine=0.98,
    )
    assert result['version'] == 2
    assert result['validation_cosine'] >= 0.98
    assert model.memory.provenance[0]['id'] == 1
    assert model.memory.provenance[0]['status'] == 'consolidated'
    assert model.memory.edges == [{'source_id': 1, 'target_id': 2, 'relation': 'similar-but-distinct'}]
    assert torch.equal(old, model.memory.states[:2])
    assert torch.equal(logits_before, model(sample_query(tok)).logits.detach())
    assert torch.equal(model.memory.versions[:2], torch.ones(2, dtype=torch.long))
    assert result['validation_relative_error'] < 0.10
    model.remember('More new knowledge.', tok, source='user')
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'm.pt'; model.save(f,tok)
        restored = IntegratedLM.load(f,tok)
        assert restored.memory.current_version == 2
        assert torch.equal(restored(sample_query(tok)).logits, model(sample_query(tok)).logits)
        assert len(restored.memory.migration_history) == 1


def test_failed_migration_is_atomic():
    model, tok = setup()
    model.remember('The river is long.', tok, source='user')
    original = {k:v.clone() for k,v in model.state_dict().items()}
    projection = copy.deepcopy(model.memory_state_proj)
    with torch.no_grad(): projection.weight.zero_(); projection.bias.zero_()
    with pytest.raises(ValueError):
        model.upgrade_memory_codec(tok, copy.deepcopy(model.memory_token_embedding),
            copy.deepcopy(model.memory_encoder), projection,
            train_anchors=['This is a memory.', 'Another example here.'],
            validation_anchors=['This separate anchor must be validated.'],
            min_validation_cosine=0.95)
    assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
    assert model.memory.current_version == 1
    assert torch.isfinite(model(sample_query(tok)).logits).all()


def test_legacy_checkpoint_load_schema1(tmp_path):
    # Hand-construct original schema1 checkpoint using the source architecture.
    import json
    model, tok = setup()
    model.remember('Old memory.', tok, source='legacy')
    model.save(tmp_path/'new.pt', tok)
    saved = torch.load(tmp_path/'new.pt', weights_only=True)
    saved['meta']['schema'] = 1
    state = saved['state_dict']
    state.pop('memory_token_embedding.weight')
    state.pop('memory.versions')
    state.pop('codec_to_memory_matrix')
    state.pop('codec_to_memory_bias')
    saved['meta']['memory'] = {k:v for k,v in saved['meta']['memory'].items()
        if k in ('provenance','encoder_fingerprint','edges')}
    # Legacy fingerprint uses main token_embedding in the codec.
    saved['meta']['memory']['encoder_fingerprint'] = model._legacy_encoder_fingerprint()
    torch.save(saved,tmp_path/'legacy.pt')
    loaded = IntegratedLM.load(tmp_path/'legacy.pt',tok)
    assert loaded.memory.count==1
    assert loaded.memory.current_version==1
    assert torch.isfinite(loaded(sample_query(tok)).logits).all()
