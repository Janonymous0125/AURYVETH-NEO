"""Regression coverage for checkpoint disk exhaustion and Windows cleanup failure."""
import errno
from pathlib import Path
import types
import warnings

import pytest
import torch

from scripts import train


def test_checkpoint_atomic_replace_and_load(tmp_path):
    target=tmp_path/'training_state.pt'
    torch.save({'previous': 7},target)
    train.atomic_torch_save({'next':torch.arange(9)},target)
    assert torch.equal(torch.load(target,weights_only=True)['next'],torch.arange(9))
    assert list(tmp_path.glob('.training-*.pt'))==[]


def test_checkpoint_insufficient_space_fails_before_temporary_write(monkeypatch,tmp_path):
    target=tmp_path/'training_state.pt'
    torch.save({'previous':7},target)
    monkeypatch.setattr(train.shutil,'disk_usage',lambda p:types.SimpleNamespace(free=100))
    with pytest.raises(OSError,match='previous checkpoint has not been touched'):
        train.atomic_torch_save({'next':torch.ones(4)},target)
    assert torch.load(target,weights_only=True)=={'previous':7}
    assert list(tmp_path.glob('.training-*.pt'))==[]


def test_checkpoint_original_failure_visible_if_cleanup_locked(monkeypatch,tmp_path):
    target=tmp_path/'training_state.pt'
    torch.save({'previous':7},target)
    original_unlink=train.os.unlink
    def locked_unlink(path,*args,**kwargs):
        if str(path).endswith('.pt') and '.training-' in str(path):
            raise PermissionError(32,'file in use')
        return original_unlink(path,*args,**kwargs)
    def fake_save(payload,path):
        Path(path).write_bytes(b'partial')
        raise OSError(errno.ENOSPC,'no space on device')
    monkeypatch.setattr(train.torch,'save',fake_save)
    monkeypatch.setattr(train.os,'unlink',locked_unlink)
    monkeypatch.setattr(train.time,'sleep',lambda _:None)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        with pytest.raises(RuntimeError,match='no space on device') as err:
            train.atomic_torch_save({'next':torch.ones(4)},target)
    assert isinstance(err.value.__cause__,OSError)
    assert 'Previous successful checkpoint was not replaced' in str(err.value)
    assert torch.load(target,weights_only=True)=={'previous':7}
    assert any('could not be removed' in str(w.message) for w in recorded)


def test_checkpoint_storage_estimator_deduplicates_aliases():
    a=torch.ones(8)
    estimate=train._serialized_storage_bytes({'a':a,'b':[a,a.view(2,4)]})
    assert estimate==a.untyped_storage().nbytes()


def test_model_save_failure_preserves_old_checkpoint_even_if_temporary_locked(monkeypatch,tmp_path):
    from model import IntegratedLM, IntegratedConfig
    from neural_tokenizer import LayeredTokenizer
    model=IntegratedLM(IntegratedConfig(vocab_size=258,block_size=32,d_model=32,
        n_layers=1,n_heads=4,n_kv_heads=2,ffn_hidden=64,memory_slots=4,
        memory_tokens=8,regions=2,active_memories=2))
    tok=LayeredTokenizer()
    target=tmp_path/'model.pt'
    model.save(target,tok)
    original_content=target.read_bytes()
    import model.integrated as integrated
    original_unlink=integrated.os.unlink
    def locked_unlink(path,*args,**kwargs):
        if '.checkpoint-' in str(path):
            raise PermissionError(32,'file in use')
        return original_unlink(path,*args,**kwargs)
    def broken_save(obj,path):
        Path(path).write_bytes(b'part')
        raise OSError(errno.ENOSPC,'disk full')
    monkeypatch.setattr(integrated.torch,'save',broken_save)
    monkeypatch.setattr(integrated.os,'unlink',locked_unlink)
    monkeypatch.setattr(integrated.time,'sleep',lambda _:None)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        with pytest.raises(OSError,match='disk full'):
            model.save(target,tok)
    assert target.read_bytes()==original_content
    assert any('Could not remove temporary checkpoint' in str(w.message) for w in recorded)
