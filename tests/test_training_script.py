import json, tempfile
from pathlib import Path
import unittest
import torch
from model import IntegratedConfig, IntegratedLM
from neural_tokenizer import LayeredTokenizer
from scripts.train import load_rows,one_loss

class TrainingSmoke(unittest.TestCase):
    def test_real_optimization_and_neural_memory_gradients(self):
        torch.manual_seed(19)
        t=LayeredTokenizer()
        cfg=IntegratedConfig(vocab_size=t.vocab_size,block_size=48,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,ffn_hidden=64,memory_tokens=32)
        model=IntegratedLM(cfg);model.train()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.jsonl'
            path.write_text(json.dumps({'context':['Passphrase is emerald.'],'prompt':'Passphrase? ','answer':'emerald.'})+'\n')
            rows=load_rows(path,t,cfg)
            initial=float(one_loss(model,rows[0],'cpu').detach())
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.004)
            for _ in range(20):
                loss=one_loss(model,rows[0],'cpu');optimizer.zero_grad();loss.backward();optimizer.step()
            final=float(one_loss(model,rows[0],'cpu').detach())
        self.assertLess(final,initial)

if __name__=='__main__':unittest.main()

class TrainedMemoryCausality(unittest.TestCase):
    def test_two_facts_same_prompt_different_generated_words(self):
        """Actual end-to-end gradient training, then frozen-weight neural memory writes."""
        torch.manual_seed(15);torch.set_num_threads(2)
        t=LayeredTokenizer()
        cfg=IntegratedConfig(vocab_size=258,block_size=64,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,
                             ffn_hidden=96,memory_slots=16,memory_tokens=64,regions=2,active_regions=2,active_memories=2)
        m=IntegratedLM(cfg);m.train()
        rows=load_rows(Path(__file__).with_name('memory_causal_training.jsonl'),t,cfg)
        opt=torch.optim.AdamW(m.parameters(),lr=0.004)
        for i in range(140):
            loss=one_loss(m,rows[i%2],'cpu')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        m.eval()
        outputs=[]
        for source in ('The secret color is amber.','The secret color is cobalt.'):
            fresh=IntegratedLM(cfg);fresh.load_state_dict(m.state_dict());fresh.eval()
            weight_before={k:v.detach().clone() for k,v in fresh.named_parameters()}
            fresh.remember(source,t,source='controlled-training')
            with torch.no_grad():
                generated=fresh.generate(torch.tensor([t.encode('Secret color? ')]),max_new_tokens=8,temperature=0)
            self.assertTrue(all(torch.equal(v,weight_before[k]) for k,v in fresh.named_parameters()))
            outputs.append(t.decode(generated[0].tolist()))
        self.assertTrue(outputs[0].startswith('Secret color? amber.'))
        self.assertTrue(outputs[1].startswith('Secret color? cobalt.'))


def test_resume_training_with_persistent_memory_and_validation(tmp_path, monkeypatch):
    """Actual CLI continuation, including memory contents, graph and validation."""
    import sys
    from scripts import train
    torch.manual_seed(9)
    tokenizer = LayeredTokenizer()
    config = IntegratedConfig(vocab_size=258,block_size=40,d_model=24,n_layers=1,
        n_heads=4,n_kv_heads=2,ffn_hidden=48,memory_tokens=40,memory_slots=8,regions=2,active_regions=2)
    prior = IntegratedLM(config)
    prior.remember('Stored information persists.',tokenizer,source='user')
    bodies = prior.memory.states.detach().clone()
    source = tmp_path/'source.pt'
    prior.save(source,tokenizer)
    (tmp_path/'train.txt').write_text('This passage teaches ordinary language.\n'*6,encoding='utf-8')
    (tmp_path/'validation.txt').write_text('A completely different sentence.\n'*4,encoding='utf-8')
    out = tmp_path/'new_run'
    monkeypatch.setattr(sys,'argv',['train','--resume-checkpoint',str(source),
        '--corpus',str(tmp_path/'train.txt'),'--validation-corpus',str(tmp_path/'validation.txt'),
        '--allow-small-dataset','--steps','2','--lr','0.00001','--out',str(out),'--log-every','1'])
    train.main()
    restored = IntegratedLM.load(out/'model.pt',tokenizer)
    import json
    metrics = json.loads((out/'train_metrics.json').read_text())
    assert metrics['validation_rows_available'] > 0
    assert metrics['validation_loss'] is not None
    assert torch.equal(restored.memory.states,bodies)
    assert restored.memory.count == 1
