"""Architecture acceptance tests: memory lives IN model forward, not app-side RAG."""
import tempfile
from pathlib import Path
import unittest
import torch
from model.integrated import IntegratedLM, IntegratedConfig
from neural_tokenizer import LayeredTokenizer


class IntegratedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.cfg=IntegratedConfig(vocab_size=258,block_size=64,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,ffn_hidden=64,
                                  memory_slots=8,memory_tokens=32,regions=3,active_memories=2)
        self.model=IntegratedLM(self.cfg)
        self.model.eval()
        self.tok=LayeredTokenizer()

    def test_lm_loss_backward_and_generate(self):
        ids=torch.tensor([[ord(x) for x in 'hello world']],dtype=torch.long)
        output=self.model(ids[:,:-1],targets=ids[:,1:])
        self.assertEqual(tuple(output.logits.shape),(1,10,258))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertGreater(float(self.model.blocks[0].attn.q_proj.weight.grad.abs().sum()),0)
        result=self.model.generate(ids[:,:4],max_new_tokens=3,temperature=0)
        self.assertEqual(tuple(result.shape),(1,7))

    def test_model_memories_modify_logits_inside_forward_before_attention(self):
        x=torch.tensor([[ord(z) for z in 'Question?']],dtype=torch.long)
        with torch.no_grad():
            baseline=self.model(x).logits.clone()
            old_params={n:p.detach().clone() for n,p in self.model.named_parameters()}
            memory_id=self.model.remember('The answer is cobalt.',self.tok,source='test',region=1)
            self.assertEqual(memory_id,1)
            changed=self.model(x).logits.clone()
        self.assertGreater(float((baseline-changed).abs().max()),1e-8)
        self.assertTrue(all(torch.equal(p,old_params[n]) for n,p in self.model.named_parameters()))
        self.assertGreater(self.model.memory.count,0)
        self.assertGreater(int(self.model.blocks[0].memory_read.calls),0)

    def test_memory_ablation_changes_actual_output_and_does_not_change_weights(self):
        self.model.remember('A river of crystal.',self.tok,source='test')
        x=torch.tensor([[71,111,63]])
        with torch.no_grad():
            withmem=self.model(x).logits
            with self.model.memory.suspended():
                nomem=self.model(x).logits
            again=self.model(x).logits
        self.assertGreater(float((withmem-nomem).abs().max()),1e-8)
        self.assertTrue(torch.equal(withmem,again))

    def test_memory_checkpoint_recovery_and_provenance(self):
        self.model.remember('火星 Mars 🚀',self.tok,source='session-1',region=2)
        x=torch.tensor([[65,66,67]])
        with torch.no_grad():before=self.model(x).logits
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            self.model.save(path,self.tok)
            restored=IntegratedLM.load(path,self.tok)
            with torch.no_grad(): after=restored(x).logits
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(restored.memory.count,1)
        self.assertEqual(restored.memory.provenance[0]['source'],'session-1')
        self.assertTrue(restored.memory.provenance[0]['status']=='provisional')

    def test_memory_bank_rejects_invalid_data_and_has_fixed_bounds(self):
        with self.assertRaises(ValueError):self.model.remember('x'*100,self.tok,source='x')
        with self.assertRaises(ValueError):self.model.remember('test',self.tok,source='x',region=99)
        with self.assertRaises(ValueError):self.model.remember('test',self.tok,source='')
        x=torch.tensor([[258]])
        with self.assertRaises(ValueError):self.model(x)
        with self.assertRaises(ValueError):self.model(torch.zeros((1,0),dtype=torch.long))

    def test_supervised_memory_training_gradient(self):
        # Differentiable in-network support memories; no app-side answer lookup.
        prompt=torch.tensor([[ord(z) for z in 'Code?']])
        target=torch.tensor([[ord(z) for z in 'ode?A']])
        support=torch.tensor([[[ord(z) for z in 'Code is A']]])
        support_lengths=torch.tensor([[9]])
        out=self.model(prompt,targets=target,support_ids=support,support_lengths=support_lengths)
        out.loss.backward()
        self.assertGreater(float(self.model.memory_encoder.weight_ih_l0.grad.abs().sum()),0)
        self.assertGreater(float(self.model.blocks[0].memory_read.to_value.weight.grad.abs().sum()),0)

    def test_causal_tokens_do_not_see_later_tokens_even_via_memory_selection(self):
        self.model.remember('This is persistent memory.',self.tok,source='s')
        a=torch.tensor([[65,66,67,68]])
        b=torch.tensor([[65,66,77,99]])
        with torch.no_grad():
            u=self.model(a).logits
            v=self.model(b).logits
        self.assertTrue(torch.allclose(u[:,:2],v[:,:2],atol=1e-6))

    def test_support_masks_ignore_padding(self):
        ids=torch.tensor([[65,66]])
        short=torch.tensor([[[68,69,70]]])
        padded=torch.tensor([[[68,69,70,12,13,14]]])
        with torch.no_grad():
            a=self.model(ids,support_ids=short,support_lengths=torch.tensor([[3]])).logits
            b=self.model(ids,support_ids=padded,support_lengths=torch.tensor([[3]])).logits
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_refuses_incompatible_tokenizer_and_saved_schema(self):
        self.model.remember('hello',self.tok,source='test')
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'saved.pt'
            self.model.save(f,self.tok)
            different=LayeredTokenizer();different.register_literal_token(' hello')
            with self.assertRaises(ValueError):IntegratedLM.load(f,different)


if __name__=='__main__':unittest.main()

class GraphAndCompatibility(unittest.TestCase):
    def test_graph_edges_are_data_and_survive_restart(self):
        torch.manual_seed(8)
        cfg=IntegratedConfig(vocab_size=258,block_size=64,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,ffn_hidden=64,
                             memory_slots=4,memory_tokens=32,regions=3,active_regions=1,active_memories=2)
        m=IntegratedLM(cfg);t=LayeredTokenizer()
        a=m.remember('alpha memory',t,source='a',region=0)
        b=m.remember('beta memory',t,source='b',region=2)
        weights={n:p.detach().clone() for n,p in m.named_parameters()}
        m.memory.link(a,b,relation='associated')
        self.assertTrue(all(torch.equal(p,weights[n]) for n,p in m.named_parameters()))
        query=torch.randn(1,3,cfg.d_model)
        values,mask=m.memory.select(query)
        self.assertEqual(tuple(values.shape),(1,3,2,32,32))
        self.assertTrue(bool(mask.any()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';m.save(path,t)
            restored=IntegratedLM.load(path,t)
        self.assertEqual(restored.memory.edges,[{'source_id':1,'target_id':2,'relation':'associated'}])
        with self.assertRaises(ValueError):m.memory.link(a,a,relation='self')

    def test_weight_change_after_memory_write_requires_migration(self):
        m=IntegratedLM(IntegratedConfig(vocab_size=258,block_size=32,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,
                                        ffn_hidden=64,memory_slots=4,memory_tokens=32))
        t=LayeredTokenizer();m.remember('Some remembered text.',t,source='source')
        with torch.no_grad():m.memory_encoder.weight_ih_l0.add_(0.01)
        with self.assertRaises(RuntimeError):m(torch.tensor([[1,2]],dtype=torch.long))

    def test_incompatible_same_size_tokenizer_rejected(self):
        m=IntegratedLM(IntegratedConfig(vocab_size=259,block_size=32,d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,
                                        ffn_hidden=64,memory_slots=4,memory_tokens=32))
        t1=LayeredTokenizer();t1.register_literal_token('foo')
        t2=LayeredTokenizer();t2.register_literal_token('bar')
        m.remember('foo',t1,source='s')
        with self.assertRaises(ValueError):m.remember('bar',t2,source='s')
        with self.assertRaises(ValueError):m.save('/tmp/should-not-write-this.pt',t2)
