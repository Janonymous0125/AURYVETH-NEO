"""v0.6.2 optional graph->recurrent-state bridge (no evidence prompt stuffing).

The archive is durable source storage, not pretrained weights. A differentiable
GRU reads source chunks, learns node/edge relations and relevance, then writes
an update into the running recurrent states. Core LM checkpoint format is
unchanged. Source text is data, not instructions; training and adversarial
retrieval evaluation remain necessary.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .context_archive import ContextArchive
from .evidence import EvidenceRanker


class GraphStateBridge(nn.Module):
    """Trainable graph evidence selection and direct recurrent-state fusion.

    For N small retrieved nodes: O(N^2 D) graph messaging, O(N L D^2) GRU
    source encoding. No Transformer attention over the conversation tokens.
    The node-level softmax is *retrieval scoring*, not Transformer self-attention.
    """

    def __init__(self, width: int, layers: int, *, relations: int = 64):
        super().__init__()
        if width < 2 or layers < 1 or relations < 2:
            raise ValueError('Invalid bridge dimensions')
        self.width, self.layers, self.relations = width, layers, relations
        self.encoder = nn.GRU(width, width, batch_first=True)
        self.edge_embedding = nn.Embedding(relations, width, padding_idx=0)
        self.message = nn.Linear(width, width, bias=False)
        self.graph_update = nn.Linear(width * 2, width)
        self.query_key = nn.Linear(width, width)
        self.node_key = nn.Linear(width, width)
        self.state_values = nn.Linear(width, layers * width)
        self.state_gate = nn.Linear(width * 2, layers * width)
        # Nonzero gain allows both the state bridge and language decoder to
        # learn from a grounded-answer loss starting at step 1.
        self.gain = nn.Parameter(torch.tensor(0.2))
        # A trainable relevance feature for exact entity/code matches in the
        # presence of semantically similar misleading source passages.
        self.lexical_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, query: torch.Tensor, source_embeddings: torch.Tensor,
                source_lengths: torch.Tensor, edge_types: torch.Tensor,
                *, lexical_features: torch.Tensor | None=None, return_scores=False):
        """query [B,D], sources [B,N,T,D], lengths [B,N], edges [B,N,N].

        Returns states [B,n_layers,D] whose updates are added to *existing*
        recurrent states. Training gradients reach document encoder and router.
        """
        if query.ndim != 2 or source_embeddings.ndim != 4:
            raise ValueError('Expected [B,D] query and [B,N,T,D] sources')
        b,n,t,d = source_embeddings.shape
        if (d != self.width or query.shape != (b,d) or n < 1 or t < 1 or
            source_lengths.shape != (b,n) or edge_types.shape != (b,n,n) or
            edge_types.dtype != torch.long or
            bool((edge_types < 0).any()) or bool((edge_types >= self.relations).any()) or
            bool((source_lengths < 1).any()) or bool((source_lengths > t).any())):
            raise ValueError('Invalid graph bridge tensor shape/values')
        packed = nn.utils.rnn.pack_padded_sequence(
            source_embeddings.reshape(b*n,t,d), source_lengths.reshape(-1).cpu(),
            batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        nodes = hidden[-1].reshape(b,n,d)
        # Relation-aware sparse graph message passing. Both endpoints receive
        # a message; the edge label is learned, not an answer lookup rule.
        mask = (edge_types > 0).to(nodes.dtype)
        directed = self.message(nodes)[:,None,:,:] + self.edge_embedding(edge_types)
        incoming = (directed * mask[...,None]).sum(dim=2) / mask.sum(dim=2).clamp_min(1)[...,None]
        enriched = nodes + torch.tanh(self.graph_update(torch.cat([nodes,incoming],dim=-1)))
        scores = ((F.normalize(self.query_key(query),dim=-1)[:,None,:] *
                   F.normalize(self.node_key(enriched),dim=-1)).sum(-1))
        if lexical_features is not None:
            if lexical_features.shape != scores.shape:
                raise ValueError('Lexical features must be [B,N]')
            scores = scores + self.lexical_gain * lexical_features
        probabilities = F.softmax(scores, dim=-1)
        evidence = (probabilities[...,None] * enriched).sum(dim=1)
        combined = torch.cat([query,evidence],dim=-1)
        state_delta = (torch.tanh(self.state_values(evidence)) *
                       torch.sigmoid(self.state_gate(combined))).reshape(b,self.layers,d)
        state_delta = torch.tanh(self.gain) * state_delta
        if return_scores:
            return state_delta, scores, probabilities
        return state_delta

    def save(self, path, tokenizer, config):
        if (config.d_model != self.width or config.n_layers != self.layers or
            config.backbone != 'recurrent'):
            raise ValueError('Model/bridge dimension mismatch')
        torch.save({'schema':1,'width':self.width,'layers':self.layers,
                    'relations':self.relations,
                    'vocab_size':config.vocab_size,
                    'tokenizer_contract':tokenizer.model_contract(),
                    'weights':self.state_dict()},path)

    @classmethod
    def load(cls, path, tokenizer, config, *, map_location='cpu'):
        checkpoint=torch.load(path,map_location=map_location,weights_only=True)
        if (checkpoint.get('schema') != 1 or
            checkpoint.get('tokenizer_contract') != tokenizer.model_contract() or
            checkpoint.get('width') != config.d_model or
            checkpoint.get('layers') != config.n_layers or
            checkpoint.get('vocab_size') != config.vocab_size or
            config.backbone != 'recurrent'):
            raise ValueError('Bridge checkpoint incompatible with tokenizer/model')
        bridge=cls(checkpoint['width'],checkpoint['layers'],relations=checkpoint['relations'])
        bridge.load_state_dict(checkpoint['weights'],strict=True)
        return bridge


@dataclass
class GraphPreparedState:
    states: list[torch.Tensor]
    hits: list[dict]
    scores: torch.Tensor | None
    probabilities: torch.Tensor | None
    query_ids: torch.Tensor


class GraphRecurrentEngine:
    """Optional first-class graph memory endpoint for a recurrent LM.

    Retrieve candidate source nodes, perform learned graph message passing,
    and *write directly to the model's live recurrent states*. Does not embed
    evidence in the language prompt; old LM forward/generate remain unchanged.
    """

    def __init__(self, model, tokenizer, archive: ContextArchive,
                 bridge: GraphStateBridge, *, ranker=None):
        if model.config.backbone != 'recurrent' or bridge.width != model.config.d_model or bridge.layers != model.config.n_layers:
            raise ValueError('Graph state bridge requires a matching recurrent backbone')
        if tokenizer.vocab_size != model.config.vocab_size or not isinstance(archive,ContextArchive):
            raise ValueError('Tokenizer or archive mismatch')
        self.model, self.tokenizer, self.archive = model, tokenizer, archive
        self.bridge, self.ranker = bridge, ranker

    @staticmethod
    def _relation_id(relation: str, modulo: int) -> int:
        return 1 + int.from_bytes(hashlib.sha256(relation.encode('utf8')).digest()[:8],'big') % (modulo-1)

    def _candidate_tensors(self, hits, *, max_source_tokens: int):
        if not hits:
            raise ValueError('At least one source required')
        if max_source_tokens < 1:
            raise ValueError('max_source_tokens must be positive')
        device=self.model.token_embedding.weight.device
        sequences=[self.tokenizer.encode(h['text'])[:max_source_tokens] for h in hits]
        if any(not s for s in sequences):
            raise ValueError('Empty tokenized evidence')
        sizes=torch.tensor([[len(s) for s in sequences]],device=device,dtype=torch.long)
        tensor=torch.zeros(1,len(hits),max(map(len,sequences)),device=device,dtype=torch.long)
        for index,seq in enumerate(sequences):
            tensor[0,index,:len(seq)]=torch.tensor(seq,device=device)
        edges=torch.zeros(1,len(hits),len(hits),device=device,dtype=torch.long)
        lookup={h['id']:i for i,h in enumerate(hits)}
        for edge in self.archive.edges:
            if edge['left'] in lookup and edge['right'] in lookup:
                a,b=lookup[edge['left']],lookup[edge['right']]
                edge_id=self._relation_id(edge['relation'],self.bridge.relations)
                edges[0,a,b]=edge_id
                edges[0,b,a]=edge_id
        return tensor,sizes,edges

    def prepare(self, question: str, *, top_k: int=4, max_hops: int=2,
                candidate_pool: int=12, max_source_tokens: int=96,
                explicit_hits: list[dict] | None=None,
                retrieval_policy: str='balanced'):
        """Prefill question, retrieve, then directly amend recurrent states.

        During training `explicit_hits` supports labeled hard negatives, while
        production uses lexical+graph candidate discovery (not a truth oracle).
        """
        if not isinstance(question,str) or not question.strip():
            raise ValueError('Question must be a nonempty string')
        if not (1 <= top_k <= candidate_pool <= 100):
            raise ValueError('Invalid candidate budget')
        if retrieval_policy not in ('legacy', 'balanced'):
            raise ValueError('Retrieval policy must be legacy or balanced')
        qids=self.tokenizer.encode('Question: '+question+'\nAnswer:')
        if not qids or len(qids)>self.model.config.block_size:
            raise ValueError('Question exceeds model context budget')
        query_ids=torch.tensor([qids],dtype=torch.long,device=self.model.token_embedding.weight.device)
        # Normal prefill computes query features and state in *one* pass.
        x=self.model.token_embedding(query_ids)
        states=[]
        for block in self.model.blocks:
            x, state=block(x,self.model.memory,return_state=True)
            states.append(state)
        query_rep=self.model.final_norm(x[:,-1])
        if explicit_hits is not None:
            hits=explicit_hits[:top_k]
        else:
            hits=self.archive.retrieve(question,top_k=candidate_pool,max_hops=max_hops,
                                       max_total=candidate_pool,policy=retrieval_policy)
            if self.ranker is not None and hits:
                hits=self.ranker.rerank(question,hits,self.tokenizer,top_k=top_k)
            else:
                hits=hits[:top_k]
        if not hits:
            return GraphPreparedState(states,[],None,None,query_ids)
        ids,lengths,edges=self._candidate_tensors(hits,max_source_tokens=max_source_tokens)
        doc_vectors=self.model.token_embedding(ids)
        lexical = torch.tensor([EvidenceRanker.lexical_overlap(
            question,[h['text'] for h in hits])],
            device=query_rep.device,dtype=query_rep.dtype)
        deltas,scores,probs=self.bridge(query_rep,doc_vectors,lengths,edges,
                                        lexical_features=lexical,return_scores=True)
        states=[h + deltas[:,i,:] for i,h in enumerate(states)]
        return GraphPreparedState(states,hits,scores,probs,query_ids)

    def answer_loss(self, question: str, answer: str, *, explicit_hits=None,
                    gold_index: int | None=None, retrieval_weight: float=0.2, **kwargs):
        """Teacher-forced answer training, without placing evidence in tokens.

        The next-token objective teaches the recurrent decoder to use graph
        state directly. An optional gold passage index supervises relevance.
        """
        if not answer:
            raise ValueError('Nonempty answer required')
        result=self.prepare(question,explicit_hits=explicit_hits,**kwargs)
        prefix=self.tokenizer.encode(' ')
        answer_ids=self.tokenizer.encode(answer,add_eos=True)
        if len(prefix)!=1 or not answer_ids:
            raise ValueError('A one-token continuation delimiter is required')
        sequence=prefix + answer_ids[:-1]
        x=torch.tensor([sequence],dtype=torch.long,device=result.query_ids.device)
        y=torch.tensor([answer_ids],dtype=torch.long,device=result.query_ids.device)
        out,_=self.model.recurrent_continue(x,result.states,targets=y)
        loss=out.loss
        if gold_index is not None:
            if result.scores is None or not 0 <= gold_index < len(result.hits):
                raise ValueError('Gold evidence index is invalid')
            loss=loss + retrieval_weight*F.cross_entropy(
                result.scores,torch.tensor([gold_index],device=y.device))
        return loss, result

    @torch.no_grad()
    def generate(self, question: str, *, max_new_tokens: int=40,
                 temperature: float=0.0, eos_id: int | None=None, **kwargs):
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError('Invalid generation budget/temperature')
        prev_model,prev_bridge=self.model.training,self.bridge.training
        self.model.eval();self.bridge.eval()
        try:
            prepared=self.prepare(question,**kwargs)
            if max_new_tokens == 0:
                return '',prepared.hits
            prefix=self.tokenizer.encode(' ')
            if len(prefix)!=1:
                raise ValueError('A one-token continuation delimiter is required')
            token=torch.tensor([[prefix[0]]],device=prepared.query_ids.device,dtype=torch.long)
            logits,states=self.model._recurrent_step(token,prepared.states)
            result=[]
            for _ in range(max_new_tokens):
                if temperature==0:
                    next_token=logits.argmax(dim=-1,keepdim=True)
                else:
                    next_token=torch.multinomial(F.softmax(logits.float()/temperature,dim=-1),1)
                idx=int(next_token.item())
                if idx==eos_id:
                    break
                result.append(idx)
                logits,states=self.model._recurrent_step(next_token,states)
            return self.tokenizer.decode(result),prepared.hits
        finally:
            self.model.train(prev_model);self.bridge.train(prev_bridge)
