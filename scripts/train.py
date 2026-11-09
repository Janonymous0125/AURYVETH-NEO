"""Auditable GPU/CPU autoregressive + neural-memory training.

Production runs require real train/validation corpora, an immutable tokenizer,
and sufficient token counts. Development overrides are explicit and recorded.
The persistent neural-memory codec remains separate from ordinary LM training.
"""
from __future__ import annotations

import argparse
import errno
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import warnings
from dataclasses import asdict

import numpy as np
import torch

from model import IntegratedConfig, IntegratedLM
from model.artifacts import write_manifest
from neural_tokenizer import LayeredTokenizer, SentencePieceTokenizer


def load_rows(path, tokenizer, config):
    """Validated supervised memory examples; all lengths are bounded."""
    rows=[]
    # A JSONL record ends at an actual LF/CRLF, not at arbitrary Unicode
    # separators inside JSON strings. `str.splitlines()` also splits on
    # U+2028/U+2029/U+0085, which are valid JSON string contents when
    # json.dumps(..., ensure_ascii=False) is used by the dataset builder.
    # Stream physical lines to avoid an extra full-file string allocation.
    with Path(path).open('r',encoding='utf8',newline='') as stream:
        for number,line in enumerate(stream,1):
            if not line.strip(): continue
            try:
                entry=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Malformed JSONL in {path}, physical line {number}, '
                    f'column {exc.colno}: {exc.msg}. '
                    'The file was not modified; check for truncation or malformed records.'
                ) from exc
            if not isinstance(entry,dict) or set(entry)!={'context','prompt','answer'} or not isinstance(entry['prompt'],str) or not isinstance(entry['answer'],str):
                raise ValueError(f'Invalid JSONL row {number} in {path}')
            contexts=entry['context']
            if isinstance(contexts,str): contexts=[contexts]
            if not isinstance(contexts,list) or len(contexts)>config.active_memories or any(not isinstance(c,str) or not c for c in contexts):
                raise ValueError(f'Invalid memory context at {number} in {path}')
            support=[tokenizer.encode(c) for c in contexts]
            if any(not 1<=len(s)<=config.memory_tokens for s in support):
                raise ValueError(f'Invalid memory length at {number} in {path}')
            prompt=tokenizer.encode(entry['prompt'])
            answer=tokenizer.encode(entry['answer'],add_eos=True)
            if not prompt or not answer or len(prompt)+len(answer)-1>config.block_size:
                raise ValueError(f'Prompt/answer exceeds model context at {number} in {path}')
            sequence=prompt+answer
            rows.append((sequence[:-1], [-100]*(len(prompt)-1)+sequence[len(prompt):], support))
    if not rows: raise ValueError(f'No memory training examples in {path}')
    return rows


def from_corpus(path, tokenizer, config):
    """Legacy small-data interface retained for existing tests only."""
    ids=tokenizer.encode(Path(path).read_text(encoding='utf8'),add_eos=True)
    result=[]
    for start in range(0,len(ids)-1,config.block_size):
        chunk=ids[start:start+config.block_size+1]
        if len(chunk)>=2: result.append((chunk[:-1],chunk[1:],[]))
    if not result: raise ValueError('Corpus too short')
    return result


def one_loss(model, row, device, *, efficient=False, loss_chunk_size=128):
    """A single differentiable example; no external memory lookup/answer injection."""
    inputs, targets, contexts=row
    inp=torch.tensor([inputs],dtype=torch.long,device=device)
    gold=torch.tensor([targets],dtype=torch.long,device=device)
    options={'return_logits':False,'loss_chunk_size':loss_chunk_size} if efficient else {}
    if not contexts:
        return model(inp,targets=gold,**options).loss
    longest=max(len(c) for c in contexts)
    support=torch.zeros((1,len(contexts),longest),dtype=torch.long,device=device)
    lengths=torch.tensor([[len(c) for c in contexts]],dtype=torch.long,device=device)
    for index,context in enumerate(contexts):
        support[0,index,:len(context)]=torch.tensor(context,device=device)
    return model(inp,targets=gold,support_ids=support,support_lengths=lengths,**options).loss


def batch_loss(model, rows, device, *, loss_chunk_size=128):
    """True batched forward; padding contributes neither loss nor future leakage.

    The caller groups rows by memory-context count. Padding support lengths
    remain positive and are masked before graph and neural-memory reading.
    """
    if not rows: raise ValueError('Empty batch')
    counts={len(r[2]) for r in rows}
    if len(counts)!=1:
        raise ValueError('Batch rows require identical numbers of memory contexts')
    batch=len(rows);longest=max(len(r[0]) for r in rows)
    inp=torch.zeros((batch,longest),dtype=torch.long,device=device)
    gold=torch.full((batch,longest),-100,dtype=torch.long,device=device)
    for index,(inputs,targets,_) in enumerate(rows):
        inp[index,:len(inputs)]=torch.tensor(inputs,dtype=torch.long,device=device)
        gold[index,:len(targets)]=torch.tensor(targets,dtype=torch.long,device=device)
    if not rows[0][2]:
        return model(inp,targets=gold,return_logits=False,
                     loss_chunk_size=loss_chunk_size).loss
    max_mem=max(len(context) for _,_,contexts in rows for context in contexts)
    memory_count=len(rows[0][2])
    support=torch.zeros((batch,memory_count,max_mem),dtype=torch.long,device=device)
    lengths=torch.ones((batch,memory_count),dtype=torch.long,device=device)
    for b,(_,_,contexts) in enumerate(rows):
        for m,context in enumerate(contexts):
            support[b,m,:len(context)]=torch.tensor(context,dtype=torch.long,device=device)
            lengths[b,m]=len(context)
    return model(inp,targets=gold,support_ids=support,support_lengths=lengths,
                 return_logits=False,loss_chunk_size=loss_chunk_size).loss


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):
            digest.update(buf)
    return digest.hexdigest()


class TokenStream:
    """Memory-mapped uint32 token stream, built without loading a corpus into RAM."""
    def __init__(self, corpus: str | Path, tokenizer, cache_dir: str | Path):
        self.corpus=Path(corpus)
        if not self.corpus.is_file() or not self.corpus.stat().st_size:
            raise ValueError(f'Empty/missing training corpus: {corpus}')
        digest=file_sha256(self.corpus)
        contract=tokenizer.model_contract()
        cache_key=hashlib.sha256(json.dumps(['doc_eos_v2',str(self.corpus.resolve()),digest,contract],sort_keys=True).encode()).hexdigest()[:24]
        cache=Path(cache_dir)
        cache.mkdir(parents=True,exist_ok=True)
        path=cache/f'{cache_key}.u32'
        if not path.is_file():
            fd,temp=tempfile.mkstemp(dir=cache,prefix='.building-',suffix='.u32')
            os.close(fd)
            count=0
            try:
                with self.corpus.open('r',encoding='utf-8') as source, open(temp,'wb') as dest:
                    in_document=False
                    for line in source:
                        # Blank lines separate documents; normal newlines remain
                        # part of a document. The tokenizer still sees source
                        # line breaks for prose and code formatting.
                        if not line.strip():
                            if in_document:
                                np.asarray([tokenizer.EOS_ID],dtype='<u4').tofile(dest)
                                count+=1
                                in_document=False
                            continue
                        # Chunk long lines to limit the tokenizer input size.
                        for begin in range(0,len(line),8192):
                            ids=tokenizer.encode(line[begin:begin+8192])
                            if ids:
                                np.asarray(ids,dtype='<u4').tofile(dest)
                                count+=len(ids)
                                in_document=True
                    if in_document:
                        np.asarray([tokenizer.EOS_ID],dtype='<u4').tofile(dest)
                        count+=1
                    dest.flush();os.fsync(dest.fileno())
                if count<2: raise ValueError('Corpus has fewer than 2 tokens')
                os.replace(temp,path)
            finally:
                if os.path.exists(temp): os.unlink(temp)
        if path.stat().st_size%4: raise ValueError('Token cache corrupt')
        self.tokens=np.memmap(path,dtype='<u4',mode='r')
        self.eos_id=int(tokenizer.EOS_ID)
        self.count=len(self.tokens)
        if self.count<2: raise ValueError('Corpus has fewer than 2 tokens')
        self.source_sha256=digest
        self.path=path

    def row(self, start:int, length:int):
        if not 0<=start<self.count-1 or length<1: raise ValueError('Invalid sample offset')
        # Never allow a target to cross EOS into the next independent document.
        # Keep the EOS label itself as the final training target.
        if int(self.tokens[start])==self.eos_id:
            start+=1
            if start>=self.count-1:
                raise ValueError('No tokens after document boundary')
        selected=np.asarray(self.tokens[start:min(start+length+1,self.count)],dtype=np.int64)
        ends=np.flatnonzero(selected[1:]==self.eos_id)
        if len(ends):
            selected=selected[:int(ends[0])+2]
        return (selected[:-1].tolist(),selected[1:].tolist(),[])

    def random_row(self, rng:random.Random, length:int):
        for _ in range(128):
            start=rng.randrange(max(1,self.count-1))
            if int(self.tokens[start])!=self.eos_id:
                return self.row(start,length)
        raise ValueError('Corpus contains too few non-EOS tokens')

    def validation_rows(self,length:int,maximum:int):
        starts=list(range(0,self.count-1,max(1,length)))
        if maximum and len(starts)>maximum:
            starts=[starts[i*len(starts)//maximum] for i in range(maximum)]
        return [self.row(i,length) for i in starts if i<self.count-2 or int(self.tokens[i])!=self.eos_id]


def resolve_device(value):
    if value=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev=torch.device(value)
    if dev.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return dev


def resolve_precision(value,device):
    if value=='auto':
        value='bf16' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if value=='fp16' and device.type!='cuda': raise ValueError('FP16 requires CUDA')
    if value=='bf16' and device.type=='cpu': raise ValueError('Use fp32 on CPU for this trainer')
    return value


def autocast_context(precision,device):
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16) if precision!='fp32' else nullcontext()


def _serialized_storage_bytes(value):
    """Lower-bound estimate for tensors in a torch.save payload, deduplicating storages."""
    seen=set()
    total=0
    pending=[value]
    while pending:
        item=pending.pop()
        if isinstance(item,torch.Tensor):
            storage=item.untyped_storage()
            identity=(storage.data_ptr(),storage.nbytes())
            if identity not in seen:
                seen.add(identity)
                total+=storage.nbytes()
        elif isinstance(item,dict):
            pending.extend(item.values())
        elif isinstance(item,(list,tuple)):
            pending.extend(item)
    return total


def _bytes_label(size):
    return f'{size / (1024**3):.2f} GiB'


def _cleanup_temporary_checkpoint(path):
    """Best effort only: a Windows lock must never replace the real save error."""
    for delay in (0,0.1,0.3):
        if delay: time.sleep(delay)
        try:
            os.unlink(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            cleanup_error=exc
    warnings.warn(f'Incomplete temporary checkpoint {path} could not be removed: '
                  f'{cleanup_error}. Remove it later after checking that it is not in use.',
                  RuntimeWarning,stacklevel=2)


def atomic_torch_save(payload,path):
    """Save through a same-directory temporary file, preserving the last good state."""
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    estimated=_serialized_storage_bytes(payload)
    # The old checkpoint stays in place until the new save succeeds. Reserve
    # space for a full second copy plus headroom for archive metadata.
    reserve=max(128*1024**2,estimated//20)
    try:
        free=shutil.disk_usage(target.parent).free
    except OSError:
        free=None
    if free is not None and free<estimated+reserve:
        raise OSError(errno.ENOSPC,
            f'Cannot safely save {target}: estimated tensor storage '
            f'{_bytes_label(estimated)} plus {_bytes_label(reserve)} reserve; '
            f'only {_bytes_label(free)} free. The previous checkpoint has not been touched. '
            'Free space or resume into an output directory on a larger drive.')
    fd,temp=tempfile.mkstemp(prefix='.training-',suffix='.pt',dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload,temp)
        os.replace(temp,target)
    except Exception as exc:
        # torch.save may leave a partially written file temporarily locked
        # on Windows. Retain the first, causal exception rather than masking
        # it with PermissionError during cleanup.
        try:
            free_after=shutil.disk_usage(target.parent).free
            free_note=f' Free space now: {_bytes_label(free_after)}.'
        except OSError:
            free_note=''
        raise RuntimeError(
            f'Checkpoint save FAILED for {target}: {type(exc).__name__}: {exc}.'
            f'{free_note} Previous successful checkpoint was not replaced. '
            'Check disk space, antivirus/indexing locks, and directory permissions; '
            'do not remove model.pt or training_state.pt before recovery.'
        ) from exc
    finally:
        if os.path.exists(temp):
            _cleanup_temporary_checkpoint(temp)


def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus');p.add_argument('--validation-corpus')
    p.add_argument('--train-jsonl');p.add_argument('--validation-jsonl')
    p.add_argument('--tokenizer',help='Trained SentencePiece .model or legacy LayeredTokenizer .json')
    p.add_argument('--allow-byte-tokenizer',action='store_true',help='Explicit 258-byte vocabulary development mode')
    p.add_argument('--allow-small-vocab',action='store_true',help='Development only: permit a trained tokenizer with fewer than 50k pieces')
    p.add_argument('--allow-cpu-large',action='store_true',help='Explicitly accept slow high-memory CPU training')
    p.add_argument('--allow-small-dataset',action='store_true',help='Development/test only; not a qualified training run')
    p.add_argument('--allow-unvalidated',action='store_true',help='Override mandatory held-out validation')
    p.add_argument('--resume-checkpoint',help='Initialize model from a previous inference checkpoint (optimizer resets)')
    p.add_argument('--resume-state',help='Restore model+optimizer+scaler+RNG for continued training')
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=10000,help='Total planned optimizer updates, including completed steps on resume')
    p.add_argument('--stop-after',type=int,default=None,help='Optional early stop for resumable planned training (same LR schedule)')
    p.add_argument('--allow-plan-change',action='store_true',help='Explicitly permit changing LR/sampling schedule on resume')
    p.add_argument('--width',type=int,default=1024)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--heads',type=int,default=16)
    p.add_argument('--kv-heads',type=int,default=4)
    p.add_argument('--block-size',type=int,default=8192,help='Maximum context capacity, not training sequence length')
    p.add_argument('--train-seq-len',type=int,default=None,help='Training-token window (default min(512, model context))')
    p.add_argument('--memory-tokens',type=int,default=64)
    p.add_argument('--planner-steps',type=int,default=0,help='Fresh model: 0=legacy-compatible; 1..8 enables trainable causal latent refinement per block')
    p.add_argument('--backbone',choices=['transformer','recurrent'],default='transformer',
                   help='v0.6 attention-free recurrent backbone, or checkpoint-compatible transformer')
    p.add_argument('--upgrade-planner-steps',type=int,default=0,help='Explicitly add zero-residual planner to an old --resume-checkpoint (optimizer resets)')
    p.add_argument('--batch-size',type=int,default=1,help='Sequences per gradient accumulation microstep')
    p.add_argument('--grad-accum',type=int,default=8,help='Microsteps per optimizer update')
    p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--min-lr',type=float,default=None)
    p.add_argument('--warmup-steps',type=int,default=100)
    p.add_argument('--weight-decay',type=float,default=0.1)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--precision',choices=['auto','fp32','fp16','bf16'],default='auto')
    p.add_argument('--device',default='auto')
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--loss-chunk-size',type=int,default=128)
    p.add_argument('--memory-sampling-ratio',type=float,default=0.25)
    p.add_argument('--seed',type=int,default=15)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--validation-max-rows',type=int,default=64)
    return p


def _lr_at_step(step,args):
    if step<=args.warmup_steps and args.warmup_steps:
        return args.lr*step/args.warmup_steps
    remaining=max(1,args.steps-args.warmup_steps)
    fraction=min(1,max(0,(step-args.warmup_steps)/remaining))
    return args.min_lr+0.5*(args.lr-args.min_lr)*(1+math.cos(math.pi*fraction))


def main():
    parser=make_parser();args=parser.parse_args()
    if not (args.corpus or args.train_jsonl): parser.error('Supply --corpus and/or --train-jsonl')
    if not args.allow_unvalidated and not (args.validation_corpus or args.validation_jsonl):
        parser.error('Independent validation data is mandatory (or explicit --allow-unvalidated)')
    if args.resume_state and args.resume_checkpoint: parser.error('Choose one resume mode')
    if args.upgrade_planner_steps and (not args.resume_checkpoint or args.planner_steps):
        parser.error('--upgrade-planner-steps requires only --resume-checkpoint, not --planner-steps or --resume-state')
    if args.min_lr is None: args.min_lr = args.lr * 0.1
    if args.steps<1 or (args.stop_after is not None and not 1<=args.stop_after<=args.steps) or (args.train_seq_len is not None and args.train_seq_len<2) or args.batch_size<1 or args.grad_accum<1 or args.loss_chunk_size<1:
        parser.error('Invalid steps/sequence/batch configuration')
    if not 0 <= args.planner_steps <= 8 or not 0 <= args.upgrade_planner_steps <= 8:
        parser.error('planner-steps and upgrade-planner-steps must be 0..8')
    if args.width<2 or args.block_size<2 or (args.train_seq_len is not None and args.train_seq_len>args.block_size) or args.eval_every<1 or args.save_every<1 or args.log_every<1:
        parser.error('Invalid model/evaluation configuration')
    if not (0<args.lr<1 and 0<=args.min_lr<=args.lr and args.warmup_steps>=0 and 0<=args.memory_sampling_ratio<=1 and args.grad_clip>0):
        parser.error('Invalid optimization configuration')
    device=resolve_device(args.device)
    precision=resolve_precision(args.precision,device)
    torch.manual_seed(args.seed);random.seed(args.seed)
    if device.type=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    rng=random.Random(args.seed)
    if args.tokenizer:
        tokenizer=(LayeredTokenizer.load(args.tokenizer) if str(args.tokenizer).endswith('.json')
                   else SentencePieceTokenizer(args.tokenizer))
    elif args.allow_byte_tokenizer or args.resume_checkpoint or args.resume_state:
        tokenizer=LayeredTokenizer()
    else:
        parser.error('Train and provide a real vocabulary with --tokenizer; 258-byte mode requires --allow-byte-tokenizer')
    if device.type=='cpu' and args.width>=1024 and tokenizer.vocab_size>=50000 and not args.allow_cpu_large and not (args.resume_state or args.resume_checkpoint):
        parser.error('Large 50k/1024 model requires CUDA for practical training; use --allow-cpu-large to explicitly override')
    if not (args.allow_byte_tokenizer or args.allow_small_vocab or args.resume_state or args.resume_checkpoint) and tokenizer.vocab_size!=50000:
        parser.error(f'Expected 50,000 trained tokens; tokenizer has {tokenizer.vocab_size}')
    if args.resume_checkpoint:
        model=IntegratedLM.load(args.resume_checkpoint,tokenizer)
        if args.upgrade_planner_steps:
            model.add_planner_to_existing_model(args.upgrade_planner_steps)
    elif args.resume_state:
        state=torch.load(args.resume_state,map_location='cpu',weights_only=True)
        if state.get('schema')!=1 or state.get('tokenizer_contract')!=tokenizer.model_contract():
            raise ValueError('Training-state schema/tokenizer mismatch')
        model=IntegratedLM(IntegratedConfig(**state['config']))
        model.load_state_dict(state['model'])
        model.memory.restore_metadata(state['memory_metadata'])
        model._tokenizer_contract=tokenizer.model_contract()
        if model.memory.count and model.memory.encoder_fingerprint!=model._encoder_fingerprint():
            raise ValueError('Saved neural memories incompatible with model')
        if model.memory.count:model._freeze_memory_codec()
    else:
        cfg=IntegratedConfig(vocab_size=tokenizer.vocab_size,block_size=args.block_size,
            d_model=args.width,n_layers=args.layers,n_heads=args.heads,n_kv_heads=args.kv_heads,
            ffn_hidden=args.width*3,tie_embeddings=True,memory_tokens=args.memory_tokens,
            memory_slots=128,regions=4,active_regions=2,active_memories=4,
            planner_steps=args.planner_steps,backbone=args.backbone)
        model=IntegratedLM(cfg)
    if (args.resume_checkpoint or args.resume_state) and args.planner_steps and model.config.planner_steps != args.planner_steps:
        parser.error('Cannot change planner architecture while resuming a checkpoint; start a fresh model or keep the checkpoint planner setting')
    if args.train_seq_len is None:
        args.train_seq_len=min(512,model.config.block_size)
    if args.train_seq_len>model.config.block_size:
        parser.error('Training context exceeds loaded model capacity')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tokenizer_artifact = None
    if isinstance(tokenizer, SentencePieceTokenizer):
        tokenizer_artifact = out/'tokenizer.model'
        if not tokenizer_artifact.exists():
            shutil.copyfile(tokenizer.path, tokenizer_artifact)
        if SentencePieceTokenizer(tokenizer_artifact).model_contract() != tokenizer.model_contract():
            raise ValueError('Output directory contains a different tokenizer')
    corpus=TokenStream(args.corpus,tokenizer,out/'cache') if args.corpus else None
    validation_corpus=TokenStream(args.validation_corpus,tokenizer,out/'cache') if args.validation_corpus else None
    if corpus and validation_corpus and corpus.source_sha256==validation_corpus.source_sha256:
        raise ValueError('Train and validation corpora are identical')
    memory_rows=load_rows(args.train_jsonl,tokenizer,model.config) if args.train_jsonl else []
    memory_buckets={}
    for row in memory_rows:
        memory_buckets.setdefault(len(row[2]),[]).append(row)
    memory_val=load_rows(args.validation_jsonl,tokenizer,model.config) if args.validation_jsonl else []
    if any(len(row[0])>args.train_seq_len for row in memory_rows+memory_val):
        raise ValueError('Memory-conditioned example exceeds --train-seq-len; raise training length or shorten examples')
    if (not args.allow_small_dataset and
        ((corpus and corpus.count<100_000) or
         (validation_corpus and validation_corpus.count<10_000) or
         (memory_rows and len(memory_rows)<100) or
         (memory_val and len(memory_val)<20))):
        raise ValueError('Training/validation corpus too small; provide real data or explicit --allow-small-dataset')
    if (not args.allow_unvalidated and not validation_corpus and not memory_val):
        raise ValueError('Validation examples are required')
    # Memory contexts and text-only sequences are sampled independently,
    # so a positive memory ratio cannot silently produce no memory training.
    model.to(device)
    model.gradient_checkpointing=args.gradient_checkpointing
    decay=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
    no_decay=[p for p in model.parameters() if p.requires_grad and p.ndim<2]
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':args.weight_decay},
                                 {'params':no_decay,'weight_decay':0.0}],
                                lr=args.lr,fused=(device.type=='cuda'))
    scaler=torch.amp.GradScaler('cuda',enabled=(precision=='fp16'))
    start_step=0;tokens_seen=0;best_val=float('inf')
    fingerprints={'corpus':corpus.source_sha256 if corpus else None,
                  'validation_corpus':validation_corpus.source_sha256 if validation_corpus else None,
                  'train_jsonl':file_sha256(args.train_jsonl) if args.train_jsonl else None,
                  'validation_jsonl':file_sha256(args.validation_jsonl) if args.validation_jsonl else None}
    training_plan={'steps':args.steps,'lr':args.lr,'min_lr':args.min_lr,
        'warmup_steps':args.warmup_steps,'weight_decay':args.weight_decay,
        'grad_accum':args.grad_accum,'batch_size':args.batch_size,
        'train_seq_len':args.train_seq_len,'memory_sampling_ratio':args.memory_sampling_ratio,
        'seed':args.seed}
    if args.resume_state:
        if state['training_plan']!=training_plan and not args.allow_plan_change:
            raise ValueError('Training plan changed; use --allow-plan-change with explicit acceptance of non-equivalent schedule')
        if state['data_fingerprints']!=fingerprints:
            raise ValueError('Resume data changed; refusing nonreproducible continuation')
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_step=int(state['completed_steps']);tokens_seen=int(state['tokens_seen'])
        best_val=float(state['best_validation_loss'])
        rng.setstate(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    if start_step>=args.steps: raise ValueError('No remaining optimizer steps')

    validation_coverage = {'rows':0, 'supervised_tokens':0}

    def validate():
        model.eval();scores=[];weighted=[]
        rows=[]
        if validation_corpus:
            rows.extend(validation_corpus.validation_rows(min(args.train_seq_len,model.config.block_size),args.validation_max_rows))
        if memory_val:
            rows.extend(memory_val if not args.validation_max_rows else memory_val[:args.validation_max_rows])
        with torch.no_grad():
            for row in rows:
                with autocast_context(precision,device):
                    score=one_loss(model,row,device,efficient=True,loss_chunk_size=args.loss_chunk_size)
                weight=sum(t!=-100 for t in row[1]);scores.append(float(score)*weight);weighted.append(weight)
        model.train()
        validation_coverage.update(rows=len(rows), supervised_tokens=sum(weighted))
        return sum(scores)/sum(weighted) if weighted else None

    def save_state(step):
        # Full resumable state (model + optimizer + RNG) and separate inference model.
        payload={'schema':1,'config':asdict(model.config),'model':model.state_dict(),
                 'memory_metadata':model.memory.snapshot(),
                 'tokenizer_contract':tokenizer.model_contract(),
                 'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                 'completed_steps':step,'tokens_seen':tokens_seen,
                 'best_validation_loss':best_val,'sampler_rng':rng.getstate(),
                 'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                 'data_fingerprints':fingerprints,'training_plan':training_plan}
        atomic_torch_save(payload,out/'training_state.pt')
        model.save(out/'model.pt',tokenizer)
        (out/'train_metrics.json').write_text(json.dumps({
           'completed_steps':step,'planned_steps':args.steps,'tokens_seen':tokens_seen,'validation_loss':latest_val,
           'best_validation_loss':best_val,'training_examples':len(memory_rows),
           'train_tokens':corpus.count if corpus else 0,
           'validation_tokens':validation_corpus.count if validation_corpus else 0,
           'validation_rows_available':(math.ceil((validation_corpus.count-1)/args.train_seq_len) if validation_corpus else 0)+len(memory_val),
           'train_examples':len(memory_rows)+(math.ceil((corpus.count-1)/args.train_seq_len) if corpus else 0),
           'device':str(device),'precision':precision,'vocab_size':model.config.vocab_size,
           'd_model':model.config.d_model,'context_capacity':model.config.block_size,
           'planner_steps':model.config.planner_steps,
           'backbone':model.config.backbone,
           'actual_train_sequence_length':args.train_seq_len,
           'qualified':not(args.allow_small_dataset or args.allow_byte_tokenizer or args.allow_small_vocab or args.allow_unvalidated),
           'qualification_scope':'training_protocol_only',
           'competence_qualified':False,
           'validation_evaluated':validation_coverage,
           'last_train_loss':latest_loss,
           'validation_perplexity':(math.exp(latest_val) if latest_val is not None and latest_val<80 else None),
           'data_fingerprints':fingerprints,
           'training_plan':training_plan,
           'note':'These metrics are not evidence of general semantic competence.'},indent=2),encoding='utf8')
        artifacts = {'model':'model.pt', 'training_state':'training_state.pt', 'metrics':'train_metrics.json'}
        if tokenizer_artifact is not None:
            artifacts['tokenizer'] = tokenizer_artifact.name
        if (out/'best_model.pt').is_file():
            artifacts['best_model'] = 'best_model.pt'
        write_manifest(out/'manifest.json', artifacts,
                       metadata={'tokenizer_contract':tokenizer.model_contract(),
                                 'config':asdict(model.config), 'completed_steps':step,
                                 'best_validation_loss':best_val,
                                 'competence_qualified':False})

    start=time.monotonic();latest_val=None;model.train()
    final_step=min(args.steps,args.stop_after or args.steps)
    if start_step>=final_step: raise ValueError('No optimizer steps before requested stop')
    for step in range(start_step+1,final_step+1):
        optimizer.zero_grad(set_to_none=True)
        step_loss=0.;step_tokens=0
        for _ in range(args.grad_accum):
            use_memory=bool(memory_rows) and (not corpus or rng.random()<args.memory_sampling_ratio)
            if use_memory:
                first=memory_rows[rng.randrange(len(memory_rows))]
                compatible=memory_buckets[len(first[2])]
                rows=[first]+[compatible[rng.randrange(len(compatible))]
                              for _ in range(args.batch_size-1)]
            else:
                rows=[corpus.random_row(rng,args.train_seq_len)
                      for _ in range(args.batch_size)]
            step_tokens+=sum(sum(t!=-100 for t in row[1]) for row in rows)
            with autocast_context(precision,device):
                loss=batch_loss(model,rows,device,loss_chunk_size=args.loss_chunk_size)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at step {step}')
            step_loss+=float(loss.detach())
            scaler.scale(loss/args.grad_accum).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
        lr=_lr_at_step(step,args)
        for group in optimizer.param_groups:group['lr']=lr
        previous_scale=scaler.get_scale()
        scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<previous_scale:
            raise FloatingPointError('FP16 optimizer update skipped due to gradient overflow; step not counted')
        tokens_seen+=step_tokens
        # batch_loss is already averaged across tokens in the microbatch.
        latest_loss=step_loss/args.grad_accum
        if step%args.eval_every==0 or step==final_step:
            latest_val=validate()
            if latest_val is not None and latest_val < best_val:
                best_val=latest_val
                model.save(out/'best_model.pt',tokenizer)
        if step%args.log_every==0 or step in (start_step+1,final_step):
            duration=max(time.monotonic()-start,1e-9)
            event={'step':step,'loss':round(latest_loss,5),
                   'validation_loss':latest_val,'lr':lr,
                   'gradient_norm':float(norm),'tokens_seen':tokens_seen,
                   'tokens_per_second':round((tokens_seen-int(state['tokens_seen']) if args.resume_state else tokens_seen)/duration,1),
                   'device':str(device),'precision':precision}
            print(json.dumps(event),flush=True)
            with (out/'events.jsonl').open('a',encoding='utf8') as event_file:
                event_file.write(json.dumps(event)+'\n')
        if step%args.save_every==0 or step==final_step:
            save_state(step)
    print(json.dumps({'completed_steps':final_step,'validation_loss':latest_val,
                      'tokens_seen':tokens_seen,'checkpoint':str(out/'model.pt')},indent=2),flush=True)

if __name__=='__main__':main()
