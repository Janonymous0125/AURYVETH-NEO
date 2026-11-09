"""Optional explicit-entity routing over existing neural memory bodies.

This is a metadata index, not a learned semantic retriever or answer lookup.
Callers supply the entity identity on both writes and reads. Untagged legacy
memories remain available through ordinary, unscoped model APIs.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import unicodedata
import copy
from .memory import NeuralMemoryBank
from .integrated import IntegratedLM
from .artifacts import write_manifest,verify_manifest


class EntityMemoryRouter:
    def __init__(self,model,tokenizer,entities=None):
        if tokenizer.model_contract()!=model._tokenizer_contract and model._tokenizer_contract is not None:
            raise ValueError('Router tokenizer/model mismatch')
        if tokenizer.vocab_size!=model.config.vocab_size:raise ValueError('Router vocabulary mismatch')
        self.model,self.tokenizer=model,tokenizer
        self._bank=model.memory
        self._scoping=False
        self.entities={}
        for identity,entity in (entities or {}).items():self.tag(identity,entity)

    @staticmethod
    def _entity(entity):
        if not isinstance(entity,str) or not entity.strip() or len(entity)>256:
            raise ValueError('Explicit bounded entity identity required')
        return unicodedata.normalize('NFC',entity.strip())

    def tag(self,memory_id,entity):
        if self._scoping:raise RuntimeError('Cannot tag a temporary memory view')
        self._check_bank()
        if type(memory_id) is not int or not 1<=memory_id<=self.model.memory.count:
            raise ValueError('Entity tag requires an existing memory ID')
        entity=self._entity(entity)
        if memory_id in self.entities and self.entities[memory_id]!=entity:
            raise ValueError('An existing entity tag cannot be silently reassigned')
        self.entities[memory_id]=entity

    def remember(self,text,*,entity,source,region=0):
        if self._scoping:raise RuntimeError('Cannot write into a temporary memory view')
        self._check_bank()
        entity=self._entity(entity)
        identity=self.model.remember(text,self.tokenizer,source=source,region=region)
        self.entities[identity]=entity
        return identity

    def selected_ids(self,entity):
        entity=self._entity(entity)
        self._check_bank()
        if any(identity>self.model.memory.count for identity in self.entities):
            raise RuntimeError('Entity index no longer matches the memory bank')
        return tuple(i for i,e in sorted(self.entities.items()) if e==entity)

    def _check_bank(self):
        if self.model.memory is not self._bank:raise RuntimeError('Memory bank was replaced; rebuild the entity index')

    @contextmanager
    def scope(self,entity,*,require_unique=False):
        """Synchronous per-call view; original bank and IDs are restored exactly."""
        if self._scoping:raise RuntimeError('Nested/concurrent scopes require separate router instances')
        identities=self.selected_ids(entity);original=self.model.memory
        if require_unique and original._suspended:
            raise LookupError('Memory is disabled')
        if require_unique and len(identities)!=1:
            if not identities:
                raise LookupError('No memory record matches this entity')
            raise LookupError('Multiple memory records match this entity')
        self._scoping=True
        try:
            if not identities:
                with original.suspended():yield identities
                return
            selected=NeuralMemoryBank(self.model.config).to(device=original.states.device,dtype=original.states.dtype)
            remap={old:i+1 for i,old in enumerate(identities)}
            for old in identities:
                index=old-1;length=int(original.lengths[index]);record=original.provenance[index]
                new=selected.write(original.states[index,:length],source=record['source'],
                    region=int(original.region_ids[index]),encoder_fingerprint=original.encoder_fingerprint)
                selected.status_codes[new-1]=original.status_codes[index]
                selected.versions[new-1]=original.versions[index]
                selected.provenance[new-1]={**record,'id':new}
            for edge in original.edges:
                if edge['source_id'] in remap and edge['target_id'] in remap:
                    selected.link(remap[edge['source_id']],remap[edge['target_id']],relation=edge['relation'])
            selected.current_version=original.current_version
            selected.migration_history=copy.deepcopy(original.migration_history)
            selected._suspended=original._suspended
            self.model.memory=selected
            yield identities
        finally:
            self.model.memory=original
            self._scoping=False

    def generate(self,input_ids,*,entity,**kwargs):
        # The LM still decodes the selected neural states; the index stores no answers.
        with self.scope(entity):return self.model.generate(input_ids,**kwargs)

    def generate_unique(self,input_ids,*,entity,**kwargs):
        """Require exactly one tagged record or raise LookupError before decoding."""
        with self.scope(entity,require_unique=True):
            return self.model.generate(input_ids,**kwargs)

    def save(self,directory):
        if self._scoping:raise RuntimeError('Cannot save a temporary memory view')
        self._check_bank()
        directory=Path(directory)
        if directory.exists():raise FileExistsError('Use a new entity-memory bundle directory')
        directory.mkdir(parents=True)
        self.model.save(directory/'model.pt',self.tokenizer)
        payload={'schema':1,'tokenizer_contract':self.tokenizer.model_contract(),
                 'entities':{str(k):v for k,v in sorted(self.entities.items())}}
        (directory/'entities.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf8')
        write_manifest(directory/'manifest.json',{'model':'model.pt','entities':'entities.json'},
                       metadata={'entity_routing':'explicit caller-supplied identities; case-sensitive',
                                 'tokenizer_contract':self.tokenizer.model_contract()})

    @classmethod
    def load(cls,directory,tokenizer):
        files,_=verify_manifest(Path(directory)/'manifest.json')
        if set(files)!= {'model','entities'}:raise ValueError('Invalid entity-memory bundle')
        payload=json.loads(files['entities'].read_text(encoding='utf8'))
        if (payload.get('schema')!=1 or payload.get('tokenizer_contract')!=tokenizer.model_contract()
                or not isinstance(payload.get('entities'),dict)):
            raise ValueError('Invalid entity-memory index')
        mapping={}
        for key,value in payload['entities'].items():
            if not key.isdecimal() or str(int(key))!=key:raise ValueError('Invalid entity memory ID')
            mapping[int(key)]=value
        return cls(IntegratedLM.load(files['model'],tokenizer),tokenizer,mapping)
