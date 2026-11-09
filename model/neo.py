"""AURYVETH-NEO stores shared chat weights once and exact planner overrides."""
from pathlib import Path
import shutil

import torch

from .artifacts import file_sha256, verify_manifest, write_manifest
from .integrated import IntegratedLM
from neural_tokenizer import SentencePieceTokenizer


NAME = 'auryveth-neo'
FORMAT = 'auryveth-neo-v1'


def package_neo(out, chat_manifest, think_manifest=None):
    chat, _ = verify_manifest(chat_manifest)
    if 'model' not in chat or 'tokenizer' not in chat:
        raise ValueError('The chat manifest must include a model and its SentencePiece tokenizer.')
    chat_sha = file_sha256(chat['model'])
    tokenizer = SentencePieceTokenizer(chat['tokenizer'])
    base = torch.load(chat['model'], map_location='cpu', weights_only=True)
    base_config = base['meta']['config']
    if base_config['backbone'] != 'recurrent' or base_config['planner_steps'] != 0:
        raise ValueError('Expected a recurrent chat model without a planner.')
    if base['meta']['tokenizer_contract'] != tokenizer.model_contract():
        raise ValueError('Chat checkpoint and tokenizer contracts differ.')
    payload = {'format': FORMAT, 'chat': base, 'think_meta': None,
               'think_overrides': {}, 'prompt_variant': None}
    metadata = {'name': NAME, 'format': FORMAT, 'chat_sha256': chat_sha}
    if think_manifest is not None:
        think, info = verify_manifest(think_manifest)
        if (info.get('task') != 'switch-transitions-v1' or info.get('cli_qualified') is not True
                or info.get('prompt_variant') != 5 or info.get('baseline_sha256') != chat_sha):
            raise ValueError('The switch checkpoint must be qualified against this chat checkpoint.')
        if file_sha256(chat['tokenizer']) != file_sha256(think['tokenizer']):
            raise ValueError('The checkpoints must share a tokenizer.')
        planner = torch.load(think['model'], map_location='cpu', weights_only=True)
        base_state, planner_state = base['state_dict'], planner['state_dict']
        if (planner['meta']['config'] != {**base_config, 'planner_steps': 3}
                or not base_state.keys() <= planner_state.keys()):
            raise ValueError('Expected the chat model and its three-pass planner.')
        overrides = {key: value for key, value in planner_state.items()
                     if key not in base_state or not torch.equal(base_state[key], value)}
        if any(not any(part in key for part in ('.planner.', '.mixer.', '.mixer_norm.')) for key in overrides):
            raise ValueError('Unexpected weight differences outside the planner and recurrent mixers.')
        if base['meta']['memory'] != planner['meta']['memory']:
            raise ValueError('Source checkpoints must have identical memory metadata.')
        payload.update(think_meta=planner['meta'], think_overrides=overrides, prompt_variant=5)
        metadata['think_sha256'] = file_sha256(think['model'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.save(payload, out/'AURYVETH-NEO.pt')
    shutil.copyfile(chat['tokenizer'], out/'tokenizer.model')
    return write_manifest(out/'manifest.json', {'model': 'AURYVETH-NEO.pt', 'tokenizer': 'tokenizer.model'},
                          metadata=metadata)

def load_neo(manifest):
    files, metadata = verify_manifest(manifest)
    if metadata.get('format') != FORMAT or metadata.get('name') != NAME:
        raise ValueError('Expected an AURYVETH-NEO bundle.')
    tokenizer = SentencePieceTokenizer(files['tokenizer'])
    bundle = torch.load(files['model'], map_location='cpu', weights_only=True)
    if bundle.get('format') != FORMAT or bundle.get('prompt_variant') not in (None, 5):
        raise ValueError('Invalid AURYVETH-NEO model format.')
    base = bundle['chat']
    chat = IntegratedLM._from_checkpoint_payload(base, tokenizer)
    if bundle['think_meta'] is None and bundle['prompt_variant'] is None and not bundle['think_overrides']:
        return chat, None, tokenizer, None
    if bundle['think_meta'] is None or bundle['prompt_variant'] != 5:
        raise ValueError('Invalid AURYVETH-NEO planner metadata.')
    think = {'meta': bundle['think_meta'], 'state_dict': base['state_dict'] | bundle['think_overrides']}
    return (chat,
            IntegratedLM._from_checkpoint_payload(think, tokenizer), tokenizer, bundle['prompt_variant'])
