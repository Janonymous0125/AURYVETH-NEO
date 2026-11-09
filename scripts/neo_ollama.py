"""Serve AURYVETH-NEO to the Ollama CLI over a local HTTP endpoint."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time

import torch

from model.neo import NAME, load_neo, package_neo
from model.artifacts import file_sha256
from scripts.chat import reply
from scripts.switch_reasoning import STATES, deliberate_switch


def switch_args(text):
    parts = text.split()
    return parts[1:] if parts and parts[0].lower() in ('think:', '/think') else None


def answer(models, messages, options):
    chat, planner, tokenizer, variant = models
    if any(not isinstance(m, dict) or m.get('role') != ('user' if i % 2 == 0 else 'assistant')
           or not isinstance(m.get('content'), str) or m.get('images') or m.get('tool_calls')
           for i, m in enumerate(messages)):
        raise ValueError('Use alternating text user and assistant messages.')
    if not messages or len(messages) % 2 == 0:
        raise ValueError('Chat must end with a user message.')
    request = switch_args(messages[-1]['content'])
    if request is not None:
        if planner is None:
            raise ValueError('This bundle has no qualified switch planner.')
        if len(request) < 2:
            raise ValueError('Use think: off toggle keep')
        language = 'zh' if request[0] in STATES['zh'] else 'en'
        result, states = deliberate_switch(planner, tokenizer, request[0], request[1:],
                                           language=language, variant=variant)
        steps = [f'{i}. {command} → {state}'
                 for i, (command, state) in enumerate(zip(request[1:], states, strict=True), 1)]
        return result, '\n'.join([f'Start: {request[0]}', *steps])
    # Interactive Ollama clients echo previous turns back on each request.
    # Drop both sides of any prior switch turn before forming the chat prompt.
    history = []
    for user, assistant in zip(messages[:-1:2], messages[1::2], strict=True):
        if switch_args(user['content']) is None:
            history.extend((user, assistant))
    history.append(messages[-1])
    budget = options.get('num_predict', 80)
    temperature = options.get('temperature', 0.)
    if type(budget) is not int or type(temperature) not in (int, float) or not 0 <= temperature <= 1:
        raise ValueError('Expected integer num_predict and temperature between 0 and 1.')
    return reply(chat, tokenizer, history, max_new_tokens=budget, temperature=temperature), ''


def handler_for(models, model_path):
    details = {'format': 'pytorch', 'family': 'auryveth-neo', 'parameter_size': '30M base',
               'quantization_level': 'F32'}
    info = {'capabilities': ['completion'], 'details': details, 'model_info': {
        'general.architecture': NAME, f'{NAME}.context_length': models[0].config.block_size}}
    stat = model_path.stat()
    listing = {'name': NAME + ':latest', 'model': NAME + ':latest', 'size': stat.st_size,
               'digest': file_sha256(model_path), 'details': details,
               'modified_at': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def write_response(self, status, value, *, stream=False):
            rows = value if stream else [value]
            body = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows).encode('utf8') + b'\n'
            self.send_response(status)
            self.send_header('Content-Type', 'application/x-ndjson' if stream else 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            self.send_response(200 if self.path == '/' else 404)
            self.end_headers()

        def do_GET(self):
            if self.path == '/':
                self.write_response(200, {'service': 'AURYVETH-NEO'})
            elif self.path == '/api/version':
                self.write_response(200, {'version': '0.1.0'})
            elif self.path == '/api/tags':
                self.write_response(200, {'models': [listing]})
            else:
                self.write_response(404, {'error': 'Endpoint unavailable.'})

        def do_POST(self):
            if self.path not in ('/api/show', '/api/generate', '/api/chat'):
                self.write_response(404, {'error': 'Endpoint unavailable.'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 1_048_576:
                    raise ValueError('Expected a JSON body up to 1 MiB.')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object.')
                name = data.get('model') or data.get('name')
                if not isinstance(name, str) or name.lower() not in (NAME, NAME + ':latest'):
                    self.write_response(404, {'error': 'Unknown model.'})
                    return
                if self.path == '/api/show':
                    self.write_response(200, info)
                    return
                if any(data.get(key) for key in ('images', 'tools', 'format', 'raw', 'suffix', 'system', 'think')):
                    raise ValueError('This endpoint supports text chat and explicit think: switch commands.')
                options = data.get('options')
                if options is None:
                    options = {}
                stream = data.get('stream', True)
                if not isinstance(options, dict) or type(stream) is not bool:
                    raise ValueError('Expected options object and boolean stream.')
                chat_request = self.path == '/api/chat'
                if chat_request:
                    messages = data.get('messages', [])
                else:
                    prompt = data.get('prompt', '')
                    if not isinstance(prompt, str):
                        raise ValueError('Expected a text prompt.')
                    messages = [{'role': 'user', 'content': prompt}] if prompt.strip() else []
                if not isinstance(messages, list):
                    raise ValueError('Expected a messages array.')
                start = time.monotonic_ns()
                result, trace = answer(models, messages, options) if messages else ('', '')
                common = {'model': NAME, 'created_at': datetime.now(timezone.utc).isoformat()}

                def content(text, thinking=''):
                    if chat_request:
                        return {'message': {'role': 'assistant', 'content': text,
                                            **({'thinking': thinking} if thinking else {})}}
                    return {'response': text, **({'thinking': thinking} if thinking else {})}

                final = {**common, **content('' if stream else result, '' if stream else trace), 'done': True,
                         'done_reason': 'stop' if messages else 'load', 'total_duration': time.monotonic_ns() - start}
                if stream:
                    rows = [{**common, **content('', trace + '\n'), 'done': False}] if trace else []
                    rows.extend(({**common, **content(result), 'done': False}, final))
                    self.write_response(200, rows, stream=True)
                else:
                    self.write_response(200, final)
            except (ValueError, UnicodeError) as exc:
                self.write_response(400, {'error': str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('checkpoints/AURYVETH-NEO'))
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--chat-manifest', type=Path, help='Training manifest for --build')
    parser.add_argument('--think-manifest', type=Path, help='Optional qualified switch manifest for --build')
    parser.add_argument('--port', type=int, default=11435)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.build:
        if args.chat_manifest is None:
            parser.error('--build requires --chat-manifest')
        package_neo(args.bundle, args.chat_manifest, args.think_manifest)
        print(f'Packaged AURYVETH-NEO in {args.bundle}')
        return
    models = load_neo(args.bundle/'manifest.json')
    with HTTPServer(('127.0.0.1', args.port), handler_for(models, args.bundle/'AURYVETH-NEO.pt')) as server:
        print(f'AURYVETH-NEO serving Ollama clients at http://127.0.0.1:{args.port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
