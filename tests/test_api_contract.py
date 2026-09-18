"""Handler regressions with real FastAPI/Pydantic and a model boundary fixture.

The socket suite separately exercises startup, middleware and HTTP serialization.
"""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
import time
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


class Scalar(int):
    def item(self):
        return int(self)


class Tensor:
    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))

    def __getitem__(self, index):
        return [Scalar(t) for t in self.tokens]


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return 'prompt'

    def __call__(self, text, **kwargs):
        return Inputs(input_ids=Tensor([10, 11]))

    def decode(self, tokens, **kwargs):
        return 'answer'


class Model:
    def __init__(self):
        self.device = SimpleNamespace(type='cpu')
        self.config = SimpleNamespace(max_position_embeddings=8192)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.generated = [7, 2]
        self.calls = []

    def parameters(self):
        yield SimpleNamespace(device=self.device)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return Tensor([10, 11] + self.generated)


def load(source):
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {'CONFIG', 'LOADED_MODEL_ID'}
            for t in node.targets
        ):
            keep.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
            'ChatMessage', 'ChatRequest', 'chat', 'generate_completion', 'list_models', 'health'
        }:
            keep.append(node)
    ns = dict(BaseModel=BaseModel, HTTPException=HTTPException, app=FastAPI(),
              torch=SimpleNamespace(no_grad=nullcontext), uuid=uuid, _time=time,
              model=Model(), tokenizer=Tokenizer(),
              Field=Field,
              inference_slots=threading.BoundedSemaphore(1))
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<api definitions>', 'exec'), ns)
    return ns


def sources():
    server = (ROOT / 'colab_server.py').read_text(encoding='utf-8')
    notebook = json.loads((ROOT / 'colab_inference_server.ipynb').read_text(encoding='utf-8'))
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    # Exclude notebook shell cells; parse only config, model-load and API cells.
    nb = '\n\n'.join(c for c in cells if c.startswith((
        'CONFIG =', 'from transformers import', 'import threading, uuid')))
    return {'server': server, 'notebook': nb}


class APIContractTests(unittest.TestCase):
    def each(self):
        for name, source in sources().items():
            yield name, load(source)

    def request(self, ns, **kwargs):
        return ns['ChatRequest'](messages=[ns['ChatMessage'](role='user', content='Hello')], **kwargs)

    def test_default_request_and_usage(self):
        for name, ns in self.each():
            with self.subTest(entrypoint=name):
                result = ns['chat'](self.request(ns))
                self.assertEqual(result['model'], ns['LOADED_MODEL_ID'])
                self.assertEqual(result['choices'][0]['finish_reason'], 'stop')
                self.assertEqual(result['usage']['prompt_tokens'], 2)
                self.assertEqual(result['usage']['completion_tokens'], 2)
                self.assertEqual(result['usage']['total_tokens'], 4)
                call = ns['model'].calls[0]
                self.assertEqual(call['max_new_tokens'], 512)
                self.assertEqual(call['temperature'], 0.1)

    def test_reject_before_tokenization_or_generation(self):
        for name, ns in self.each():
            for kwargs, param in [({'model': 'not-loaded'}, 'model'), ({'stream': True}, 'stream')]:
                with self.subTest(entrypoint=name, kwargs=kwargs):
                    with self.assertRaises(HTTPException) as error:
                        ns['chat'](self.request(ns, **kwargs))
                    self.assertEqual(error.exception.status_code, 400)
                    self.assertEqual(error.exception.detail['param'], param)
                    self.assertTrue(error.exception.detail['message'])
                    self.assertEqual(ns['tokenizer'].calls, 0)
                    self.assertEqual(ns['model'].calls, [])
                    self.assertTrue(ns['inference_slots'].acquire(blocking=False))
                    ns['inference_slots'].release()

    def test_loaded_identity_survives_config_edit(self):
        for name, ns in self.each():
            with self.subTest(entrypoint=name):
                loaded = ns['LOADED_MODEL_ID']
                ns['CONFIG']['model_id'] = 'edited-but-not-loaded'
                self.assertEqual(ns['list_models']()['data'][0]['id'], loaded)
                self.assertEqual(ns['health']()['model'], loaded)
                self.assertEqual(ns['chat'](self.request(ns, model=loaded))['model'], loaded)
                with self.assertRaises(HTTPException):
                    ns['chat'](self.request(ns, model='edited-but-not-loaded'))

    def test_finish_reasons_from_generated_tokens(self):
        cases = [
            ([7, 8, 9], 2, 3, 'length'),
            ([7, 2], 2, 3, 'stop'),
            ([7, 8, 2], 2, 3, 'stop'),  # EOS exactly at budget
            ([7, 8, 3], [2, 3], 3, 'stop'),
            ([7, 8, 9], [2, 3], 3, 'length'),
            ([7, 8, 9], None, 3, 'length'),
            ([7], None, 3, 'stop'),  # Other generation stopping criterion
            ([9], 2, 1, 'length'),
            ([2], 2, 1, 'stop'),
        ]
        for name, ns in self.each():
            for tokens, eos, budget, expected in cases:
                with self.subTest(entrypoint=name, tokens=tokens, eos=eos, budget=budget):
                    ns['model'].generated = tokens
                    ns['model'].generation_config.eos_token_id = eos
                    result = ns['chat'](self.request(ns, max_tokens=budget, stream=False))
                    self.assertEqual(result['choices'][0]['finish_reason'], expected)
                    self.assertEqual(result['usage']['completion_tokens'], len(tokens))
                    self.assertEqual(ns['model'].calls[-1]['eos_token_id'], eos)

    def test_notebook_api_parity(self):
        trees = {name: ast.parse(source) for name, source in sources().items()}
        for name in ['ChatMessage', 'ChatRequest', 'chat', 'generate_completion', 'list_models', 'health']:
            nodes = [next(n for n in tree.body if getattr(n, 'name', None) == name)
                     for tree in trees.values()]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]), name)


if __name__ == '__main__':
    unittest.main()
