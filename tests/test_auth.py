"""Mock inference tests; needs the existing FastAPI/Pydantic/httpx test runtime."""
import ast
import contextlib
import json
import os
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'test-only-token-' + 'x' * 32


def load_server():
    source = (ROOT / 'colab_server.py').read_text(encoding='utf-8')
    config = source[source.index('CONFIG ='):source.index('import subprocess, sys')]
    api = source[source.index('import threading'):source.index('threading.Thread(')]
    ns = {'__name__': 'test_server'}
    model = Mock()
    model.device = 'cpu'
    model.config.max_position_embeddings = 8192
    model.parameters.side_effect = lambda: iter([types.SimpleNamespace(device=types.SimpleNamespace(type='cpu'))])
    inputs = {'input_ids': types.SimpleNamespace(shape=(1, 4))}
    class Inputs(dict):
        def to(self, device):
            return self
    tokenizer = Mock(return_value=Inputs(inputs))
    tokenizer.apply_chat_template.return_value = 'hello'
    tokenizer.decode.return_value = 'answer'
    class Output:
        shape = (1, 6)
        def __getitem__(self, item):
            return [0, 1, 2, 3, 4, 5]
    model.generate.return_value = Output()
    ns.update(model=model, tokenizer=tokenizer)
    with patch.dict(os.environ, {'COLAB_API_TOKEN': TOKEN}), patch.dict('sys.modules', {
        'torch': types.SimpleNamespace(no_grad=contextlib.nullcontext),
        'uvicorn': types.SimpleNamespace(),
    }):
        exec(compile(config, 'config', 'exec'), ns)
        exec(compile(api, 'api', 'exec'), ns)
    return ns


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_server()
        self.client = TestClient(self.ns['app'], raise_server_exceptions=False)
        self.headers = {'Authorization': 'Bearer ' + TOKEN}
        self.payload = {'messages': [{'role': 'user', 'content': 'hello'}]}

    def post(self, **changes):
        return self.client.post('/v1/chat/completions', headers=self.headers, json=self.payload | changes)

    def test_auth_precedes_parsing_and_all_inference(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'}, {'Authorization': 'Basic ' + TOKEN},
                        [('Authorization', 'Bearer '+TOKEN), ('Authorization', 'Bearer '+TOKEN)]):
            response = self.client.post('/v1/chat/completions', headers=headers, content='{invalid')
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers['www-authenticate'], 'Bearer')
        self.ns['tokenizer'].assert_not_called()
        self.ns['model'].generate.assert_not_called()
        for route in ('/health', '/v1/models', '/docs', '/openapi.json'):
            self.assertEqual(self.client.get(route).status_code, 401)
        for route in ('/health', '/v1/models'):
            self.assertEqual(self.client.get(route, headers=self.headers).status_code, 200)

    def test_valid_request_and_output_budget(self):
        self.assertEqual(self.post(max_tokens=17).status_code, 200)
        self.assertEqual(self.ns['model'].generate.call_args.kwargs['max_new_tokens'], 17)
        for value in (0, -1, 1025, 1.5, True, '10'):
            self.assertEqual(self.post(max_tokens=value).status_code, 422, value)

    def test_pre_tokenization_limits(self):
        for messages in ([], [{'role': 'user', 'content': 'x'}]*65,
                         [{'role': 'user', 'content': 'x'*9000}]*2):
            self.assertIn(self.post(messages=messages).status_code, (413, 422))
        self.ns['tokenizer'].assert_not_called()
        self.ns['tokenizer'].apply_chat_template.assert_not_called()

    def test_token_and_context_limits(self):
        self.ns['tokenizer'].return_value['input_ids'].shape = (1, 4097)
        self.assertEqual(self.post().status_code, 413)
        self.ns['model'].generate.assert_not_called()
        self.ns['tokenizer'].return_value['input_ids'].shape = (1, 4)
        self.ns['model'].config.max_position_embeddings = 10
        self.assertEqual(self.post(max_tokens=7).status_code, 413)
        self.ns['model'].generate.assert_not_called()
        self.assertEqual(self.post(max_tokens=6).status_code, 200)

    def test_concurrency_rejects_before_tokenization(self):
        entered, release = threading.Event(), threading.Event()
        original = self.ns['model'].generate.return_value
        def blocked(**kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test timeout')
            return original
        self.ns['model'].generate.side_effect = blocked
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.post)
            try:
                self.assertTrue(entered.wait(5))
                response = self.post()
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response.headers['retry-after'], '1')
                self.assertEqual(self.ns['tokenizer'].call_count, 1)
            finally:
                release.set()
            self.assertEqual(future.result().status_code, 200)

    def test_slot_released_on_generation_and_tokenizer_errors(self):
        for target in (self.ns['model'].generate, self.ns['tokenizer'].apply_chat_template):
            target.side_effect = RuntimeError('mock failure')
            self.assertEqual(self.post().status_code, 500)
            target.side_effect = None
            self.assertEqual(self.post().status_code, 200)

    def test_invalid_config_fails_without_disclosing_secret(self):
        source = (ROOT/'colab_server.py').read_text(encoding='utf-8')
        config = source[source.index('CONFIG ='):source.index('import subprocess, sys')]
        for value in ('', 'short', 'x'*257, 'contains spaces'+'x'*32):
            with patch.dict(os.environ, {'COLAB_API_TOKEN': value}):
                with self.assertRaises(RuntimeError) as ctx:
                    exec(config, {})
                if value:
                    self.assertNotIn(value, str(ctx.exception))

    def test_notebook_parity(self):
        source=(ROOT/'colab_server.py').read_text(encoding='utf-8')
        nb=json.loads((ROOT/'colab_inference_server.ipynb').read_text())
        config=''.join(nb['cells'][2]['source'])
        api=''.join(nb['cells'][8]['source'])
        self.assertEqual(ast.dump(ast.parse(config)), ast.dump(ast.parse(source[source.index('CONFIG ='):source.index('import subprocess, sys')])))
        self.assertEqual(ast.dump(ast.parse(api)), ast.dump(ast.parse(source[source.index('import threading'):source.index('import subprocess, re, os')])) )

if __name__ == '__main__':
    unittest.main()
