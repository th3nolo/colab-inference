"""Offline generated-cell tests. Run with uv and an existing Python interpreter."""
import ast
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from runtime_fixture import TOKEN, cells, model_modules

ROOT = Path(__file__).resolve().parents[1]


def generated(source, model=None, revision=None):
    result = subprocess.run(
        [os.environ.get('NODE_BINARY', 'node'), '--input-type=module', '-e',
         "import {buildCode} from './deploy.mjs'; let s=''; for await (const c of process.stdin) s+=c; const p=JSON.parse(s); process.stdout.write(buildCode(p.source,p.model,'test-token',p.revision));"],
        input=json.dumps(dict(source=source, model=model, revision=revision)),
        text=True, encoding='utf-8', capture_output=True, cwd=ROOT, check=True, timeout=10,
    )
    return result.stdout


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.states = []
        output = types.ModuleType('google.colab.output')

        def eval_js(script, ignore_result=False):
            self.assertTrue(ignore_result)
            self.assertTrue(script.startswith('window.top.postMessage('))
            payload = script[len('window.top.postMessage('):].rsplit(', "https://colab.research.google.com")', 1)[0]
            self.states.append(json.loads(payload))

        output.eval_js = eval_js
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        colab.output = output
        google.colab = colab
        self.old_modules = {name: sys.modules.get(name) for name in ['google', 'google.colab', 'google.colab.output']}
        sys.modules.update({'google': google, 'google.colab': colab, 'google.colab.output': output})
        self.scope = {}
        self.source = 'CONFIG = {"model_id": "default/model", "model_revision": "old"}\nstarts = globals().get("starts", 0) + 1\ntunnel_url = "https://fixture.invalid"\n'

    def tearDown(self):
        for name, old in self.old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

    def test_default_preserved_and_identical_rerun_skips_startup(self):
        code = generated(self.source)
        exec(code, self.scope)
        exec(code, self.scope)
        self.assertEqual(self.scope['CONFIG']['model_id'], 'default/model')
        self.assertEqual(self.scope['starts'], 1)
        self.assertEqual(self.states[-1]['status'], 'complete')

    def test_custom_model_revision_and_changed_model_restart(self):
        revision = 'a' * 40
        exec(generated(self.source, 'org/model', revision), self.scope)
        self.assertEqual(self.scope['CONFIG'], {'model_id': 'org/model', 'model_revision': revision})
        with self.assertRaisesRegex(RuntimeError, 'Restart'):
            exec(generated(self.source, 'org/other', revision), self.scope)
        self.assertEqual(self.scope['starts'], 1)
        self.assertEqual(self.states[-1]['status'], 'error')

    def test_traceback_and_failed_startup_retry_guard(self):
        code = generated('CONFIG = {"model_id": "org/model"}\nraise ValueError("fixture failure")')
        with self.assertRaisesRegex(ValueError, 'fixture failure'):
            exec(code, self.scope)
        self.assertIn('ValueError: fixture failure', self.states[-1]['detail'])
        with self.assertRaisesRegex(RuntimeError, 'Restart'):
            exec(code, self.scope)

    def test_missing_tunnel_is_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'without a tunnel URL'):
            exec(generated('CONFIG = {"model_id": "org/model"}'), self.scope)
        self.assertEqual(self.states[-1]['status'], 'error')

    def test_custom_model_requires_repro_support_before_startup(self):
        with self.assertRaisesRegex(RuntimeError, 'CONFIG model_revision support'):
            exec(generated('CONFIG = {"model_id": "org/model"}', 'new/model', 'a' * 40), self.scope)
        self.assertNotIn('_ci_started', self.scope)

    def test_real_server_source_preserved_through_config_override(self):
        source = (ROOT / 'colab_server.py').read_text(encoding='utf-8')
        original = ast.parse(source)
        config = next(n for n in original.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets))
        keys = [k.value for k in config.value.keys]
        self.assertIn('model_revision', keys)
        revision = 'b' * 40
        config.value.values[keys.index('model_id')] = ast.Constant(value='fixture/model')
        config.value.values[keys.index('model_revision')] = ast.Constant(value=revision)
        ast.fix_missing_locations(original)
        executed = []

        def mock_server_exec(code, namespace):
            # Exercise generation, parsing and compilation of the actual combined
            # source, but never run its installers/model/tunnel startup.
            self.assertIsInstance(code, types.CodeType)
            self.assertEqual(code.co_filename, 'colab_server.py')
            executed.append(code)
            namespace['tunnel_url'] = 'https://fixture.invalid'

        self.scope['exec'] = mock_server_exec
        exec(generated(source, 'fixture/model', revision), self.scope)
        self.assertEqual(ast.dump(self.scope['_ci_tree']), ast.dump(original))
        self.assertEqual(len(executed), 1)
        self.assertEqual(self.states[-1]['status'], 'complete')

    def test_malformed_source_reports_syntax_error(self):
        with self.assertRaises(SyntaxError):
            exec(generated('CONFIG = {'), self.scope)
        self.assertIn('SyntaxError', self.states[-1]['detail'])

    def test_generated_override_reaches_real_model_load_and_http_contract(self):
        for entrypoint in ('server', 'notebook'):
            with self.subTest(entrypoint=entrypoint):
                config, load, api = cells(entrypoint)
                # Execute actual config/load/router code through buildCode's
                # generated Python cell. Only GPU and tunnel/provider boundaries
                # are fixtures; socket startup is exercised by runtime.test.mjs.
                source = config + '\n' + load + '\n' + api[:api.index('threading.Thread(')]
                source += '\ntunnel_url = "https://fixture.invalid"\n'
                namespace, loads = {}, []
                revision = 'b' * 40
                with patch.dict(os.environ, {'COLAB_API_TOKEN': TOKEN}), patch.dict(sys.modules, model_modules(loads)):
                    exec(generated(source, 'fixture/custom', revision), namespace)
                    with TestClient(namespace['app']) as client:
                        response = client.post('/v1/chat/completions', headers={'Authorization': 'Bearer ' + TOKEN},
                                               json={'messages': [{'role': 'user', 'content': 'hello'}]})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['model'], 'fixture/custom')
                self.assertEqual(len(loads), 2)
                for kind, model_id, options in loads:
                    self.assertEqual(model_id, 'fixture/custom', kind)
                    self.assertEqual(options['revision'], revision, kind)
                    self.assertIs(options['trust_remote_code'], False, kind)
                self.assertIs(loads[0][2]['use_safetensors'], True)
                self.assertEqual(self.states[-1]['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
