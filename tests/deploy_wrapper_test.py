"""Offline generated-cell tests. Run with uv and an existing Python interpreter."""
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generated(source, model=None, revision=None):
    result = subprocess.run(
        [os.environ.get('NODE_BINARY', 'node'), '--input-type=module', '-e',
         "import {buildCode} from './deploy.mjs'; let s=''; for await (const c of process.stdin) s+=c; const p=JSON.parse(s); process.stdout.write(buildCode(p.source,p.model,'test-token',p.revision));"],
        input=json.dumps(dict(source=source, model=model, revision=revision)),
        text=True, capture_output=True, cwd=ROOT, check=True,
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

    def test_malformed_source_reports_syntax_error(self):
        with self.assertRaises(SyntaxError):
            exec(generated('CONFIG = {'), self.scope)
        self.assertIn('SyntaxError', self.states[-1]['detail'])


if __name__ == '__main__':
    unittest.main()
