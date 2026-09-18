import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
setup=module('runtime_setup',ROOT/'scripts/runtime_setup.py')
sync=module('sync_setup',ROOT/'scripts/sync_setup.py')

class SetupTests(unittest.TestCase):
    def test_checksum_failure_returns_no_executable(self):
        with patch.object(setup.urllib.request,'urlopen',return_value=io.BytesIO(b'tampered')):
            with self.assertRaisesRegex(RuntimeError,'SHA-256 mismatch'):
                setup.verified_download('https://example.test/bin','0'*64,100)

    def test_bounded_download(self):
        with patch.object(setup.urllib.request,'urlopen',return_value=io.BytesIO(b'oversized')):
            with self.assertRaisesRegex(RuntimeError,'size limit'):
                setup.verified_download('https://example.test/bin','0'*64,2)

    def test_valid_digest(self):
        with patch.object(setup.urllib.request,'urlopen',return_value=io.BytesIO(b'verified')):
            self.assertEqual(setup.verified_download('https://example.test/bin',hashlib.sha256(b'verified').hexdigest(),100),b'verified')

    def test_unsupported_runtime_does_not_download(self):
        with patch.object(setup.platform,'system',return_value='Windows'),patch.object(setup,'verified_download') as download:
            with self.assertRaises(RuntimeError): setup.setup_runtime({'model_revision':'a'*40})
            download.assert_not_called()

    def test_invalid_revision_fails_before_download(self):
        with patch.object(setup,'sys',SimpleNamespace(version_info=(3,12),modules={},executable=sys.executable)),patch.object(setup.platform,'system',return_value='Linux'),patch.object(setup.platform,'machine',return_value='x86_64'),patch.object(setup,'verified_download') as download:
            with self.assertRaises(ValueError):setup.setup_runtime({'model_revision':'main'})
            download.assert_not_called()

    def test_mock_setup_uses_locked_install_and_private_binaries(self):
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w:gz') as tar:
            entry=tarfile.TarInfo('uv-x86_64-unknown-linux-gnu/uv');entry.size=2
            tar.addfile(entry,io.BytesIO(b'uv'))
        with tempfile.TemporaryDirectory() as directory,patch.object(setup,'sys',SimpleNamespace(version_info=(3,12),modules={},executable=sys.executable)),patch.object(setup.platform,'system',return_value='Linux'),patch.object(setup.platform,'machine',return_value='x86_64'),patch.object(setup.tempfile,'mkdtemp',return_value=directory),patch.object(setup,'LOCK_TEXT','package==1.0 --hash=sha256:abc'),patch.object(setup,'verified_download',side_effect=[archive.getvalue(),b'cf']),patch.object(setup.subprocess,'check_call') as install:
            binary=setup.setup_runtime({'model_revision':'a'*40})
            self.assertEqual(Path(binary).read_bytes(),b'cf')
            args=install.call_args.args[0]
            for flag in ('--require-hashes','--no-deps','--only-binary','--no-config','--reinstall'):self.assertIn(flag,args)
            self.assertEqual(Path(directory,'requirements.lock').read_text(),'package==1.0 --hash=sha256:abc')
            self.assertFalse(any(k.startswith(('UV_','PIP_')) for k in install.call_args.kwargs['env']))

    def test_bad_uv_digest_never_installs(self):
        with patch.object(setup,'sys',SimpleNamespace(version_info=(3,12),modules={},executable=sys.executable)),patch.object(setup.platform,'system',return_value='Linux'),patch.object(setup.platform,'machine',return_value='x86_64'),patch.object(setup,'verified_download',side_effect=RuntimeError('SHA-256 mismatch')),patch.object(setup.subprocess,'check_call') as install:
            with self.assertRaises(RuntimeError):setup.setup_runtime({'model_revision':'a'*40})
            install.assert_not_called()

    def test_already_imported_runtime_requires_restart(self):
        fake_sys = SimpleNamespace(version_info=(3,12),modules={'torch':object()},executable=sys.executable)
        with patch.object(setup,'sys',fake_sys),patch.object(setup.platform,'system',return_value='Linux'),patch.object(setup.platform,'machine',return_value='x86_64'),patch.object(setup,'verified_download') as download:
            with self.assertRaisesRegex(RuntimeError,'Restart'):setup.setup_runtime({'model_revision':'a'*40})
            download.assert_not_called()

    def test_generated_setup_and_notebook_parity(self):
        server=(ROOT/'colab_server.py').read_text(encoding='utf-8')
        self.assertIn(sync.render(),server)
        cells=[''.join(c['source']) for c in json.loads((ROOT/'colab_inference_server.ipynb').read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
        for source in cells:
            if not source.startswith("!"): ast.parse(source)
        setup_cell=next(c for c in cells if '# BEGIN GENERATED SETUP' in c)
        self.assertIn(sync.render(),setup_cell)
        load=next(c for c in cells if c.startswith('from transformers'))
        self.assertIn(load,server)
        self.assertEqual(load.count('revision=CONFIG["model_revision"]'),2)
        self.assertIn('trust_remote_code=False',load)
        self.assertIn('use_safetensors=True',load)
        self.assertTrue(any('CLOUDFLARED_PATH' in c and 'nohup' in c for c in cells))

    def test_all_requirements_are_exact_and_hashed(self):
        import re
        lock=(ROOT/'requirements.lock').read_text()
        requirements=[line for line in lock.splitlines() if line and not line.startswith((' ','#'))]
        self.assertGreater(len(requirements),40)
        for line in requirements:
            self.assertTrue('==' in line or 'torch @ https://download.pytorch.org/' in line,line)
        for chunk in re.split(r'\n(?=[a-z])',lock):
            if re.search(r'^[a-z].*(?:==| @ )',chunk,re.M):self.assertIn('--hash=sha256:',chunk)
        self.assertIn('8695f3c6b7966d44560275b90c5c28e5091ba33ddbb1ab33b2173782ca1e9145',lock)

if __name__=='__main__':unittest.main()
