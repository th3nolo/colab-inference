"""Generate only setup/config/load cells; leave sibling API/auth/tunnel cells intact."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def render():
    template = (ROOT / "scripts/runtime_setup.py").read_text(encoding="utf-8")
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    # Literal text stays reviewable; no executable encoded payloads.
    return template.replace('LOCK_TEXT = "__LOCK_TEXT__"', 'LOCK_TEXT = r"""' + lock + '"""')

def sync():
    path = ROOT / "colab_server.py"
    server = path.read_text(encoding="utf-8")
    start = server.index('# BEGIN GENERATED SETUP')
    end = server.index('# END GENERATED SETUP') + len('# END GENERATED SETUP')
    block = '# BEGIN GENERATED SETUP\n' + render() + '\nCLOUDFLARED_PATH = setup_runtime(CONFIG)\nprint("[1/4] Dependencies installed")\n# END GENERATED SETUP'
    server = server[:start] + block + server[end:]
    path.write_text(server, encoding="utf-8", newline="\n")
    notebook_path = ROOT / "colab_inference_server.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    tree = ast.parse(server)
    config = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    config_source = server[server.index("CONFIG ="):server.index("# BEGIN GENERATED SETUP")].rstrip()
    load_start = server.index('from transformers import AutoModelForCausalLM, AutoTokenizer')
    load_end = server.index('\n# ', load_start)
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] != 'code': continue
        if source.startswith('CONFIG = '): cell['source'] = config_source.splitlines(keepends=True)
        elif '# BEGIN GENERATED SETUP' in source or source.startswith('!pip install') or source.startswith('import subprocess, sys'): cell['source'] = block.splitlines(keepends=True)
        elif source.startswith('from transformers import AutoModelForCausalLM, AutoTokenizer'): cell['source'] = server[load_start:load_end].rstrip().splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook, indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')

if __name__ == '__main__': sync()
