import { readFile } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';

export const MARKER = '# colab-inference:managed-deploy:v1';
const CDP = 'http://127.0.0.1:9222';

export function notebookKey(value) {
  const u = new URL(value);
  if (u.origin !== 'https://colab.research.google.com' ||
      !/^\/(?:drive\/[^/]+|github\/.+\.ipynb)$/.test(u.pathname)) {
    throw new Error('Expected a saved Colab /drive/... or /github/...ipynb notebook URL');
  }
  return u.origin + u.pathname;
}

export function parseArgs(args) {
  const options = { port: 3000, timeout: 600 };
  const names = { '--tab-id': 'tabId', '--notebook-url': 'notebookUrl', '--model': 'model', '--revision': 'revision', '--port': 'port', '--timeout': 'timeout' };
  for (let i = 0; i < args.length; i++) {
    const name = names[args[i]];
    if (!name || !args[i + 1] || args[i + 1].startsWith('--')) throw new Error(`Unknown option or missing value: ${args[i]}`);
    if (Object.hasOwn(options, name) && !['port', 'model', 'timeout'].includes(name)) throw new Error(`Duplicate selector: ${args[i]}`);
    options[name] = args[++i];
  }
  if (Boolean(options.tabId) === Boolean(options.notebookUrl)) throw new Error('Specify exactly one of --tab-id or --notebook-url');
  if (options.notebookUrl) options.notebookUrl = notebookKey(options.notebookUrl);
  if (!/^[A-Za-z0-9_-]+$/.test(options.tabId ?? 'valid')) throw new Error('Invalid tab ID');
  if (Boolean(options.model) !== Boolean(options.revision)) throw new Error('--model and --revision must be supplied together');
  if (options.revision && !/^[a-fA-F0-9]{40}$/.test(options.revision)) throw new Error('Revision must be a 40-hex Hugging Face commit');
  if (options.model && !/^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(options.model)) throw new Error('Model must be a Hugging Face org/model ID');
  for (const [key, max] of [['port', 65535], ['timeout', 3600]]) {
    if (!/^\d+$/.test(String(options[key])) || Number(options[key]) < 1 || Number(options[key]) > max) throw new Error(`Invalid ${key}`);
    options[key] = Number(options[key]);
  }
  return options;
}

export function selectTarget(tabs, options) {
  const matches = tabs.filter(t => {
    if (t.type !== 'page') return false;
    try { return options.tabId ? t.id === options.tabId && Boolean(notebookKey(t.url)) : notebookKey(t.url) === options.notebookUrl; }
    catch { return false; }
  });
  if (matches.length !== 1) throw new Error(`Expected one matching notebook tab; found ${matches.length}`);
  const tab = matches[0];
  const ws = new URL(tab.webSocketDebuggerUrl);
  if (ws.protocol !== 'ws:' || !['localhost', '127.0.0.1', '[::1]'].includes(ws.hostname) || ws.port !== '9222' || ws.pathname !== `/devtools/page/${tab.id}` || ws.username || ws.password) throw new Error('Unexpected CDP WebSocket endpoint');
  return { ...tab, key: notebookKey(tab.url) };
}

export function buildCode(source, model, token, revision = null) {
  const fingerprint = createHash('sha256').update(JSON.stringify({ source, model, revision })).digest('hex');
  const data = Buffer.from(JSON.stringify({ source, model: model ?? null, revision, token, fingerprint })).toString('base64');
  return `${MARKER}
# Tool-owned: reruns replace this whole cell. Keep personal code in other cells.
import ast as _ci_ast, base64 as _ci_b64, json as _ci_json, traceback as _ci_tb
from google.colab import output as _ci_output
_ci_data = _ci_json.loads(_ci_b64.b64decode('${data}'))
def _ci_report(status, detail=''):
    _ci_output.eval_js('window.top.postMessage(' + _ci_json.dumps({'channel': 'colab-inference:deploy-status:v1', 'token': _ci_data['token'], 'status': status, 'detail': detail}) + ', "https://colab.research.google.com")', ignore_result=True)
_ci_report('running')
try:
    if globals().get('_ci_previous') == _ci_data['fingerprint']:
        _ci_report('complete', 'Already executed this source/model in this runtime; skipped duplicate startup')
    else:
        if globals().get('_ci_started'):
            raise RuntimeError('Restart the Colab runtime before deploying changed code/model or retrying a failed startup; the managed cell is updated')
        _ci_tree = _ci_ast.parse(_ci_data['source'], filename='colab_server.py')
        _ci_configs = [n for n in _ci_tree.body if isinstance(n, _ci_ast.Assign) and any(isinstance(t, _ci_ast.Name) and t.id == 'CONFIG' for t in n.targets)]
        if len(_ci_configs) != 1 or not isinstance(_ci_configs[0].value, _ci_ast.Dict):
            raise RuntimeError('Expected exactly one literal CONFIG dictionary in colab_server.py')
        _ci_keys = [i for i, k in enumerate(_ci_configs[0].value.keys) if isinstance(k, _ci_ast.Constant) and k.value == 'model_id']
        if len(_ci_keys) != 1:
            raise RuntimeError('Expected exactly one CONFIG model_id')
        if _ci_data['model'] is not None:
            _ci_revisions = [i for i, k in enumerate(_ci_configs[0].value.keys) if isinstance(k, _ci_ast.Constant) and k.value == 'model_revision']
            if len(_ci_revisions) != 1:
                raise RuntimeError('Custom model deployment requires the reproducible server with CONFIG model_revision support')
            _ci_configs[0].value.values[_ci_keys[0]] = _ci_ast.Constant(value=_ci_data['model'])
            _ci_configs[0].value.values[_ci_revisions[0]] = _ci_ast.Constant(value=_ci_data['revision'])
        _ci_ast.fix_missing_locations(_ci_tree)
        _ci_started = True
        exec(compile(_ci_tree, 'colab_server.py', 'exec'), globals())
        if not globals().get('tunnel_url'):
            raise RuntimeError('Server script finished without a tunnel URL; inspect notebook output')
        _ci_previous = _ci_data['fingerprint']
        _ci_report('complete', 'Server script completed; verify the endpoint separately')
except BaseException:
    _ci_report('error', _ci_tb.format_exc())
    raise
`;
}

// Runs in the selected page. All user data arrives through JSON, never interpolation.
export async function updateCell(payload) {
  const key = () => location.origin + location.pathname;
  if (key() !== payload.key) throw new Error('Notebook navigated away from the selected target');
  const nb = globalThis.colab?.global?.notebook;
  if (!Array.isArray(nb?.cells)) throw new Error('Colab notebook API unavailable');
  const owned = nb.cells.filter(c => typeof c.getText === 'function' && c.getText().split(/\r?\n/)[0] === payload.marker);
  if (owned.length !== 1) throw new Error(`Expected exactly one code cell beginning with ${payload.marker}; found ${owned.length}. Create that cell manually first.`);
  const cell = owned[0];
  if (typeof cell.setText !== 'function' || typeof cell.executeCellCommand !== 'function') throw new Error('Unsupported Colab code cell API');
  if (globalThis.__colabInferenceDeploy?.status === 'running' || globalThis.__colabInferenceDeploy?.status === 'pending') throw new Error('A deployment is already pending/running; inspect it or restart the runtime and reload the notebook');
  await cell.setText(payload.code);
  if (globalThis.colab?.global?.notebook !== nb || key() !== payload.key || !nb.cells.includes(cell) || cell.getText() !== payload.code || nb.cells.filter(c => typeof c.getText === 'function' && c.getText().split(/\r?\n/)[0] === payload.marker).length !== 1) throw new Error('Notebook/cell ownership changed before execution');
  if (['pending', 'running'].includes(globalThis.__colabInferenceDeploy?.status)) throw new Error('A concurrent deployment already started');
  globalThis.__colabInferenceDeploy = { token: payload.token, status: 'pending' };
  // Colab evaluates output JS in a cross-origin iframe, not in this page.
  const listener = event => {
    let origin;
    try { origin = new URL(event.origin); } catch { return; }
    if (origin.protocol !== 'https:' || !(origin.hostname.endsWith('.googleusercontent.com') || origin.origin === 'https://colab.research.google.com')) return;
    const state = event.data;
    if (state?.channel !== 'colab-inference:deploy-status:v1' || state.token !== payload.token || !['running', 'error', 'complete'].includes(state.status)) return;
    globalThis.__colabInferenceDeploy = { token: payload.token, status: state.status, detail: String(state.detail ?? '') };
    if (state.status !== 'running') globalThis.removeEventListener('message', listener);
  };
  globalThis.addEventListener('message', listener);
  // Do not await execution: some Colab versions resolve only after the cell finishes.
  try {
    Promise.resolve(cell.executeCellCommand()).catch(e => {
      globalThis.removeEventListener('message', listener);
      globalThis.__colabInferenceDeploy = { token: payload.token, status: 'error', detail: String(e) };
    });
  } catch (e) {
    globalThis.removeEventListener('message', listener);
    globalThis.__colabInferenceDeploy = { token: payload.token, status: 'error', detail: String(e) };
    throw e;
  }
  return { submitted: true };
}

export function evaluateResult(message) {
  if (message.error) throw new Error(`CDP: ${message.error.message}`);
  if (message.result?.exceptionDetails) throw new Error(message.result.exceptionDetails.exception?.description ?? message.result.exceptionDetails.text);
  if (message.result?.result?.subtype === 'error') throw new Error(message.result.result.description);
  return message.result?.result?.value;
}

export async function connect(url) {
  const ws = new WebSocket(url);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { ws.close(); reject(new Error('CDP connection timed out')); }, 10000);
    ws.addEventListener('open', () => { clearTimeout(timer); resolve(); }, { once: true });
    ws.addEventListener('error', () => { clearTimeout(timer); reject(new Error('CDP connection failed')); }, { once: true });
  });
  let id = 0;
  const pending = new Map();
  ws.addEventListener('message', e => {
    let message;
    try { message = JSON.parse(e.data); } catch { return; }
    const request = pending.get(message.id);
    if (request) { pending.delete(message.id); clearTimeout(request.timer); request.resolve(message); }
  });
  const fail = () => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error('CDP connection closed')); } pending.clear(); };
  ws.addEventListener('close', fail);
  ws.addEventListener('error', fail);
  return {
    close: () => ws.close(),
    async evaluate(expression) {
      const requestId = ++id;
      const response = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => { pending.delete(requestId); reject(new Error('CDP evaluation timed out')); }, 15000);
        pending.set(requestId, { resolve, reject, timer });
        try { ws.send(JSON.stringify({ id: requestId, method: 'Runtime.evaluate', params: { expression, returnByValue: true, awaitPromise: true } })); }
        catch (e) { clearTimeout(timer); pending.delete(requestId); reject(e); }
      });
      return evaluateResult(response);
    },
  };
}

export async function waitForExecution(client, key, token, timeout, { now = Date.now, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)) } = {}) {
  const deadline = now() + timeout * 1000;
  while (now() < deadline) {
    const state = await client.evaluate(`(() => { if (location.origin + location.pathname !== ${JSON.stringify(key)}) throw new Error('Notebook navigated during deployment'); return globalThis.__colabInferenceDeploy; })()`);
    if (state?.token !== token) throw new Error('Execution status lost or replaced; inspect notebook before retrying');
    if (state.status === 'error') throw new Error(state.detail);
    if (state.status === 'complete') {
      return state.detail;
    }
    await sleep(1000);
  }
  throw new Error('Execution status timed out (cell may still run). Inspect notebook output and any prompts manually; no success was confirmed.');
}

export async function main(args) {
  const options = parseArgs(args);
  if (typeof WebSocket !== 'function') throw new Error('Node.js 22.15+ is required (built-in WebSocket); no dependencies are installed automatically');
  const response = await fetch(`${CDP}/json/list`, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`CDP tab listing failed: HTTP ${response.status}`);
  const tab = selectTarget(await response.json(), options);
  console.log(`Selected notebook: ${tab.key} (tab ${tab.id})`);
  const token = randomUUID();
  const source = await readFile(new URL('./colab_server.py', import.meta.url), 'utf8');
  const payload = { key: tab.key, marker: MARKER, code: buildCode(source, options.model, token, options.revision), token };
  const client = await connect(tab.webSocketDebuggerUrl);
  try {
    await client.evaluate(`(${updateCell.toString()})(${JSON.stringify(payload)})`);
    const detail = await waitForExecution(client, tab.key, token, options.timeout);
    console.log(detail);
    console.log(`Copy the tunnel URL from the notebook, then run: node proxy.mjs <tunnel-url> ${options.port}`);
  } finally { client.close(); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).catch(e => { console.error(`ERROR: ${e.message}`); process.exitCode = 1; });
}
