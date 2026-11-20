"""Run the selected frontier transport with exactly the supplied prompt and images."""
import json
import os
import subprocess


def provider():
    name = os.environ.get('TRALENS_FRONTIER_PROVIDER', 'chatgpt')
    if name == 'disabled':
        raise RuntimeError('Remote ChatGPT translation requires an owner-protected reader.')
    return {'codex': 'Codex CLI', 'chatgpt': 'ChatGPT'}[name]


def infer(root, model, prompt, images, schema, timeout):
    if provider() == 'ChatGPT':
        from .chatgpt import infer as direct
        return direct(model, prompt, images, schema, timeout)
    contract, output = root / 'schema.json', root / 'result.json'
    contract.write_text(json.dumps(schema))
    cmd = ['codex', 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
           '--disable', 'shell_tool', '--disable', 'plugins', '-c', 'web_search="disabled"',
           '-s', 'read-only', '-m', model, '--output-schema', str(contract), '-o', str(output)]
    for path in images:
        cmd.extend(['-i', str(path.resolve(strict=True))])
    result = subprocess.run([*cmd, '-'], cwd=root, input=prompt, text=True, capture_output=True, timeout=timeout)
    if result.returncode or not output.is_file():
        raise RuntimeError(f'{model} call failed: ' + result.stderr[-1200:])
    return json.loads(output.read_text())
