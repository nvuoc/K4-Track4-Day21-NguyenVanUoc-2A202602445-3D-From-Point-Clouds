"""Check submission including untracked deliverables, notebook code, report images."""
import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf8")
    notebook = json.loads((ROOT/'src/NguyenVanUoc_2A202602445_Colab.ipynb').read_text(encoding='utf8'))
    assert notebook['nbformat'] == 4
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']),filename=f'notebook cell {i}')
    namespace = {}
    source_cell = next(c for c in notebook['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('implementation = '))
    source = ''.join(source_cell['source'])
    assignment = ast.parse(source).body[0]
    payload = ast.literal_eval(assignment.value)
    for relative, content in payload.items():
        assert (ROOT/relative).read_text(encoding='utf8') == content.replace('\r\n', '\n'), f'Notebook payload stale: {relative}'
    report = ROOT/'report/REPORT.md'
    for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report.read_text(encoding='utf8')):
        assert (report.parent/link).is_file(), f'Missing report image: {link}'
    all_files = subprocess.run(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,
                               capture_output=True,text=True,check=True).stdout.splitlines()
    spec = importlib.util.spec_from_file_location('submission_checker',ROOT/'tools/check_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tracked_files = lambda: [ROOT/f for f in dict.fromkeys(all_files)]
    code = module.main()
    if code:
        raise SystemExit(code)
    print('PASS: notebook syntax/payload, report images, tracked + untracked deliverables')

if __name__=='__main__':
    main()
