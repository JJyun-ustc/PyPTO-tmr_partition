"""Resume the frozen collection after repairing only offline log association."""
import hashlib,json,subprocess,sys
from pathlib import Path
from recover_formal_parser import recover
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TOOLS=ROOT/'examples/a2a3/tensormap_and_ringbuffer/benchmark_streaming_action'
freeze=json.loads((OUT/'formal-freeze.json').read_text())
for seq in (2250,5000):
    dest=OUT/f'final-formal-{seq}'
    cmd=[sys.executable,str(TOOLS/'run_interleaved.py'),'formal','--output',str(dest),'--seq-len',str(seq)]
    print('START formal',seq,flush=True)
    with (OUT/f'final-formal-{seq}.log').open('w') as log:
        run=subprocess.run(cmd,cwd=ROOT,check=False,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode and 'ValueError: sample/timing mismatch' not in (OUT/f'final-formal-{seq}.log').read_text():
        raise RuntimeError('collector failed beyond the diagnosed offline parser issue')
    if (dest/'samples.json').exists():
        (dest/'samples-positional-parser.json').write_bytes((dest/'samples.json').read_bytes())
    recover(dest)
    print('PASS formal',seq,flush=True)
for path,digest in freeze['harness_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
with (OUT/'formal-analysis.log').open('w') as log:
    subprocess.run([sys.executable,str(TOOLS/'summarize.py')]+[str(OUT/f'final-formal-{s}') for s in (3500,2250,5000)],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
print('PASS all frozen measurements, invocation checks and summaries',flush=True)
