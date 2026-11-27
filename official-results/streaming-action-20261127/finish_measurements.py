"""Continue serial board collection only after all golden and pilot jobs pass."""
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TOOLS = ROOT / 'examples/a2a3/tensormap_and_ringbuffer/benchmark_streaming_action'
validation = OUT / 'final-validation.log'
end = time.monotonic() + 1800
while not validation.exists() or 'PASS pilot 3500' not in validation.read_text():
    if time.monotonic() > end or (validation.exists() and 'Traceback' in validation.read_text()):
        raise RuntimeError('Golden/pilot validation did not pass; no formal samples collected')
    time.sleep(2)

pilot = json.loads((OUT / 'final-pilot-3500/samples.json').read_text())
cv = {}
for mode in ('shared', 'structure', 'C_guarded', 'D_guarded'):
    values = [r['device_wall_ms'] for r in pilot if r['mode'] == mode and not r['warmup']]
    if len(values) != 5:
        raise ValueError('pilot sample count')
    cv[mode] = statistics.stdev(values) / statistics.mean(values)
if max(cv.values()) > .10:
    raise RuntimeError(f'Pilot instability; sample-count freeze needs review: {cv}')
freeze = {'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'periods':12,'warmup_per_policy_per_period':1,'ordinary_per_policy_per_period':5,
          'reason':'predeclared minimum 12 periods retained after independent pilot stability check; no formal data inspected',
          'pilot_coefficient_of_variation':cv,
          'bootstrap':'12 paired periods, 10000 resamples, seed1234',
          'harness_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TOOLS.rglob('*.py'))},
          'admission_sha256':hashlib.sha256((OUT/'calibration-admission.json').read_bytes()).hexdigest(),
          'source_and_binaries':json.loads((OUT/'final-pilot-3500/manifest.json').read_text())['fingerprints']}
(OUT/'formal-freeze.json').write_text(json.dumps(freeze,indent=2))
commands=[]
for phase,seq in [('diagnose',3500),('formal',3500),('formal',2250),('formal',5000)]:
    dest=OUT/f'final-{phase}-{seq}'
    cmd=[sys.executable,str(TOOLS/'run_interleaved.py'),phase,'--output',str(dest),'--seq-len',str(seq)]
    commands.append(cmd)
    (OUT/'measurement-commands.json').write_text(json.dumps(commands,indent=2))
    print('START',phase,seq,flush=True)
    with (OUT/f'final-{phase}-{seq}.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
    print('PASS',phase,seq,flush=True)
for p,h in freeze['harness_sha256'].items():
    if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:
        raise RuntimeError('Python harness changed during formal measurement')
cmd=[sys.executable,str(TOOLS/'summarize.py')]+[str(OUT/f'final-formal-{s}') for s in (3500,2250,5000)]
with (OUT/'formal-analysis.log').open('w') as log:
    subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
print('PASS all frozen measurements and summaries',flush=True)
