"""Associate buffered native timing logs by invocation ID, never text position."""
import argparse
import subprocess
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'examples/a2a3/tensormap_and_ringbuffer/benchmark_streaming_action'
sys.path.insert(0,str(TOOLS))
from run_interleaved import MODES,read_device


def recover(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    expected=manifest['schedule']
    host=(folder/'host.log').read_text()
    if json.loads((folder/'exit.json').read_text())['returncode'] != 0 or '[qwen] PASSED' not in host:
        raise ValueError('worker did not complete successfully')
    begins=[json.loads(x) for x in re.findall(r'^STREAM_SAMPLE_BEGIN (.+)$',host,re.M)]
    ends=[int(x) for x in re.findall(r'^STREAM_SAMPLE_END (\d+)$',host,re.M)]
    if begins!=expected or ends!=list(range(len(expected))):
        raise ValueError('missing/changed sample markers')
    values={}
    for line in host.splitlines():
        if 'name=chip.run.runner_run.device_wall ' not in line:
            continue
        inv=int(re.search(r'\binv=(\d+)\b',line)[1])
        duration=int(re.search(r'\bdur=(\d+)\b',line)[1])/1e6
        if inv in values:
            raise ValueError('duplicate native timing invocation')
        values[inv]=duration
    if set(values)!=set(range(1,len(expected)+1)):
        raise ValueError('native invocation coverage mismatch')
    completed=[int(x) for x in re.findall(r'\binv=(\d+)\b[^\n]*\bname=chip\.run ts=',host)]
    if completed!=list(range(1,len(expected)+1)):
        raise ValueError('complete native chip.run invocation sequence mismatch')
    actual=read_device(folder)
    if len(actual)!=len(expected) or [x['mode'] for x in actual] != [MODES[x['mode']] for x in expected]:
        raise ValueError('actual device mode sequence mismatch')
    spec=importlib.util.spec_from_file_location('frozen_source_verifier',ROOT/'examples/a2a3/tensormap_and_ringbuffer/benchmark_qkv/run_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.fingerprint()!=manifest['fingerprints']:
        raise ValueError('runtime changed after collection')
    rows=[]
    for index,(label,counters) in enumerate(zip(expected,actual)):
        if counters['mode'] and (counters['eligible']!=counters['eligible_claimed'] or counters['dropped']):
            raise ValueError('eligible block coverage mismatch')
        rows.append({**label,'device_wall_ms':values[index+1],'native_invocation_id':index+1,'counters':counters})
    hashes=dict(re.findall(r'func_id=(\d+).*?code_fnv1a64=(0x[0-9a-f]+)',host))
    if len(hashes)!=37:
        raise ValueError('loaded compute fingerprint coverage')
    (folder/'kernel-hashes.json').write_text(json.dumps(hashes,indent=2))
    (folder/'samples.json').write_text(json.dumps(rows,indent=2))
    (folder/'parser-recovery.json').write_text(json.dumps({'method':'native invocation ID = ordinal + 1, independently validated by complete chip.run invocation sequence','samples':len(rows),'native_timings':len(values),'raw_logs_modified':False,'collection_harness_modified':False,'reason':'buffered native stdout can cross Python begin/end text boundaries; original positional parser is not used for final timing association'},indent=2))
    print('VALIDATED native-ID timing association',folder,len(rows),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='collect':
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--output',type=Path,required=True)
        parser.add_argument('--seq-len',type=int,default=3500)
        args=parser.parse_args(sys.argv[2:])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        log_path=args.output.parent/(args.output.name+'.collector.log')
        with log_path.open('w') as log:
            result=subprocess.run([sys.executable,str(TOOLS/'run_interleaved.py'),'formal','--output',str(args.output),'--seq-len',str(args.seq_len)],cwd=ROOT,check=False,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode and 'ValueError: sample/timing mismatch' not in log_path.read_text():
            raise RuntimeError('collection failed beyond the diagnosed positional parser issue')
        if (args.output/'samples.json').exists():
            (args.output/'samples-positional-parser.json').write_bytes((args.output/'samples.json').read_bytes())
        recover(args.output.resolve())
    else:
        recover(Path(sys.argv[1]))
