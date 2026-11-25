"""Read approved personal Modal artifacts without starting any remote functions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',default='alex_261124_ref6x256_192325')
    parser.add_argument('--artifacts',action='store_true')
    args=parser.parse_args()
    from modal import config
    assert os.environ.get('MODAL_PROFILE')==config._profile=='alexander-tian'
    volume=modal.Volume.from_name('project182-artifacts',create_if_missing=False,version=2)
    control=ROOT/'outputs/forecast_reference_train_control'/args.batch
    plan=json.loads((control/'resolved_plan.json').read_text())
    def fetch(c):
        name=c['run_name'];out=control/'artifacts'/name;out.mkdir(parents=True,exist_ok=True)
        def read(filename,optional=False):
            try: raw=b''.join(volume.read_file(f'forecast_reference_train/{name}/{filename}'))
            except Exception as exc:
                if optional and isinstance(exc,(FileNotFoundError,modal.exception.NotFoundError)):return None
                raise
            temporary=out/(filename+'.download');temporary.write_bytes(raw);temporary.replace(out/filename)
            return raw
        raw=read('manifest.json',optional=True)
        if raw is None:return {'run':name,'status':'queued_or_starting','step':0}
        manifest=json.loads(raw)
        if args.artifacts:
            for filename in ['training.json','reference.npz','resolved_backbone.json']+list(c['source_sha256']):read(filename,optional=True)
            for step in c['training']['checkpoint_steps']:
                for kind,ext in [('metrics','json'),('predictions','npz')]:read(f'{kind}_step_{step}.{ext}',optional=True)
            read('precision_audit.json',optional=True)
            read('quick_validation.json',optional=True)
            for filename,sha in c['source_sha256'].items():
                assert hashlib.sha256((out/filename).read_bytes()).hexdigest()==sha
        return dict(run=name,status=manifest['status'],step=manifest['completed_steps'],
                    training_seconds=round(manifest.get('training_elapsed_seconds',0),1),
                    evaluations={s:r['status'] for s,r in manifest.get('evaluations',{}).items()},
                    error=manifest.get('error'),precision_audit=manifest.get('precision_audit',{}).get('status'))
    with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(fetch,plan['configs']))
    print(json.dumps(rows,indent=2))
    (control/'latest_status.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
