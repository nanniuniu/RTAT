"""Export completed and pending path results from an existing experiment plan."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from review_io import digest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True,help='Local archived/new-run plan; the plan itself is not exported')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new snapshot output')
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    rows=[];seen=set()
    for source in ('archived','jobs'):
        for job in plan[source]:
            key=(job['variant'],job['seed'])
            if key in seen:raise ValueError('Duplicate variant/seed')
            seen.add(key)
            folder=Path(job['path']) if source=='archived' else Path(plan['root'])/'runs'/job['id']
            completion=folder/'completion.json'
            row=dict(variant=key[0],seed=key[1],status='pending')
            if completion.exists():
                done=json.loads(completion.read_text(encoding='utf-8'))
                if done['status']!='completed':raise ValueError('Invalid completion record')
                row.update(status='completed',rmse=done['test_metrics']['rmse'],mae=done['test_metrics']['mae'],
                    best_epoch=done.get('best_epoch',done.get('best_validation_epoch')),
                    origin='previously_completed' if source=='archived' else 'supplementary_run',
                    prediction_sha256=digest(folder/'test_predictions.npz'))
            rows.append(row)
    write_json(args.output,dict(dataset=plan['dataset'],minutes=plan['minutes'],expected_seeds=plan['seeds'],
        as_of=datetime.now(timezone.utc).isoformat(),runs=rows))
    print('Exported',sum(r['status']=='completed' for r in rows),'completed of',len(rows),'runs')


if __name__=='__main__':main()
