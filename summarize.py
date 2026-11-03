"""Aggregate completed runs using sample SD; retain explicit missing-seed status."""
from pathlib import Path
import argparse
import json
import sys
import statistics
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from review_io import write_json


def summarize(rows):
    groups={}
    for row in rows:
        if row.get('status')!='completed':continue
        key=(row['dataset'],row['minutes'],row['variant'],tuple(row.get('disabled_components',[])),row.get('initialization','direct'))
        g=groups.setdefault(key,{})
        if row['seed'] in g:raise ValueError('Duplicate seed in a comparison group: '+str(key))
        g[row['seed']]=row
    result=[]
    for key,runs in sorted(groups.items()):
        dataset,minutes,variant,disabled,init=key
        item=dict(dataset=dataset,minutes=minutes,variant=variant,disabled_components=list(disabled),initialization=init,
            seeds=sorted(runs),missing_seeds=sorted(set((0,1,2))-set(runs)),n=len(runs),complete=set(runs)=={0,1,2})
        for metric in ('rmse','mae'):
            values=[r['test_metrics'][metric] for r in runs.values()]
            item[metric+'_mean']=statistics.mean(values)
            item[metric+'_sample_sd']=statistics.stdev(values) if len(values)>1 else None
        full=groups.get((dataset,minutes,'full',(),init),{})
        common=sorted(set(runs)&set(full))
        if variant!='full' and common:
            paired=dict(seeds=common,complete=common==[0,1,2])
            for metric in ('rmse','mae'):
                values=[runs[s]['test_metrics'][metric]-full[s]['test_metrics'][metric] for s in common]
                paired[metric+'_differences']=values
                paired[metric+'_mean_difference']=statistics.mean(values)
            item['paired_variant_minus_full']=paired
        result.append(item)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path)
    p.add_argument('--snapshot',type=Path,help='Exported path_ablation_snapshot.json; do not combine duplicate runs')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.runs_root and not args.snapshot:p.error('Provide --runs-root or --snapshot')
    if args.output.exists():raise FileExistsError(args.output)
    rows=[]
    if args.runs_root:
        for f in sorted(args.runs_root.rglob('completion.json')):rows.append(json.loads(f.read_text()))
    if args.snapshot:
        value=json.loads(args.snapshot.read_text())
        for r in value['runs']:
            if r['status']=='completed':rows.append(dict(r,dataset=value['dataset'],minutes=value['minutes'],
                test_metrics=dict(rmse=r['rmse'],mae=r['mae'])))
    result=summarize(rows)
    write_json(args.output,dict(standard_deviation='sample; ddof=1',groups=result))
    print('Wrote',len(result),'groups to',args.output)


if __name__=='__main__':main()
