"""Run fresh, independent dataset/interval/variant/seed jobs sequentially."""
from pathlib import Path
import argparse
import itertools
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
VARIANTS=('backbone','tod','fixed_residual','gated_residual','full')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--dataset',choices=('nyctaxi','nycbike'),required=True)
    p.add_argument('--minutes',type=int,nargs='+',choices=(30,60,90),required=True)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['full'])
    p.add_argument('--seeds',nargs='+',type=int,choices=(0,1,2),default=[0,1,2])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    for key in ('minutes','variants','seeds'):
        values=getattr(args,key)
        if len(values)!=len(set(values)):p.error('Duplicate '+key)
    if args.workers<0:p.error('--workers must be nonnegative')
    commands=[]
    for minutes,variant,seed in itertools.product(args.minutes,args.variants,args.seeds):
        output=args.output_root.resolve()/f'{args.dataset}_{minutes}_{variant}_seed{seed}'
        if output.exists():p.error('Existing run directory: '+str(output))
        command=[sys.executable,str(ROOT/'train.py'),'--data-root',str(args.data_root.resolve()),
                 '--dataset',args.dataset,'--minutes',str(minutes),'--variant',variant,
                 '--seed',str(seed),'--output',str(output),'--device',args.device,
                 '--workers',str(args.workers)]
        commands.append(command)
    print(json.dumps({'mode':'preview' if args.dry_run else 'train','jobs':len(commands)}),flush=True)
    for command in commands:
        print(json.dumps(command),flush=True)
        if not args.dry_run:subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
