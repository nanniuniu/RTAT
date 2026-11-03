"""Arrange existing benchmark arrays and external features in the portable layout."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from review_io import ROOT, digest, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow-root',type=Path,required=True,help='Directory containing data_nyctaxi and data_nycbike')
    p.add_argument('--external-root',type=Path,required=True,help='Directory containing rebuilt external data_nyctaxi/data_nycbike')
    p.add_argument('--output-root',type=Path,required=True,help='A new directory')
    p.add_argument('--dataset',choices=('nyctaxi','nycbike','all'),default='all')
    p.add_argument('--minutes',type=int,choices=(30,60,90),help='Omit to prepare all intervals')
    p.add_argument('--link-flow',action='store_true',help='Hard-link flow arrays on the same volume; treat linked data as read-only')
    args=p.parse_args()
    profiles=json.loads((ROOT/'data_metadata/expected_data_sha256.json').read_text())
    selected=[x for x in profiles if (args.dataset=='all' or x['dataset']==args.dataset) and (args.minutes is None or x['minutes']==args.minutes)]
    if args.output_root.exists():raise FileExistsError('Choose a new output directory')
    jobs=[]
    for cfg in selected:
        for rel,expected in cfg['files'].items():
            kind,sub=rel.split('/',1)
            src=(args.flow_root if kind=='flow' else args.external_root)/sub
            if not src.is_file() or digest(src)!=expected:raise ValueError('Missing or mismatched data: '+str(src))
            jobs.append((src,args.output_root/rel,kind))
    args.output_root.mkdir(parents=True)
    for src,dst,kind in jobs:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if args.link_flow and kind=='flow':os.link(src,dst)
        else:shutil.copyfile(src,dst)
    for cfg in selected:
        rel=f"data_{cfg['dataset']}/external_{cfg['minutes']}.json"
        shutil.copyfile(ROOT/'data_metadata'/rel,args.output_root/'external'/rel)
    write_json(args.output_root/'prepared.json',dict(profiles=[dict(dataset=x['dataset'],minutes=x['minutes']) for x in selected],
        hashes_verified=True,flow_storage='hardlinks_read_only' if args.link_flow else 'copies'))
    print('Prepared',len(selected),'profiles in',args.output_root)


if __name__=='__main__':main()
