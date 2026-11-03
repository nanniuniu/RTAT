"""Run a saved model on raw-count observations and unstandardized external features."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
import torch
from run_reproducible import InputScaler
import models


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','input-scaler','external-scaler','flow','external','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--slots',type=Path,help='Integer .npy vector of reconstructed half-hour slot IDs, 0..47')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=checkpoint['config'];meta=cfg.get('plan',cfg)
    variant=meta.get('variant',meta.get('job',{}).get('variant','full'))
    if variant=='gated_tod':variant='full'
    with np.load(args.input_scaler,allow_pickle=False) as z:scaler=InputScaler(z['minimum'],z['maximum'])
    stats=json.loads(args.external_scaler.read_text())
    model=models.build(variant,scaler,meta.get('minutes',30),meta.get('disabled_components',[]))
    # Preserve the scale used when the checkpoint was created.
    for name,value in model.named_buffers():
        if name.startswith('anchor_') and not torch.equal(value,checkpoint['model_state'][name]):
            raise ValueError('Input scaler does not match checkpoint anchor buffers')
    model.load_state_dict(checkpoint['model_state'],strict=True)
    del checkpoint
    model.to(args.device).eval()
    x=np.load(args.flow,mmap_mode='r',allow_pickle=False)
    ext=np.asarray(np.load(args.external,allow_pickle=False),dtype=np.float32).copy()
    if x.ndim!=5 or x.shape[1:]!=(12,2,12,16) or ext.shape!=(len(x),38) or not len(x):
        raise ValueError('Expected aligned X[N,12,2,12,16] and external[N,38]')
    if not np.isfinite(ext).all():raise ValueError('Nonfinite external features')
    ext[:,:2]=(ext[:,:2]-np.asarray(stats['mean']))/np.asarray(stats['std'])
    if variant in ('tod','full'):
        if args.slots is None:raise ValueError('--slots is required for time-of-day conditioning')
        ext=models.external_for(variant,ext,np.load(args.slots,allow_pickle=False))
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(x),args.batch_size):
            flow=scaler.transform(x[start:start+args.batch_size])
            if not np.isfinite(flow).all():raise ValueError('Nonfinite flow input')
            pred=model(torch.from_numpy(flow).to(args.device),torch.from_numpy(ext[start:start+args.batch_size]).to(args.device))
            if not torch.isfinite(pred).all():raise ValueError('Nonfinite predictions')
            predictions.append(pred.cpu().numpy())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:np.save(f,np.concatenate(predictions),allow_pickle=False)
    print('Saved raw-count predictions:',args.output)


if __name__=='__main__':main()
