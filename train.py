"""Train one RTAT configuration; validate each epoch and test once after selection."""
from pathlib import Path
import argparse
import json
import math
import os
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
import torch
import base_runner as reference
from common_helpers import evaluate_detailed
from review_io import profile, validated_data, training_data, heldout_data, write_json, digest, transfer_parameters
import models


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=('nyctaxi','nycbike'),required=True)
    p.add_argument('--minutes',type=int,choices=(30,60,90),required=True)
    p.add_argument('--variant',choices=models.VARIANTS,default='full')
    p.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--init-from',type=Path,help='30-minute checkpoint; transfer learned parameters only')
    p.add_argument('--learning-rate',type=float,help='Explicit override, recorded in config.json')
    p.add_argument('--disable-component',action='append',choices=tuple(models.COMPONENTS),default=[])
    p.add_argument('--validate-only',action='store_true',help='Validate full data hashes, shapes and calendars; no training')
    return p.parse_args()


def train(args,cfg):
    if args.output is None:
        raise ValueError('--output is required for training')
    if args.workers < 0 or cfg['learning_rate'] <= 0:
        raise ValueError('Invalid workers or learning rate')
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu for functional checks')
        torch.cuda.set_device(device)
    folder=args.output.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'status.json',dict(status='preparing',pid=os.getpid()))
    torch.set_num_threads(cfg['threads'])
    train_ds,val_ds,scaler,stats,hashes=training_data(args.data_root,cfg)
    cfg['data_sha256']=hashes
    cfg['external_scaler']=stats
    cfg['test_loader_during_training']=False
    cfg['initialization']='direct' if args.init_from is None else 'fine_tune_from_30min'
    cfg['source_code_sha256']={str(p.relative_to(Path(__file__).parent)):digest(p)
        for p in sorted(Path(__file__).parent.rglob('*.py')) if '__pycache__' not in p.parts}
    reference.set_seed(cfg['seed'])
    def loader(ds,shuffle=False):
        return reference.make_loader(ds,cfg['batch_size'],shuffle=shuffle,seed=cfg['seed'],
                                     workers=cfg['workers'],limit_samples=None,prefetch_factor=2)
    tr,va=loader(train_ds,True),loader(val_ds)
    model=models.build(cfg['variant'],scaler,cfg['minutes'],cfg['disabled_components'])
    if args.init_from:
        if cfg['minutes']==30:raise ValueError('--init-from is for 60/90-minute fine-tuning')
        source=torch.load(args.init_from,map_location='cpu',weights_only=True)
        cfg['transfer']=transfer_parameters(model,source,cfg['dataset'])
        cfg['transfer']['checkpoint_sha256']=digest(args.init_from)
        del source
    model.to(device)
    cfg['parameter_count']=sum(p.numel() for p in model.parameters())
    write_json(folder/'config.json',cfg)
    write_json(folder/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        workers=cfg['workers'],threads=cfg['threads']))
    write_json(folder/'external_scaler.json',stats)
    np.savez(folder/'input_scaler.npz',minimum=scaler.minimum,maximum=scaler.maximum)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=device.type=='cuda')
    steps,warmup=cfg['scheduler_epochs']*len(tr),cfg['warmup_epochs']*len(tr)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:reference.warmup_cosine_factor(min(k,steps),
        total_steps=steps,warmup_steps=warmup,start_factor=.005,min_factor=.005))
    amp=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'],enabled=device.type=='cuda')
    best=float('inf'); patience=0
    for epoch in range(1,cfg['max_epochs']+1):
        model.train(); sse=0.; values=processed=skips=overflow=0; started=time.perf_counter()
        for batch in tr:
            flow,ext,target=reference.unpack_batch(batch,device,True)
            processed+=len(target);opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                loss=torch.nn.functional.mse_loss(model(flow,ext).float(),target.float())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            amp.scale(loss).backward();amp.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'])
            if not torch.isfinite(norm):
                if device.type!='cuda':raise RuntimeError('Nonfinite CPU gradients')
                amp.step(opt);amp.update();skips+=1;overflow+=1
                if overflow>=16:raise RuntimeError('Repeated AMP gradient overflow')
                continue
            overflow=0;amp.step(opt);amp.update();sched.step()
            sse+=float(loss.detach())*target.numel();values+=target.numel()
        if processed!=cfg['train_stop'] or not values:raise RuntimeError('Incomplete training epoch')
        metric,_,_=reference.evaluate_loader(model,va,device,use_external=True,collect=False)
        if not all(math.isfinite(metric[k]) for k in ('mse','rmse','mae')):raise RuntimeError('Invalid validation metrics')
        row=dict(epoch=epoch,train_mse=sse/values,validation_mse=metric['mse'],validation_rmse=metric['rmse'],
                 validation_mae=metric['mae'],processed_train_samples=processed,skipped_amp_updates=skips,
                 learning_rate=opt.param_groups[0]['lr'],epoch_seconds=time.perf_counter()-started)
        with (folder/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        if metric['mse']<best:
            best=metric['mse'];patience=0
            tmp=folder/'best_checkpoint.partial'
            torch.save(dict(model_state=model.state_dict(),config=cfg,epoch=epoch,validation_metrics=metric,
                            model_only_not_exact_resume=True),tmp)
            os.replace(tmp,folder/'best_checkpoint.pt')
        else:patience+=1
        write_json(folder/'status.json',dict(status='training',last_epoch=row,patience=patience))
        print(json.dumps(row),flush=True)
        if patience>=cfg['patience']:break
    state=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model_state'],strict=True)
    best_epoch=state['epoch'];expected=state['validation_metrics'];del state
    val_metric,arrays=evaluate_detailed(model,va,device,collect=True)
    for k in ('mse','rmse','mae'):
        if not math.isclose(val_metric[k],expected[k],rel_tol=2e-6,abs_tol=2e-6):raise RuntimeError('Reloaded validation differs')
    np.savez_compressed(folder/'validation_predictions.npz',prediction=arrays[0],target=arrays[1])
    selection=dict(best_epoch=best_epoch,validation_verified=True,checkpoint_sha256=digest(folder/'best_checkpoint.pt'))
    write_json(folder/'validation_selection.json',selection)
    with (folder/'test_evaluation_started.json').open('x',encoding='utf-8') as f:
        json.dump(dict(test_evaluation_count=1,selection=selection),f)
    test=heldout_data(args.data_root,cfg,scaler,stats)
    metric,arrays=evaluate_detailed(model,loader(test),device,collect=True)
    np.savez_compressed(folder/'test_predictions.npz',prediction=arrays[0],target=arrays[1])
    done=dict(status='completed',dataset=cfg['dataset'],minutes=cfg['minutes'],variant=cfg['variant'],seed=cfg['seed'],
        disabled_components=cfg['disabled_components'],initialization=cfg['initialization'],epochs_executed=epoch,
        best_epoch=best_epoch,test_evaluation_count=1,validation_metrics=val_metric,test_metrics=metric,
        checkpoint_sha256=selection['checkpoint_sha256'],prediction_sha256=digest(folder/'test_predictions.npz'))
    write_json(folder/'completion.json',done);write_json(folder/'status.json',done)
    print(json.dumps(done),flush=True)


def main():
    args=parse_args();cfg=profile(args.dataset,args.minutes)
    cfg.update(variant=args.variant,seed=args.seed,workers=args.workers,disabled_components=sorted(set(args.disable_component)))
    if args.learning_rate is not None:cfg['learning_rate']=args.learning_rate
    if args.validate_only:
        *_,hashes=validated_data(args.data_root,cfg)
        print(json.dumps(dict(status='validated',dataset=args.dataset,minutes=args.minutes,hashes=hashes)));return
    new_output=args.output is not None and not args.output.exists()
    try:train(args,cfg)
    except BaseException as error:
        # Never overwrite a completed run or mark an existing directory as a new failure.
        if new_output and args.output is not None and (args.output/'status.json').exists():
            write_json(args.output/'status.json',dict(status='failed',error=type(error).__name__+': '+str(error)))
        print(f'{type(error).__name__}: {error}',file=sys.stderr)
        raise


if __name__=='__main__':main()
