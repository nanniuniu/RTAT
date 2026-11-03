"""Portable configuration, validated data access, and weight-only transfer."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
from external_protocol import load_external_bundle, standardize_external
from tod_core import read_time_slots
from run_reproducible import InputScaler, fit_input_scaler
from common_helpers import BoundedDataset
import models

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def profile(dataset, minutes):
    if dataset not in ('nyctaxi','nycbike') or minutes not in (30,60,90):
        raise ValueError('Unsupported dataset/interval')
    return json.loads((ROOT/f'configs/{dataset}_{minutes}.json').read_text(encoding='utf-8'))


def paths(data_root, cfg):
    base = Path(data_root)
    name, m = cfg['dataset'], cfg['minutes']
    return dict(x=base/f'flow/data_{name}/data_x_{m}.npy',
                y=base/f'flow/data_{name}/data_y_{m}.npy',
                external=base/f'external/data_{name}/external_{m}.npy',
                metadata=base/f'external/data_{name}/external_{m}.json',
                calendar=base/f'external/data_{name}/timestamps_{m}.csv')


def validated_data(data_root, cfg):
    pp = paths(data_root, cfg)
    x, y = [np.load(pp[k], mmap_mode='r', allow_pickle=False) for k in ('x','y')]
    n = cfg['samples']
    if x.shape != (n,12,2,12,16) or y.shape != (n,2,12,16):
        raise ValueError('Flow array shapes do not match the profile')
    external, meta = load_external_bundle(pp['metadata'], dataset=cfg['dataset'], minutes=cfg['minutes'],
        sample_count=n, x_path=pp['x'], y_path=pp['y'])
    if digest(pp['external']) != meta['feature_sha256']:
        raise ValueError('Unexpected external feature file')
    slots = read_time_slots(pp['calendar'], start=0, stop=n, expected_sha256=meta['timestamp_sha256'],
        dataset=cfg['dataset'], minutes=cfg['minutes'], sample_count=n)
    return x, y, external, slots, {k:meta[k] for k in ('x_sha256','y_sha256','feature_sha256','timestamp_sha256')}


def training_data(data_root, cfg):
    x,y,external,slots,hashes = validated_data(data_root, cfg)
    stop, vstop = cfg['train_stop'], cfg['validation_stop']
    scaler = fit_input_scaler(x[:vstop], stop)
    ext, stats = standardize_external(external[:vstop], stop)
    ext = models.external_for(cfg['variant'], ext, slots[:vstop])
    return (BoundedDataset(x[:vstop],y[:vstop],ext,0,stop,scaler),
            BoundedDataset(x[:vstop],y[:vstop],ext,stop,vstop,scaler), scaler, stats, hashes)


def heldout_data(data_root, cfg, scaler, stats):
    pp = paths(data_root,cfg)
    start, stop = cfg['validation_stop'],cfg['samples']
    # Files were hashed before training. Recheck after selection to catch replacement during a run.
    for key, field in (('x','x_sha256'),('y','y_sha256'),('external','feature_sha256'),('calendar','timestamp_sha256')):
        if digest(pp[key]) != cfg['data_sha256'][field]:
            raise ValueError('Data changed during training: '+key)
    x,y=[np.load(pp[k],mmap_mode='r',allow_pickle=False)[start:stop] for k in ('x','y')]
    ext=np.load(pp['external'],allow_pickle=False)[start:stop].copy()
    ext[:,:2]=(ext[:,:2]-np.asarray(stats['mean']))/np.asarray(stats['std'])
    slots=read_time_slots(pp['calendar'],start=start,stop=stop,expected_sha256=cfg['data_sha256']['timestamp_sha256'],
        dataset=cfg['dataset'],minutes=cfg['minutes'],sample_count=stop)
    ext=models.external_for(cfg['variant'],ext,slots)
    return BoundedDataset(x,y,ext,0,stop-start,scaler)


def transfer_parameters(model, state, target_dataset):
    cfg=state.get('config',{})
    p=cfg.get('plan',cfg)
    if p.get('dataset') != target_dataset or p.get('minutes') != 30:
        raise ValueError('Initialization must come from a 30-minute run of the same dataset')
    source=state['model_state']; target=model.state_dict()
    names=dict(model.named_parameters())
    if set(source) != set(target):
        raise ValueError('Source and target model variants differ')
    before={n:t.detach().clone() for n,t in model.named_buffers()}
    for name in names:
        if source[name].shape != target[name].shape or not torch.isfinite(source[name]).all():
            raise ValueError('Invalid transfer parameter: '+name)
        target[name]=source[name]
    model.load_state_dict(target,strict=True)
    if any(not torch.equal(t,before[n]) for n,t in model.named_buffers()):
        raise RuntimeError('Target scaling buffers changed')
    return dict(source_dataset=p['dataset'],source_minutes=30,source_seed=p.get('seed'),
        source_epoch=state['epoch'],parameter_tensors=len(names),target_buffers_preserved=True,
        optimizer_state_loaded=False)
