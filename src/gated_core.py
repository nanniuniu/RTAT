"""Identical scheme3 gate, with the existing per-granularity backbone configuration."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from base_runner import ModelConfiguration

PARAMETERS=64720146

def configuration(minutes):
    if minutes not in (30,60,90):raise ValueError('Unapproved granularity')
    drop=.4 if minutes==30 else .2
    return ModelConfiguration(embed_dim=128,depths=(2,18,2,2),heads=(2,4,8,16),
        patch_size=(3,2,2),window_size=(2,2,2),grid_shape=(12,16),
        dropout=drop,attention_dropout=drop,stochastic_depth=.3,
        use_lag_bias=True,use_temporal_mask=True,use_shifted_windows=True,
        use_external=True,external_dim=38)

class GatedRecent(nn.Module):
    """Same RTAT correction as the completed recent-residual run, with gated anchor.

    g=2*sigmoid(z) in (0,2), initialized to one. The gate reads only the four
    observed recent frames, with shared 1x1 convolutions (8->16->2).
    """
    def __init__(self,backbone,scaler,enabled=True):
        super().__init__()
        self.backbone=backbone
        self.enabled=bool(enabled)
        minimum=np.asarray(scaler.minimum,dtype=np.float32)
        scale=np.asarray(scaler.scale,dtype=np.float32)
        if minimum.shape!=scale.shape or minimum.shape!=(12,2,12,16):raise ValueError('Scaler shape changed')
        self.register_buffer('anchor_minimum',torch.from_numpy(minimum[11].copy()))
        self.register_buffer('anchor_scale',torch.from_numpy(scale[11].copy()))
        with torch.random.fork_rng(devices=[]):
            self.gate=nn.Sequential(nn.Conv2d(8,16,1),nn.GELU(),nn.Conv2d(16,2,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def anchor(self,flow):
        return (flow[:,11].float()+1)*.5*self.anchor_scale+self.anchor_minimum

    def gate_value(self,flow):
        # Float32 avoids overflow of the anchor multiplier under mixed precision.
        with torch.autocast(device_type=flow.device.type,enabled=False):
            recent=flow[:,8:12].float().reshape(flow.shape[0],8,12,16)
            return 2*torch.sigmoid(self.gate(recent))

    def forward(self,flow,external):
        correction=self.backbone(flow,external).float()
        g=self.gate_value(flow) if self.enabled else 1.
        return correction+g*self.anchor(flow)

def build(plan,scaler):
    model=GatedRecent(configuration(plan['minutes']).build(),scaler)
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise RuntimeError('Parameter count changed')
    return model
