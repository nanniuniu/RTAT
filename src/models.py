"""Public model factory. All path variants are independently trainable."""
from dataclasses import replace
from gated_core import GatedRecent, configuration
from tod_core import DirectWithTimeOfDay, pack_external
from common_helpers import RecentResidualWrapper

VARIANTS = ('backbone', 'tod', 'fixed_residual', 'gated_residual', 'full')
COUNTS = dict(backbone=64719968, tod=64812128, fixed_residual=64719968,
              gated_residual=64720146, full=64812306)
COMPONENTS = dict(lag_bias='use_lag_bias', temporal_mask='use_temporal_mask',
                  shifted_windows='use_shifted_windows', external='use_external')


def build(variant, scaler, minutes=30, disabled=()):
    if variant not in VARIANTS or any(x not in COMPONENTS for x in disabled):
        raise ValueError('Unknown model or component')
    if 'external' in disabled and variant in ('tod', 'full'):
        raise ValueError('Time-of-day conditioning requires the external branch; use gated_residual or backbone.')
    cfg = replace(configuration(minutes), **{COMPONENTS[x]: False for x in disabled})
    model = cfg.build()
    if variant in ('tod', 'full'):
        model = DirectWithTimeOfDay(model)
    if variant in ('gated_residual', 'full'):
        model = GatedRecent(model, scaler)
    elif variant == 'fixed_residual':
        model = RecentResidualWrapper(model, scaler, True)
    if not disabled and sum(p.numel() for p in model.parameters()) != COUNTS[variant]:
        raise RuntimeError('Model parameter count changed')
    return model


def external_for(variant, external, slots):
    return pack_external(external, slots) if variant in ('tod', 'full') else external
