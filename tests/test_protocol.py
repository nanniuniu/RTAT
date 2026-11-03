"""CPU-only checks of release behavior; no datasets or CUDA training required."""
from pathlib import Path
import gc
import argparse
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from data_protocol import chronological_split
from run_reproducible import InputScaler,fit_input_scaler
from external_protocol import standardize_external
from review_io import profile,transfer_parameters
from rebuild_external_features import lags
from tod_core import pack_external
from summarize import summarize
import models
import train as train_entry
from common_helpers import BoundedDataset


class ProtocolTests(unittest.TestCase):
    def test_six_splits_and_lags(self):
        for ds in ('nyctaxi','nycbike'):
            for minutes in (30,60,90):
                cfg=profile(ds,minutes);s=chronological_split(cfg['samples'],minutes)
                self.assertEqual(s.train.stop,cfg['train_stop'])
                self.assertEqual(s.validation.stop,cfg['validation_stop'])
                self.assertEqual(s.test.stop,cfg['samples'])
                self.assertEqual(lags(minutes)[-4:],[4,3,2,1])
                self.assertEqual(lags(minutes)[0]*minutes,28*1440)

    def test_train_only_scaling_and_unscaled_calendar(self):
        x=np.zeros((3,12,2,12,16),np.float32);x[1]=2;x[2]=1000
        s=fit_input_scaler(x,2)
        self.assertTrue(np.all(s.maximum==2))
        self.assertTrue(np.all(s.transform(x[2])>1))
        ext=np.zeros((3,38),np.float32);ext[1,:2]=2;ext[2,:2]=1000
        scaled,stats=standardize_external(ext,2)
        self.assertEqual(stats['mean'],[1.,1.]);self.assertEqual(stats['std'],[1.,1.])
        packed=pack_external(scaled,np.array([0,20,47]))
        np.testing.assert_array_equal(packed[:,-1],[0,20,47])
        with self.assertRaises(ValueError):pack_external(scaled,np.array([0,20,48]))

    def test_partial_summary_never_claims_three_seeds(self):
        rows=[dict(dataset='nyctaxi',minutes=30,variant='full',seed=s,status='completed',
            test_metrics=dict(rmse=v,mae=v)) for s,v in ((0,2.),(1,4.))]
        group=summarize(rows)[0]
        self.assertFalse(group['complete']);self.assertEqual(group['missing_seeds'],[2])
        self.assertAlmostEqual(group['rmse_sample_sd'],2**.5)
        with self.assertRaises(ValueError):summarize(rows+[rows[0]])


class ModelTests(unittest.TestCase):
    def test_all_path_variants_and_full_gradients(self):
        torch.set_num_threads(1)
        s=InputScaler(np.zeros((12,2,12,16),np.float32),np.ones((12,2,12,16),np.float32)*10)
        flow=torch.zeros((1,12,2,12,16));ext=torch.zeros(1,38)
        for variant in models.VARIANTS:
            torch.manual_seed(123)
            model=models.build(variant,s).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),models.COUNTS[variant])
            e=torch.cat((ext,torch.tensor([[7.]])),dim=1) if variant in ('tod','full') else ext
            with torch.no_grad():out=model(flow,e)
            self.assertEqual(tuple(out.shape),(1,2,12,16));self.assertTrue(torch.isfinite(out).all())
            if variant in ('gated_residual','full'):
                torch.testing.assert_close(model.gate_value(flow),torch.ones((1,2,12,16)))
                torch.testing.assert_close(model.anchor(flow),torch.full((1,2,12,16),5.))
            if variant=='full':
                self.assertTrue(all(torch.count_nonzero(m.tod_embedding.weight)==0 for m in model.backbone.backbone.external_embeddings))
                model(flow,e).square().mean().backward()
                self.assertTrue(torch.isfinite(model.gate[-1].weight.grad).all())
                self.assertGreater(torch.count_nonzero(model.gate[-1].weight.grad).item(),0)
                grad=model.backbone.backbone.external_embeddings[0].tod_embedding.weight.grad
                self.assertTrue(torch.isfinite(grad).all());self.assertGreater(torch.count_nonzero(grad).item(),0)
            del model;gc.collect()

    def test_transfer_preserves_target_buffers(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2));self.register_buffer('anchor_scale',torch.ones(2)*8)
        model=Tiny()
        state=dict(config=dict(dataset='nyctaxi',minutes=30,seed=0),epoch=4,
            model_state=dict(weight=torch.ones(2)*3,anchor_scale=torch.ones(2)*99))
        transfer_parameters(model,state,'nyctaxi')
        torch.testing.assert_close(model.weight,torch.ones(2)*3)
        torch.testing.assert_close(model.anchor_scale,torch.ones(2)*8)
        with self.assertRaises(ValueError):transfer_parameters(model,state,'nycbike')


class TrainingLifecycleTests(unittest.TestCase):
    def test_selection_then_single_test_and_output_protection(self):
        # A tiny differentiable model tests orchestration independently of the full-model test above.
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.5))
            def forward(self,flow,external):
                return flow[:,11]*self.weight
        scaler=InputScaler(np.zeros((12,2,12,16),np.float32),np.ones((12,2,12,16),np.float32))
        x=np.ones((4,12,2,12,16),np.float32);y=np.ones((4,2,12,16),np.float32)*2
        ext=np.zeros((4,39),np.float32)
        ds=BoundedDataset(x,y,ext,0,2,scaler);val=BoundedDataset(x,y,ext,2,3,scaler)
        stats=dict(mean=[0.,0.],std=[1.,1.],columns=[0,1],fit_range=[0,2])
        cfg=profile('nyctaxi',30)
        cfg.update(samples=4,train_stop=2,validation_stop=3,max_epochs=1,batch_size=2,threads=1,
            variant='full',seed=0,workers=0,disabled_components=[])
        with tempfile.TemporaryDirectory(prefix='rtat-release-test-') as temp:
            root=Path(temp).resolve()
            self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            out=root/'run'
            args=argparse.Namespace(output=out,device='cpu',workers=0,data_root=root,init_from=None)
            def final_data(*_):
                self.assertTrue((out/'validation_selection.json').exists())
                self.assertTrue((out/'test_evaluation_started.json').exists())
                return BoundedDataset(x,y,ext,3,4,scaler)
            with patch.object(train_entry,'training_data',return_value=(ds,val,scaler,stats,{})), \
                 patch.object(train_entry,'heldout_data',side_effect=final_data) as heldout, \
                 patch.object(models,'build',side_effect=lambda *a:Tiny()):
                train_entry.train(args,cfg)
                self.assertEqual(heldout.call_count,1)
                done=json.loads((out/'completion.json').read_text())
                self.assertEqual(done['test_evaluation_count'],1)
                self.assertEqual(done['best_epoch'],1)
                self.assertEqual(done['status'],'completed')
                with self.assertRaises(FileExistsError):train_entry.train(args,cfg)


if __name__=='__main__':unittest.main(verbosity=2)
