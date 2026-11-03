"""CPU model check with random weights and synthetic inputs, not a paper result."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from run_reproducible import InputScaler
import models

def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    scaler=InputScaler(np.zeros((12,2,12,16),np.float32),np.ones((12,2,12,16),np.float32)*100)
    raw=np.random.default_rng(0).uniform(0,100,(1,12,2,12,16)).astype(np.float32)
    flow=torch.from_numpy(scaler.transform(raw))
    ext=models.external_for('full',np.zeros((1,38),np.float32),np.array([12],np.int64))
    model=models.build('full',scaler,30).eval()
    with torch.inference_mode():
        output=model(flow,torch.from_numpy(ext))
    assert tuple(output.shape)==(1,2,12,16) and bool(torch.isfinite(output).all())
    print('Parameters:',sum(p.numel() for p in model.parameters()))
    print('Output shape:',tuple(output.shape))
    print('Synthetic forward passed. Random weights; no benchmark accuracy claim.')

if __name__=='__main__':main()
