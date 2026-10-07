"""Checks for ranking, objective, and recoverable-training research errors."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from speechrelcal.calibration import fit_calibration, probability, product
from speechrelcal.metrics import average_precision, normalized_average_precision
from speechrelcal.models import make_model
from speechrelcal.training import _query_loss, train_model


def test_rank_preserving_maps_and_full_state_normalization():
    rng = np.random.default_rng(41)
    raw = rng.normal(size=1200)
    truth = raw + rng.normal(size=1200)
    labels = np.where(truth <= -.5, 0, np.where(truth >= .5, 2, 1))
    cal = fit_calibration(raw, labels, 'ordinal')
    values = np.linspace(-2, 2, 101)
    for state, direction in [('higher',1),('lower',-1)]:
        for interface in ['sigmoid','slope_only','intercept_only','full_affine']:
            p = probability(values, cal, state, interface)
            assert (direction*np.diff(p) > 0).all()
    p = [probability(values,cal,state) for state in ['lower','similar','higher']]
    np.testing.assert_allclose(np.sum(p,axis=0),1,atol=1e-14)
    binary = fit_calibration(raw,(truth>0).astype(int),'binary')
    np.testing.assert_allclose(probability(values,binary,'same')+probability(values,binary,'different'),1)


def test_product_and_multi_positive_objective():
    p = torch.tensor([[.8,.7],[.9,.3],[.2,.8]],dtype=torch.float32)
    m = torch.ones_like(p)
    model = make_model('log_linear',2)
    np.testing.assert_allclose(model(p,m).detach().numpy(),product(p.numpy()),atol=1e-6)
    scores = torch.tensor([-torch.inf,1.,2.,3.],requires_grad=True)
    positive = torch.tensor([False,True,False,True])
    loss = _query_loss(scores,positive)
    expected = -torch.log_softmax(scores,0)[positive].mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    torch.testing.assert_close(scores.grad[0],torch.tensor(0.))
    # Independent hand calculation: positives at ranks 1 and 3.
    assert average_precision(np.array([3.,2.,1.]),np.array([1,0,1])) == pytest.approx(5/6)
    assert normalized_average_precision(np.array([3.,2.,1.]),np.array([1,0,1])) == pytest.approx(.5)


def test_checkpoint_recovery_preserves_updates(tmp_path):
    rng=np.random.default_rng(7)
    arrays=[rng.uniform(.1,.9,(6,3)).astype(np.float32) for _ in range(5)]
    valid=np.array([False,True,False,True,False,False])
    queries=[(i,{},np.arange(6),0,valid) for i in range(5)]
    def probability_fn(role,anchor,relation,candidates):
        return arrays[anchor],np.ones((6,3),dtype=np.float32)
    config=dict(learning_rate=.001,weight_decay=.01,query_batch_size=4)
    torch.manual_seed(17);whole=make_model('deepsets',3,16)
    train_model(whole,queries,probability_fn,config,19,7,tmp_path/'whole.pt')
    torch.manual_seed(17);partial=make_model('deepsets',3,16)
    train_model(partial,queries,probability_fn,config,19,3,tmp_path/'partial.pt')
    resumed=make_model('deepsets',3,16)
    train_model(resumed,queries,probability_fn,config,19,7,tmp_path/'partial.pt')
    for name,value in whole.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],rtol=0,atol=0)


def test_portable_cli(tmp_path):
    import pandas as pd
    def cli(*args):
        subprocess.run([sys.executable,'-m','speechrelcal.cli',*map(str,args)],check=True,capture_output=True,text=True)
    rng=np.random.default_rng(12)
    raw=rng.normal(size=1000);latent=raw+rng.normal(size=1000)
    pairs=pd.DataFrame(dict(factor='pitch',kind='ordinal',score=raw,state=np.where(latent<-.5,0,np.where(latent>.5,2,1))))
    pairs.to_csv(tmp_path/'pairs.csv',index=False)
    cli('calibrate','--pairs',tmp_path/'pairs.csv','--output',tmp_path/'cal.json')
    manifests=[]
    for group in ['fit','val']:
        directory=tmp_path/group;directory.mkdir()
        np.savez(directory/'q.npz',raw=np.array([[0],[1.],[.2],[-.5]]),positive=np.array([False,True,False,False]),
                 candidate_uids=np.array([f'{group}{i}' for i in range(4)]),candidate_speakers=np.array([group]*4))
        rows=[dict(query_id=group,anchor_speaker=group,regime='heldout_pair_tuple',self_index=0,
                   factors=['pitch'],states=['higher'],arrays='q.npz')]
        (directory/'raw.json').write_text(json.dumps(rows))
        cli('prepare','--queries',directory/'raw.json','--calibration',tmp_path/'cal.json','--interface','full_state','--output',directory/'prepared')
        manifests.append(directory/'prepared/queries.json')
    cli('evaluate','--queries',manifests[1],'--output',tmp_path/'product')
    assert json.loads((tmp_path/'product/metrics.json').read_text())['heldout_pair_tuple']==1
    cli('select','--queries',manifests[0],'--validation',manifests[1],'--model','log_linear','--epochs','0','1','--output',tmp_path/'select')
    cli('train','--queries',manifests[0],'--model','log_linear','--epochs','1','--output',tmp_path/'trained')
    cli('evaluate','--queries',manifests[1],'--checkpoint',tmp_path/'trained/model.pt','--output',tmp_path/'learned')
    assert (tmp_path/'learned/metrics.json').exists()
    # Relation-aware preparation must match the DeepSets+R channel order.
    cli('prepare','--queries',tmp_path/'fit/raw.json','--calibration',tmp_path/'cal.json',
        '--interface','sigmoid','--relation-aware','--output',tmp_path/'aware')
    with np.load(tmp_path/'aware/query_0.npz') as d:
        assert (d['inputs'][...,1]==0).all() and (d['inputs'][...,2]==1).all()
        output=make_model('deepsets_r',1,16)(torch.from_numpy(d['inputs']),torch.from_numpy(d['mask']))
        assert output.shape==(4,) and torch.isfinite(output).all()
