from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from cmc.models import MLP
from cmc.run import Config
from cmc.rules import make_feedback
from cmc.review_checks import equivalent_readout,mean_error_update_parts,readout_budget


def test_shared_error_covariance_reconstructs_and_need_not_vanish():
    gamma=torch.tensor([[.1,.8],[.9,.2],[.3,.4]],dtype=torch.float64)
    h=torch.tensor([[1.,3.],[2.,-1.],[-2.,4.]],dtype=torch.float64)
    full,rank,cov=mean_error_update_parts(gamma,h,torch.tensor([2.,-1.]),.01)
    torch.testing.assert_close(full,rank+cov)
    assert cov.norm()>0
    assert torch.linalg.matrix_rank(rank)==1


def test_readout_reparameterization_preserves_every_prediction():
    torch.manual_seed(17);h=torch.randn(30,7);w=torch.randn(3,7);b=torch.randn(3)
    mean=h.mean(0);scale=(h-mean).square().mean().sqrt()
    ww,bb=equivalent_readout(w,b,mean,scale)
    torch.testing.assert_close(h@w.T+b,((h-mean)/scale)@ww.T+bb)


def test_scalar_feature_scaling_matches_weight_rate_with_unchanged_intercept_rate():
    torch.manual_seed(73)
    h=.038*torch.randn(128,20,dtype=torch.float64)+torch.randn(20,dtype=torch.float64)
    mean=h.mean(0); c=h-mean; scale=c.square().mean().sqrt()
    target=torch.nn.functional.one_hot(torch.randint(4,(128,)),4).to(h.dtype)
    w=torch.randn(4,20,dtype=h.dtype); b=torch.randn(4,dtype=h.dtype)
    wp,bp=equivalent_readout(w,b,mean,scale)
    u=w.clone(); intercept=b+w@mean
    wp.requires_grad_();bp.requires_grad_();u.requires_grad_();intercept.requires_grad_()
    scaled=torch.optim.SGD([wp,bp],lr=.001)
    centered=torch.optim.SGD([{'params':[u],'lr':.001/float(scale**2)},
                              {'params':[intercept],'lr':.001}])
    for i in range(30):
        ix=torch.arange(i%4*32,(i%4+1)*32)
        for opt,z in [(scaled,(c[ix]/scale)@wp.T+bp),(centered,c[ix]@u.T+intercept)]:
            opt.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(z,target[ix],reduction='sum').div(len(ix)).backward();opt.step()
        torch.testing.assert_close((c/scale)@wp.T+bp,c@u.T+intercept,rtol=1e-12,atol=1e-12)


def test_condition_bootstrap_does_not_gain_precision_from_replicating_runs():
    from cmc.reporting import condition_correlation
    x=np.array([1.,1.1,2.,2.2,3.,3.1,4.,4.2])
    y=np.array([2.,1.8,1.,1.1,4.,3.8,2.,2.2])
    groups=np.repeat(np.arange(4),2)
    a=condition_correlation(x,y,groups,draws=200)
    b=condition_correlation(np.tile(x,5),np.tile(y,5),np.tile(groups,5),draws=200)
    np.testing.assert_allclose([a[k] for k in ('r','lo','hi')],[b[k] for k in ('r','lo','hi')],atol=1e-12)
    assert a['n_conditions']==b['n_conditions']==4
    assert b['n']==5*a['n']


def test_readout_budget_reconstructs_simultaneous_step():
    torch.manual_seed(9);m=MLP(5,[8,8],3);cfg=Config(width=8,depth=2,lr=.002)
    x=torch.randn(40,5)+.4;y=torch.randint(3,(40,));fb=make_feedback(m,3,cfg,'cpu')
    before=[p.detach().clone() for p in m.parameters()]
    row,parts=readout_budget(m,x,y,cfg,fb,3)
    assert row['reconstruction_error']<1e-6
    assert abs(sum(v for k,v in row.items() if k.endswith('_projection'))-1)<1e-4
    assert max(r['reconstruction_error'] for r in parts)<1e-7
    for p,q in zip(m.parameters(),before):torch.testing.assert_close(p,q)


def test_full_recovery_variant_matches_original_integrator():
    import sys
    from pathlib import Path
    from dataclasses import replace
    from cmc.reduced import Init,ReducedConfig,simulate
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from review_model_ablation import cascades,model_detection
    rng=np.random.default_rng(8);widths=[15,12];C=4
    init=Init(H0=3.,mu0=[rng.normal(0,.2,w) for w in widths],sigma0=[.4,.4],
              zbar0=np.zeros(C),q=[1.,1.],r=[1.,1.],Lam0=.7,lam0=[.2,.1],
              widths=widths,C=C,tbar=np.ones(C)/C,sigma_x2=.1)
    B=[rng.normal(size=(w,C)) for w in widths]
    cfg=ReducedConfig(eta_hid=.01,eta_out=.01,steps=300,log_every=1)
    reference=simulate(init,cfg,B)
    collapse=simulate(init,replace(cfg,escape=False),B)
    variants=cascades(collapse,init,cfg);full=variants[variants.variant=='full']
    for col in ['loss_model','Lambda_l1','Lambda_l2']:
        np.testing.assert_allclose(full[col].iloc[:len(reference)],reference[col],rtol=1e-12,atol=1e-12)
    assert model_detection(full,reference.attrs['prior_loss'],cfg)==dict(onset=reference.attrs['onset'],exit=reference.attrs['exit'])


def test_learning_time_requires_sustained_crossing_and_full_horizon():
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from review_reporting import sustained_crossing
    df=pd.DataFrame({'step':np.arange(0,101,10),'probe_loss':[2,0,0,2,2,0,0,0,0,0,0]})
    assert sustained_crossing(df,1)==50
    assert np.isnan(sustained_crossing(df.iloc[:-1],1))


def test_gain_and_hidden_rate_give_same_plain_sgd_trajectory():
    from copy import deepcopy
    from cmc.rules import step
    torch.manual_seed(19);a=MLP(5,[8,8],3);b=deepcopy(a)
    ca=Config(lr=.01,fb_scale=1.);cb=Config(lr=.01,fb_scale=4.,hid_lr_mult=.25)
    fa=make_feedback(a,3,ca,'cpu');fb=make_feedback(b,3,cb,'cpu')
    oa=torch.optim.SGD([{'params':a.hidden.parameters(),'lr':.01},{'params':a.out.parameters(),'lr':.01}])
    ob=torch.optim.SGD([{'params':b.hidden.parameters(),'lr':.0025},{'params':b.out.parameters(),'lr':.01}])
    for _ in range(7):
        x=torch.randn(32,5);y=torch.randint(3,(32,))
        step(a,x,y,3,ca,fa,oa);step(b,x,y,3,cb,fb,ob)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)


def test_anatomy_probe_on_available_devices():
    from cmc.diagnostics import probe_pass
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        torch.manual_seed(11);m=MLP(5,[8],3).to(device);cfg=Config(anatomy=1,width=8,depth=1)
        x=torch.randn(64,5,device=device);y=torch.randint(3,(64,),device=device)
        fb=make_feedback(m,3,cfg,device);r=probe_pass(m,x,y,3,cfg,fb,heavy=True)
        assert np.isfinite(r['R_l1']) and r['R_l1']>=0
