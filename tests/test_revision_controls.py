"""Numerical controls for adaptive updates and norm-preserving feedback controls."""
import copy
import torch
from cmc.models import MLP
from cmc.rules import make_feedback, step, head_forward
from cmc.run import Config, make_optimizer
from cmc.drift import drift_budget, masked_updates, gate_sets
from cmc.conditions import condition_key

def test_row_normalization_preserves_total_feedback_scale():
    torch.manual_seed(10)
    model=MLP(7,[11,9],4)
    raw=make_feedback(model,4,Config(),"cpu")["B"]
    normalized=make_feedback(model,4,Config(fb_row_norm=1),"cpu")["B"]
    for a,b in zip(raw,normalized):
        assert torch.allclose(a.norm(),b.norm(),rtol=1e-6)
        assert torch.allclose(b.norm(dim=1),b.norm(dim=1).mean().expand(len(b)),rtol=1e-6)

def test_applied_step_measurement_does_not_change_adam_trajectory():
    torch.manual_seed(30)
    model=MLP(7,[11,9],4); twin=copy.deepcopy(model)
    cfg=Config(optimizer="adam",lr=1e-4)
    fb=make_feedback(model,4,cfg,"cpu")
    opt=make_optimizer(model,cfg); opt2=make_optimizer(twin,cfg)
    x=torch.randn(17,7); y=torch.arange(17)%4
    for _ in range(3):
        result=step(model,x,y,4,cfg,fb,opt,state={"measure_update":True})
        step(twin,x,y,4,cfg,fb,opt2,state={})
        for a,b in zip(model.parameters(),twin.parameters()): assert torch.equal(a,b)
        metrics=result["update_metrics"]
        assert metrics["step_descent_out"]>0
        assert abs(metrics["step_cos_out"])<=1.000001

def test_measured_bp_step_matches_first_order_loss_decrease():
    torch.manual_seed(9)
    model=MLP(7,[11],4).double();cfg=Config(rule="bp",lr=1e-6)
    opt=make_optimizer(model,cfg);x=torch.randn(17,7,dtype=torch.float64);y=torch.arange(17)%4
    res=step(model,x,y,4,cfg,{},opt,state={"measure_update":True})["update_metrics"]
    predicted=res["step_descent_l1"]+res["step_descent_out"]
    assert abs(res["actual_batch_loss_decrease"]-predicted)<1e-4*predicted

def test_exact_drift_budget_and_snapshot_nonmutation():
    torch.manual_seed(8)
    model=MLP(7,[11,9],4).double();old=copy.deepcopy(model.state_dict())
    x=torch.randn(23,7,dtype=torch.float64);y=torch.arange(23)%4
    cfg=Config();fb=make_feedback(model,4,cfg,"cpu");fb["B"]=[b.double() for b in fb["B"]]
    rows=drift_budget(model,x,y,4,cfg,fb)
    for row in rows:assert row["reconstruction_error"]<1e-12
    sets=gate_sets(model,x)
    rows=masked_updates(model,x,y,4,cfg,fb,sets,sets,step_norm=1e-5)
    for row in rows:assert abs(row["predicted_decrease"]-row["actual_decrease"])<1e-9
    for k,v in model.state_dict().items():assert torch.equal(v,old[k])

def test_condition_audit_merges_aliases_but_keeps_bias_control():
    base=dict(C=10,seed=0,fb_seed=0,exp="main",tag="base",gate_thr=.5)
    alias=dict(base,exp="sweep",tag="q0.5",out_bias_q=.5,classes=10,steps=1500,gate_thr=.1)
    assert condition_key(base)==condition_key(alias)
    assert condition_key(base)!=condition_key(dict(base,freeze_bias=1))
    assert condition_key(base,True)!=condition_key(dict(base,seed=1),True)

def test_model_timing_keeps_initial_plateaus_and_separates_nondetection():
    import importlib.util
    from pathlib import Path
    import pandas as pd
    spec=importlib.util.spec_from_file_location("reduced_crosscheck",Path(__file__).parents[1]/"scripts/reduced_crosscheck.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    points=pd.DataFrame(dict(exp=["test"]*3,tag=["a","b","c"],layer=[1]*3,
                             p_model=[.1,.3,.5],p_meas=[.2,.4,.6],cos_model=[.9,.7,.5],cos_meas=[.8,.6,.4]))
    timing=pd.DataFrame(dict(onset_meas=[0,5,4],onset_model=[0,5,-1],exit_meas=[20,30,50],
                             exit_model=[20,-1,-1],model_exit_censored=[False,True,False]))
    summary,pairs=module.summarize(points,timing,[])
    assert summary["onset_exit_n"]==2
    assert summary["exit_valid_n"]==1
    assert summary["exit_censored_n"]==1
    assert summary["n_network_only_plateau"]==1
    assert summary["onset_relative_valid_n"]==1
