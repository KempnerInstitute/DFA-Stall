#!/usr/bin/env python3
"""Replay three baseline seeds for independent decoder and readout-closure checks.

Protocol is written before measurement. No optimizer or ridge selection is made
from evaluation outcomes. Outputs contain all seeds, snapshots and controls.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cmc import data as D
from cmc.models import MLP
from cmc.rules import make_feedback,step
from cmc.run import Config,make_optimizer
from cmc.review_checks import equivalent_readout,readout_budget


def stats(h,y,C=10):
    h=h.double();t=F.one_hot(y,C).double();hc=h-h.mean(0);tc=t-t.mean(0)
    cov=hc.T@hc/len(h);lam=hc.T@tc/len(h)
    ridge=1e-3*cov.trace()/cov.shape[0]
    coef=torch.linalg.solve(cov+ridge*torch.eye(cov.shape[0],device=h.device),lam)
    score=(lam*coef).sum()/tc.square().mean(0).mean()
    return dict(cov_energy=float(lam.square().sum()),normalized_covariance=float(score),
                rms=float(hc.square().mean().sqrt())),coef,h.mean(0),t.mean(0)


def frozen_readouts(h,he,y,ye,weight,bias,seed,checkpoint):
    """Three equivalent initial functions; fixed SGD on the same sampled indices."""
    m=h.mean(0);s=(h-m).square().mean().sqrt().clamp_min(1e-12)
    versions={'raw':(torch.zeros_like(m),torch.ones_like(s)),
              'centered':(m,torch.ones_like(s)),'scaled':(m,s)}
    weights=[];biases=[];train=[];evaluate=[]
    for mean,scale in versions.values():
        w,b=equivalent_readout(weight,bias,mean,scale)
        weights.append(w.detach().clone().requires_grad_());biases.append(b.detach().clone().requires_grad_())
        train.append((h-mean)/scale);evaluate.append((he-mean)/scale)
    opt=torch.optim.SGD(weights+biases,lr=1e-3)
    target=F.one_hot(y,10).float();te=F.one_hot(ye,10).float()
    generator=torch.Generator().manual_seed(19000+seed)
    indices=torch.randint(len(h),(3000,128),generator=generator).to(h.device)
    rows=[]
    for it in range(3001):
        if it in (0,10,25,50,100,200,300,500,1000,1500,3000):
            with torch.no_grad():
                for name,x,w,b in zip(versions,evaluate,weights,biases):
                    z=x@w.T+b
                    rows.append(dict(seed=seed,checkpoint=checkpoint,readout=name,step=it,
                                     loss=float(F.binary_cross_entropy_with_logits(z,te,reduction='none').sum(1).mean()),
                                     accuracy=float((z.argmax(1)==ye).float().mean())))
        if it==3000:break
        opt.zero_grad(set_to_none=True);ix=indices[it]
        loss=sum(F.binary_cross_entropy_with_logits(x[ix]@w.T+b,target[ix],reduction='none').sum(1).mean()
                 for x,w,b in zip(train,weights,biases))
        loss.backward();opt.step()
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    torch.set_num_threads(2);dest=ROOT/'results/review_20260919';dest.mkdir(exist_ok=True)
    protocol=dict(seeds=[0,1,2],training_steps=3000,fit_examples=10000,evaluation_examples=5000,
                  snapshots='initialization; deepest participation minimum in first 300 original-run updates; step 3000',
                  readout_steps=3000,readout_lr=1e-3,readout_batch=128,ridge_relative=1e-3,
                  controls=['raw','centered','centered and divided by training RMS'],
                  initial_function='checkpoint logits preserved by transforming weights and bias',
                  fit_split='training',evaluation_split='validation; disjoint from fitting',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__),ROOT/'src/cmc/review_checks.py']})
    if args.device=='cuda':protocol['hardware']=dict(name=torch.cuda.get_device_name(),memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    ds=D.load('mnist');dev=torch.device(args.device)
    for seed in protocol['seeds']:
        target=dest/f'replay_seed{seed}.json'
        if target.exists():continue
        cfg=Config(seed=seed,fb_seed=seed,device=args.device,threads=2)
        original=pd.read_csv(ROOT/f'results/E13_adam/sgd_base/seed{seed}_fb{seed}.csv')
        early=original[original.step<=300];minimum=int(early.loc[early.p_l3.idxmin(),'step'])
        snaps={0:'initial',minimum:'collapse',3000:'recovery'}
        sample=D.Sampler(ds['ytr'],128,seed=seed)
        probe=D.Sampler(ds['yval'],2048,seed=seed+999).draw()
        xp,yp=ds['Xval'][probe].to(dev),ds['yval'][probe].to(dev)
        gen=torch.Generator().manual_seed(29000+seed)
        fitidx=torch.randperm(len(ds['ytr']),generator=gen)[:10000]
        evalidx=torch.randperm(len(ds['yval']),generator=gen)[:5000]
        xf,yf=ds['Xtr'][fitidx].to(dev),ds['ytr'][fitidx].to(dev)
        xe,ye=ds['Xval'][evalidx].to(dev),ds['yval'][evalidx].to(dev)
        torch.manual_seed(seed);model=MLP(784,[300]*3,10).to(dev)
        fb=make_feedback(model,10,cfg,dev);opt=make_optimizer(model,cfg)
        budgets=[];common=[];decoders=[];readouts=[];movement=[];checks=[];initial=None
        times=set([0,10,25,50,100,120,200,300,500,1000,1500,3000,minimum])
        for it in range(3001):
            if it:
                ix=sample.draw();step(model,ds['Xtr'][ix].to(dev),ds['ytr'][ix].to(dev),10,cfg,fb,opt,state={})
            if it in times:
                budget,parts=readout_budget(model,xp,yp,cfg,fb,10)
                budgets.append(dict(seed=seed,step=it,**budget));common.extend(dict(seed=seed,step=it,**r) for r in parts)
                with torch.no_grad():
                    _,_,hp=model.forward_blocks(xp,detach=False)
                    if initial is None:initial=[(h-h.mean(0)).clone() for h in hp]
                    for l,h in enumerate(hp,1):
                        centered=h-h.mean(0);u=(1-h.square()).square().mean(0)
                        p=float(u.sum().square()/(len(u)*u.square().sum()))
                        movement.append(dict(seed=seed,step=it,layer=l,centered_displacement=float((centered-initial[l-1]).square().sum()/initial[l-1].square().sum()),p=p))
                        if l==3 and it in original.step.values:
                            old=float(original.loc[original.step==it,'p_l3'].iloc[0]);checks.append(dict(step=it,p_replay=p,p_original=old,absolute_error=abs(p-old)))
            if it in snaps:
                with torch.no_grad():
                    _,_,hf=model.forward_blocks(xf,detach=False);_,_,he=model.forward_blocks(xe,detach=False)
                    for l,(h,ev) in enumerate(zip(hf,he),1):
                        values,coef,mean,tmean=stats(h,yf)
                        pred=(ev.double()-mean)@coef+tmean
                        decoders.append(dict(seed=seed,checkpoint=snaps[it],training_step=it,layer=l,
                                             heldout_accuracy=float((pred.argmax(1)==ye).float().mean()),**values))
                readouts.extend(frozen_readouts(hf[-1],he[-1],yf,ye,model.out.weight.detach(),model.out.bias.detach(),seed,snaps[it]))
                print(f'seed {seed}: completed {snaps[it]} checkpoint at step {it}',flush=True)
        target.write_text(json.dumps(dict(config=asdict(cfg),readout_budget=budgets,common_update=common,
                                         decoders=decoders,readouts=readouts,movement=movement,replay_checks=checks),indent=1)+'\n')
    for key in ['readout_budget','common_update','decoders','readouts','movement']:
        records=[row for f in sorted(dest.glob('replay_seed*.json')) for row in json.loads(f.read_text())[key]]
        pd.DataFrame(records).to_csv(dest/f'{key}.csv',index=False)
    print('Finished readout and feature validation.',flush=True)


if __name__=='__main__':main()
