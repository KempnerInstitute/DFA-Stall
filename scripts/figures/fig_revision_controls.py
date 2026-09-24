#!/usr/bin/env python3
"""Figures for the optimizer, drift, batch-noise and width-depth controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import _figlib as F
from _figlib import S


def runs(root,family,tag):
    paths=sorted((root/family/tag).glob("seed*_fb*.csv"))
    if not paths:raise FileNotFoundError(f"{family}/{tag}")
    return pd.concat([pd.read_csv(p).assign(seed=p.stem) for p in paths],ignore_index=True)


def curve(ax,data,x,y,color,label,band=True):
    data=data.dropna(subset=[x,y])
    grouped=data.groupby(x)[y];m=grouped.mean();sd=grouped.std()
    ax.plot(m.index,m,color=color,label=label,lw=1.2)
    if band:S.band(ax,m.index,m,sd,color,alpha=.13)


def optimizer(args):
    root=Path(args.results);fig,ax=S.plt.subplots(2,2,figsize=(S.FULL_WIDTH,4.25))
    for i,(tag,label) in enumerate((("sgd_base","SGD"),("adam0.001_base",r"Adam $10^{-3}$"),("adam0.0001_base",r"Adam $10^{-4}$"))):
        d=runs(root,"E13_adam",tag)
        curve(ax[0,0],d,"step","probe_loss",S.color(i),label)
        curve(ax[0,1],d[d.step<=800],"step","p_l3",S.color(i),label)
        curve(ax[1,1],d[d.step<=800],"step","step_cos_l3",S.color(i),label)
    meta=json.loads((root/"E13_adam/sgd_base/seed0_fb0.meta.json").read_text())
    ax[0,0].axhline(meta["prior_loss"],color=S.MUTED,ls="--",lw=.7,label="class-prior loss")
    ax[0,0].set(xlim=(0,1500),ylabel="probe loss");ax[0,0].legend(fontsize=6)
    ax[0,1].set(ylabel=r"participation $p_3$",ylim=(0,1.03))
    for i,(tag,label) in enumerate((("base","plain DFA"),("center_e","centered error"),("prior","prior bias"),("bp","BP"))):
        d=runs(root,"E13_adam","adam0.001_"+tag)
        curve(ax[1,0],d[d.step<=800],"step","cos_l3",S.color(i),label)
    ax[1,0].set(ylabel="hidden cosine",ylim=(0,1.05));ax[1,0].legend(fontsize=6)
    ax[1,1].set(ylabel="applied-step alignment");ax[1,1].axhline(0,color=S.MUTED,ls="--",lw=.7)
    for a,letter,title in zip(ax.flat,"abcd",("loss recovery","gate concentration","Adam controls","actual optimizer displacement")):
        a.set_xlabel("step");S.panel_title(a,letter,title)
    fig.subplots_adjust(left=.09,right=.98,bottom=.11,top=.94,wspace=.38,hspace=.62)
    S.save(fig,"fig_revision_adam",args.out_root)


def dynamics(args):
    root=Path(args.results)/"revision_dynamics";d=pd.read_csv(root/"drift.csv")
    fig,ax=S.plt.subplots(2,3,figsize=(S.FULL_WIDTH,4.0))
    names=[("leading","mean error"),("residual","gate–error residual"),("covariance","local covariance"),("upstream","upstream change")]
    for l,a in enumerate(ax[0],1):
        data=d[(d.config=="base")&(d.layer==l)&(d.step<=300)]
        for i,(term,label) in enumerate(names):curve(a,data,"step",term+"_norm",S.color(i),label)
        a.set(yscale="log",xlabel="step",ylabel="mean-shift norm",ylim=(1e-5,10))
        S.panel_title(a,"abc"[l-1],f"layer {l} drift budget")
    ax[0,0].legend(fontsize=5.2,loc="lower left",handlelength=1.1,labelspacing=.2)
    residual=pd.read_csv(root/"residual.csv")
    for i,(config,label) in enumerate((("base","tanh MNIST"),("relu","ReLU MNIST"),("cifar_mlp","tanh CIFAR-10"))):
        data=residual[(residual.config==config)&(residual.layer==3)&(residual.step<=300)]
        curve(ax[1,0],data,"step","ratio",S.color(i),label)
    ax[1,0].set(xlabel="step",ylabel=r"$\|r_3\|/\|\bar\gamma_3\odot B_3\bar e\|$",yscale="log")
    ax[1,0].axhline(1,color=S.MUTED,lw=.7,ls="--");ax[1,0].legend(fontsize=5.2)
    S.panel_title(ax[1,0],"d","residual across settings")
    masks=pd.read_csv(root/"masks.csv")
    selections=[("full","all units"),("fixed120","top fifth at 120"),("current","current top fifth"),("random","random fifth"),("complement120","outside fixed fifth")]
    for i,(key,label) in enumerate(selections):
        sub=masks[(masks.layer==3)&(masks.selection==key)]
        curve(ax[1,1],sub,"step","cosine",S.color(i),label)
        sub=masks[(masks.step==3000)&(masks.selection==key)]
        g=sub.groupby("layer").cosine
        ax[1,2].errorbar(np.arange(1,4)+(i-2)*.045,g.mean(),yerr=g.std(),fmt="o-",ms=2.5,lw=1,color=S.color(i),capsize=1)
    ax[1,1].set(xlabel="step",ylabel="gradient alignment",xscale="log")
    ax[1,1].legend(fontsize=4.8,labelspacing=.15,loc="upper left",handlelength=1)
    ax[1,2].set(xlabel="hidden layer",ylabel="gradient alignment",xticks=[1,2,3])
    for a in ax[1,1:]:a.axhline(0,color=S.MUTED,ls="--",lw=.7)
    S.panel_title(ax[1,1],"e","matched steps: layer 3");S.panel_title(ax[1,2],"f","matched steps: step 3000")
    fig.subplots_adjust(left=.075,right=.98,bottom=.12,top=.93,wspace=.49,hspace=.7)
    S.save(fig,"fig_revision_dynamics",args.out_root)


def batch(args):
    root=Path(args.results);n=pd.read_csv(root/"revision_dynamics/batch_noise.csv")
    r=pd.read_csv(root/"revision_per_run.csv");r=r[r.family=="E16_batch"]
    fig,ax=S.plt.subplots(2,2,figsize=(S.FULL_WIDTH,4.05))
    for i,(dataset,label) in enumerate((("mnist","softmax MNIST"),("cifar10","calibrated CIFAR-10"))):
        sub=n[n.dataset==dataset]
        for j,(key,name) in enumerate((("raw_rms","raw mean error"),("centered_rms","centered fluctuation"))):
            curve(ax[0,i],sub,"batch",key,S.color(j),name)
            m=sub.groupby("batch")["predicted_"+key].mean()
            ax[0,i].plot(m.index,m,color=S.INK,ls="--",lw=.7,alpha=.65,zorder=4, label="sampling identity" if j == 0 else None)
        ax[0,i].set(xscale="log",yscale="log",xlabel="batch size",ylabel="RMS norm")
        ax[0,i].legend(fontsize=6);S.panel_title(ax[0,i],"ab"[i],label+": fixed network")
        sub=r[r.tag.str.startswith(dataset)];g=sub.groupby("batch")
        ax[1,i].errorbar(g.maxcos.mean().index,g.maxcos.mean(),yerr=g.maxcos.std(),fmt="o-",ms=3,color=S.color(2),capsize=2)
        ax[1,i].set(xscale="log",xlabel="batch size",ylabel="peak hidden cosine",ylim=(0,1.05))
        S.panel_title(ax[1,i],"cd"[i],label+": training")
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.94,wspace=.35,hspace=.68)
    S.save(fig,"fig_revision_batch",args.out_root)


def grid(args):
    root=Path(args.results)
    m=pd.read_csv(root/"reduced_grid.csv");t=pd.read_csv(root/"reduced_grid_coverage.csv")
    for d in (m,t):
        d["width"]=d.tag.str.extract(r"w(\d+)").astype(int)
        d["depth"]=d.tag.str.extract(r"d(\d+)").astype(int)
    deep=m[m.layer==m.depth].copy();deep["abs_error"]=(deep.p_model-deep.p_meas).abs()
    fig,ax=S.plt.subplots(2,3,figsize=(S.FULL_WIDTH,4.3))
    for a,data,key,title,letter in [(ax[0,0],deep,"p_meas","network participation","a"),
                                  (ax[0,1],deep,"p_model","model participation","b"),
                                  (ax[0,2],deep,"abs_error","absolute prediction error","c"),
                                  (ax[1,0],t,"exit_meas","network exit (steps)","d"),
                                  (ax[1,1],t,"exit_model","model exit (steps)","e")]:
        data=data.copy();data.loc[data[key]<0,key]=np.nan
        table=data.pivot_table(index="depth",columns="width",values=key).reindex(index=[1,3,5,7],columns=[100,300,600,900])
        im=a.imshow(table.values,origin="lower",aspect="auto",cmap="viridis",vmin=0,vmax=3000 if "exit" in key else 1)
        a.set(xticks=range(4),xticklabels=[100,300,600,900],yticks=range(4),yticklabels=[1,3,5,7],xlabel="width",ylabel="depth")
        for y,x in np.argwhere(~np.isfinite(table.values)):
            cell=data[(data.depth==[1,3,5,7][y])&(data.width==[100,300,600,900][x])]
            onset_key='onset_model' if key=='exit_model' else 'onset_meas'
            censored='exit' in key and (cell[onset_key]>=0).any()
            a.scatter(x,y,marker='^' if censored else 'x',s=24,
                      facecolors='none' if censored else S.INK,edgecolors=S.INK,linewidths=.8)
        bar=fig.colorbar(im,ax=a,fraction=.045,pad=.035)
        S.panel_title(a,letter,title)
    from fig3_master_curve import panel_width
    panel_width(ax[1,2],pd.read_csv(root/"kappa_transient_per_run.csv"),str(root))
    S.panel_title(ax[1,2],"f","width at fixed depth")
    fig.subplots_adjust(left=.075,right=.95,bottom=.13,top=.94,wspace=.75,hspace=.8)
    S.save(fig,"fig_revision_grid",args.out_root)


def main():
    args=F.parse(__doc__);S.use_style()
    optimizer(args);dynamics(args);batch(args)
    if (Path(args.results)/"reduced_grid.json").exists():grid(args)


if __name__=="__main__":main()
