#!/usr/bin/env python3
"""Main readout-learning evidence and supplementary closure validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
import _figlib as F
from _figlib import S


def useful(results,root):
    dest=results/'review_20260919';d=pd.read_csv(dest/'decoders.csv');r=pd.read_csv(dest/'readouts.csv')
    fig,axes=S.plt.subplots(1,3,figsize=(S.FULL_WIDTH,2.15),gridspec_kw={'width_ratios':[.85,1.2,1.]})
    ax=axes[0]
    for i,c in enumerate(['initial','collapse','recovery']):
        v=d[(d.layer==3)&(d.checkpoint==c)].heldout_accuracy
        ax.scatter(np.full(len(v),i)+np.linspace(-.06,.06,len(v)),v,s=12,alpha=.55,color=S.color(0))
        ax.errorbar(i,v.mean(),yerr=v.std(),fmt='o',color=S.INK,ms=3,capsize=3)
    ax.set(xticks=[0,1,2],xticklabels=['initial','collapse','recovery'],ylim=(.8,.98),ylabel='held-out accuracy')
    ax.tick_params(axis='x',labelsize=6.5);S.panel_title(ax,'a','frozen-feature decoding')
    ax=axes[1]
    # Reference: the same raw-feature readout continued from the initial (uncollapsed) checkpoint.
    for name,style,label in (('raw',(0,(3,2)),'raw, initial'),('scaled',(0,(1,1.3)),'rescaled, initial')):
        g=r[(r.checkpoint=='initial')&(r.readout==name)].groupby('step').accuracy
        ax.plot(g.mean().index,g.mean(),color=S.MUTED,lw=1.2,ls=style,label=label)
    for i,(name,label) in enumerate([('raw','raw'),('centered','centered'),('scaled','centered + scaled')]):
        g=r[(r.checkpoint=='collapse')&(r.readout==name)].groupby('step').accuracy
        m,s=g.mean(),g.std();ax.plot(m.index,m,color=S.color(i),label=label)
        S.band(ax,m.index,m,s,S.color(i),alpha=.12)
    ax.set(xlabel='readout continuation step',ylabel='held-out accuracy',xlim=(0,3000))
    ax.set_ylim(0,1.3)   # headroom for a two-column key above the curves
    ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.legend(fontsize=6,loc='upper left',ncol=2,handlelength=1.3,columnspacing=.8,handletextpad=.3,borderaxespad=.2)
    S.panel_title(ax,'b','readout learning')
    data=pd.concat([pd.read_csv(p).assign(run=p.stem) for p in (results/'E1_headline/base').glob('seed*_fb*.csv')])
    mean=data.groupby('step').mean(numeric_only=True);ax=axes[2]
    for _,g in data.groupby('run'):
        g=g.dropna(subset=['p_l3','cosA_l3']);ax.plot(g.p_l3,g.cosA_l3,color='#bbc1c8',lw=.45,alpha=.35)
    g=mean[['p_l3','cosA_l3']].dropna();ax.plot(g.p_l3,g.cosA_l3,color=S.MUTED,lw=.8)
    points=ax.scatter(g.p_l3,g.cosA_l3,c=g.index,cmap='viridis',norm=Normalize(0,3000),s=8,zorder=3)
    key=ax.inset_axes([.05,.9,.36,.045]);bar=S.plt.colorbar(points,cax=key,orientation='horizontal',ticks=[0,3000])
    bar.ax.tick_params(labelsize=5.8,length=1.5,pad=1);bar.outline.set_linewidth(.4)
    key.text(1.06,.5,'step',transform=key.transAxes,fontsize=5.8,va='center')
    for t,m,label in [(0,'x','initial'),(100,'s','plateau'),(3000,'*','recovered')]:
        ax.scatter(mean.loc[t,'p_l3'],mean.loc[t,'cosA_l3'],marker=m,s=35 if m=='*' else 18,color=S.INK,zorder=4,label=label)
    ax.axhline(0,color=S.MUTED,lw=.7,ls='--');ax.set(xlabel=r'participation $p_3$',ylabel=r'gradient cosine $\cos\alpha_3$')
    ax.legend(fontsize=6.2,loc='lower right',handletextpad=.2);S.panel_title(ax,'c','useful hidden updates')
    fig.subplots_adjust(left=.075,right=.99,bottom=.23,top=.88,wspace=.6)
    S.save(fig,'fig_useful_learning',root=root)


def validation(results,root):
    dest=results/'review_20260919';r=pd.read_csv(dest/'readout_budget.csv');h=pd.read_csv(results/'revision_dynamics/drift.csv')
    h=h[(h.config=='base')&(h.layer==3)].copy()
    for c in [c for c in h if c.endswith('_projection')]:h[c]=h[c]/h.actual_norm
    fig,axes=S.plt.subplots(2,3,figsize=(S.FULL_WIDTH,4.55))
    names=[('leading','mean drive'),('upstream','upstream'),('covariance','local covariance'),('residual','gate residual')]
    for ax,d,letter,title in [(axes[0,0],h,'a','hidden mean drift'),(axes[0,1],r,'b','readout mean drift')]:
        for i,(key,label) in enumerate(names):
            if key=='residual' and d is r:key,label='output_map','output-map error'
            col=key+'_projection';g=d[d.step<=100].groupby('step')[col];m,s=g.mean(),g.std()
            ax.plot(m.index,m,color=S.color(i),label=label);S.band(ax,m.index,m,s,S.color(i),alpha=.1)
        ax.axhline(0,color=S.MUTED,lw=.6);ax.set(xlabel='step',ylabel='signed drift fraction',xlim=(0,100))
        ax.legend(fontsize=5.1,loc='center right',handlelength=1);S.panel_title(ax,letter,title)
    ax=axes[0,2]
    for i,(key,label) in enumerate([('mean_error_norm','measured mean error'),('output_map_error','output-map discrepancy')]):
        g=r[r.step<=300].groupby('step')[key];m,s=g.mean(),g.std();ax.plot(m.index,m,color=S.color(i),label=label)
        ax.fill_between(m.index,np.maximum(m-s,1e-8),m+s,color=S.color(i),alpha=.1,linewidth=0)
    ax.set(xlabel='step',ylabel='norm',yscale='log',xlim=(0,300));ax.legend(fontsize=5.4,loc='upper right');S.panel_title(ax,'c','readout nonlinearity')
    ax=axes[1,0];a=pd.read_csv(dest/'recovery_trajectories.csv')
    run=pd.concat([pd.read_csv(p) for p in (results/'E7_anatomy/base').glob('seed*_fb*.csv')]).groupby('step').lam_l3.mean()
    run=run[run.index<=700];ax.plot(run.index,run,color=S.INK,label='network',lw=1.8)
    for i,(v,label) in enumerate([('full','full closure'),('no_local','no local growth'),('no_participation','no participation factor'),('measured_init','measured initial covariance')]):
        m=a[(a.variant==v)&(a.step<=700)].groupby('step').Lambda_l3.mean()
        ax.plot(m.index,m,color=S.color(i),ls='--',lw=1.1,label=label)
    ax.set(xlabel='step',ylabel=r'$\|\Lambda_3\|_F^2$',yscale='log',ylim=(1e-4,10),xlim=(0,700))
    S.panel_title(ax,'d','covariance trajectories')
    ax=axes[1,1]
    for i,(key,label) in enumerate(names):
        g=h[h.step<=300].groupby('step')[key+'_norm'];v,s=g.mean(),g.std()
        ax.plot(v.index,v,color=S.color(i),label=label)
        ax.fill_between(v.index,np.maximum(v-s,1e-8),v+s,color=S.color(i),alpha=.12,linewidth=0)
    ax.set(xlabel='step',ylabel='hidden mean-shift norm',xlim=(0,300),yscale='log')
    ax.legend(fontsize=5.2,loc='upper right',handlelength=1)
    S.panel_title(ax,'e','absolute hidden drift')
    ax=axes[1,2];m=pd.read_csv(dest/'common_update.csv')
    for i in [1,2,3]:
        g=m[(m.layer==i)&(m.step<=300)].groupby('step').covariance_to_rank;v,s=g.mean(),g.std()
        ax.plot(v.index,v,color=S.color(i-1),label=f'L{i}');S.band(ax,v.index,v,s,S.color(i-1),alpha=.12)
    ax.set(xlabel='step',ylabel='covariance / rank-one norm',xlim=(0,300),yscale='log')
    ax.legend(fontsize=6,loc='lower right');S.panel_title(ax,'f','shared-error update')
    fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.93,wspace=.52,hspace=.65)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,fontsize=5.5,title='Covariance trajectories (d)',title_fontsize=5.5,frameon=False)
    S.save(fig,'fig_review_validation',root=root)


if __name__=='__main__':
    args=F.parse(__doc__);S.use_style();useful(Path(args.results),args.out_root);validation(Path(args.results),args.out_root)
