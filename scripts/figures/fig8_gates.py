#!/usr/bin/env python3
"""Gate turnover, norm-matched descent, and the direction/size of DFA gradients.

This figure complements the main participation/alignment trajectory without
repeating it. Gaussian reconstruction is shown separately with its ablations.
"""
from pathlib import Path
import pandas as pd
import _figlib as F
from _figlib import S


def curve(ax, data, key, label, color):
    g = data.dropna(subset=[key]).groupby('step')[key]
    mean, sd = g.mean(), g.std()
    ax.plot(mean.index, mean, color=color, label=label, lw=1.3)
    S.band(ax, mean.index, mean, sd, color, alpha=.12)


def main():
    args = F.parse(__doc__); S.use_style(); root = Path(args.results)
    fig, axes = S.plt.subplots(2, 2, figsize=(S.FULL_WIDTH, 3.8))
    g = pd.read_csv(root / 'gauss_reconstruction.csv'); ax = axes[0, 0]
    for layer in [1, 2, 3]:
        curve(ax, g[g.layer == layer], 'jaccard_vs_ref', f'layer {layer}', S.color(layer-1))
    ax.axhline(g.jaccard_null.dropna().iloc[0], color=S.MUTED, ls='--', lw=.8, label='chance')
    ax.set(xlabel='step', ylabel='Jaccard overlap', ylim=(0, 1.03))
    ax.legend(fontsize=6, loc='upper right'); S.panel_title(ax, 'a', 'gate-set turnover')
    masks = pd.read_csv(root / 'revision_dynamics/masks.csv'); ax = axes[0, 1]
    for i, (key, label) in enumerate([('full','all units'),('fixed120','fixed top fifth'),
                                    ('current','current top fifth'),('random','random fifth'),
                                    ('complement120','outside fixed fifth')]):
        curve(ax, masks[(masks.layer == 3) & (masks.selection == key)], 'cosine', label, S.color(i))
    ax.set(xlabel='step', ylabel='step alignment', xlim=(0, 3000))
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xticks([0,100,1000,3000], labels=['0','100','1000','3000'])
    ax.axhline(0, color=S.MUTED, ls='--', lw=.7)
    ax.legend(fontsize=5.4, loc='upper left', handlelength=1.1, labelspacing=.15)
    S.panel_title(ax, 'b', 'descent from different gate sets')
    for i, (tag, label) in enumerate([('base','random feedback'),('aligned','pre-aligned feedback')]):
        paths = sorted((root/'E1_headline'/tag).glob('seed*_fb*.csv'))
        if not paths: raise FileNotFoundError(tag)
        data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        curve(axes[1,0], data, 'cosA_l3', label, S.color(i))
        curve(axes[1,1], data, 'gnorm_ratio_l3', label, S.color(i))
    axes[1,0].axhline(0, color=S.MUTED, ls='--', lw=.7)
    axes[1,0].set(xlabel='step', ylabel=r'gradient cosine $\cos\alpha_3$', xlim=(0,3000))
    axes[1,1].set(xlabel='step', ylabel=r'$\|G^{\rm DFA}_3\|/\|G^{\rm BP}_3\|$', xlim=(0,3000))
    axes[1,0].legend(fontsize=6, loc='center right')
    axes[1,1].legend(fontsize=6, loc='lower center')
    S.panel_title(axes[1,0], 'c', 'gradient direction')
    S.panel_title(axes[1,1], 'd', 'relative gradient magnitude')
    fig.subplots_adjust(left=.095, right=.98, bottom=.12, top=.93, wspace=.38, hspace=.65)
    S.save(fig, 'fig8_gates', root=args.out_root)


if __name__ == '__main__':
    main()
