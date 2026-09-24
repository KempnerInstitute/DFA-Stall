#!/usr/bin/env python3
"""Archive and verify the exact CIFAR-10 records shared by the figure and table.

Requires the original E11/E12 logs. The resulting archive is sufficient to
regenerate the figure and table without those full logs or image data.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cmc.cifar_reporting import CONDITIONS, selected_layer, summarize, trajectory


def main():
    destination = ROOT / 'results/cifar_reporting'
    destination.mkdir(exist_ok=True)
    frames, sources = [], []
    for experiment, tag in CONDITIONS:
        paths = sorted((ROOT / 'results' / experiment / tag).glob('seed*_fb*.csv'))
        if len(paths) != 3:
            raise ValueError(f'Expected three runs for {experiment}/{tag}, found {len(paths)}')
        for path in paths:
            metadata_path = path.with_suffix('.meta.json')
            meta = json.loads(metadata_path.read_text())
            layer = selected_layer(meta)
            raw = pd.read_csv(path)
            columns = {'cosine': f'cos_l{layer}', 'participation': f'p_l{layer}', 'accuracy': 'probe_acc'}
            selected = raw[['step', *columns.values()]].rename(columns={v: k for k, v in columns.items()})
            selected = selected.assign(experiment=experiment, tag=tag, run=path.stem,
                                       seed=meta['seed'], fb_seed=meta['fb_seed'], layer=layer)
            frames.append(selected)
            sources.append(dict(experiment=experiment, tag=tag, run=path.stem, layer=layer,
                                columns=columns, metadata=meta,
                                csv=str(path.relative_to(ROOT)), csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                metadata_file=str(metadata_path.relative_to(ROOT)),
                                metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest()))
    records = pd.concat(frames, ignore_index=True)
    per_run = summarize(records)
    audit = []
    for experiment, tag in CONDITIONS:
        group = per_run[(per_run.experiment == experiment) & (per_run.tag == tag)]
        layer = int(group.layer.iloc[0])
        original = pd.read_csv(ROOT / 'results/paper_summary_sources' / f'{experiment}__summary_per_run.csv')
        original = original[original.tag == tag].set_index(['seed', 'fb_seed'])
        for row in group.itertuples():
            old = original.loc[(row.seed, row.fb_seed)]
            np.testing.assert_allclose([row.maxcos, row.minp, row.acc_3000],
                                       [old[f'maxcos_l{layer}'], old[f'minp_w300_l{layer}'], old.acc_3000],
                                       rtol=0, atol=1e-14)
        steps, mean, _ = trajectory(records, experiment, tag)
        figure_peak = float(mean[steps <= 2000].max())
        assert figure_peak <= group.maxcos.mean() + 1e-14
        audit.append(dict(experiment=experiment, tag=tag, layer=layer, runs=len(group),
                          maxcos_mean=group.maxcos.mean(), maxcos_sd=group.maxcos.std(),
                          minp_mean=group.minp.mean(), minp_sd=group.minp.std(),
                          figure_mean_peak_2000=figure_peak,
                          layer1_reference_maxcos=original.maxcos_l1.mean(),
                          layer1_reference_minp=original.minp_w300_l1.mean()))
    records.to_csv(destination / 'trajectories.csv', index=False)
    per_run.to_csv(destination / 'per_run.csv', index=False)
    pd.DataFrame(audit).to_csv(destination / 'audit.csv', index=False)
    manifest = dict(protocol=dict(cosine_window=[0, 4000], participation_window=[0, 300],
                                  accuracy_step=3000, figure_window=[0, 2000],
                                  table_reduction='Per-run extrema, then mean and sample SD (ddof=1)',
                                  figure_reduction='Pointwise run mean and population SD (ddof=0)'), sources=sources)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Checked {len(per_run)} runs in {len(audit)} conditions against preserved summaries.')
    print(pd.DataFrame(audit).to_string(index=False))


if __name__ == '__main__':
    main()
