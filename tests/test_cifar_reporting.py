"""Regression checks for the CIFAR figure/table layer and reduction mismatch."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from cmc.cifar_reporting import load_records, selected_layer, summarize, trajectory

ROOT = Path(__file__).resolve().parents[1]


def test_archived_cifar_geometry_uses_documented_layers_and_matching_runs():
    records = load_records(ROOT / 'results')
    sources = json.loads((ROOT / 'results/cifar_reporting/manifest.json').read_text())['sources']
    for source in sources:
        expected = 3 if source['metadata']['arch'] == 'cnn' else 1
        assert selected_layer(source['metadata']) == source['layer'] == expected
        assert source['columns']['cosine'] == f'cos_l{expected}'
        assert source['columns']['participation'] == f'p_l{expected}'
        rows = records[(records.experiment == source['experiment']) &
                       (records.tag == source['tag']) & (records.run == source['run'])]
        assert rows.layer.eq(expected).all() and len(rows) == 401
    summary = summarize(records)
    assert len(summary) == len(sources) == 54
    for (experiment, tag), group in summary.groupby(['experiment', 'tag']):
        steps, mean, _ = trajectory(records, experiment, tag)
        assert mean[steps <= 2000].max() <= group.maxcos.mean() + 1e-14


def test_table_extrema_are_taken_per_run_before_averaging():
    frames = []
    for seed, cosines in enumerate([[.2, .8, .5, .4], [.9, .3, .4, .2]]):
        frames.append(pd.DataFrame(dict(experiment='example', tag='cnn', run=f'seed{seed}',
                                        seed=seed, fb_seed=seed, layer=3,
                                        step=[0, 300, 3000, 4000], cosine=cosines,
                                        participation=[.9, .2 + seed * .2, .1, .05],
                                        accuracy=[.1, .2, .7 + seed * .1, .9])))
    records = pd.concat(frames)
    summary = summarize(records)
    np.testing.assert_allclose(summary.maxcos, [.8, .9])
    np.testing.assert_allclose(summary.minp, [.2, .4])
    np.testing.assert_allclose(summary.acc_3000, [.7, .8])
    _, mean, _ = trajectory(records, 'example', 'cnn')
    assert summary.maxcos.mean() > mean.max()
