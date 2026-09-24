"""Shared CIFAR-10 records for the manuscript figure and geometry table."""
from pathlib import Path
import pandas as pd


TABLE_CONDITIONS = [
    ('E11_cifar', 'cnn', 'CNN: plain DFA'),
    ('E11_cifar', 'cnn_bp', 'CNN: BP'),
    ('E11_cifar', 'cnn_priorbias', 'CNN: prior bias'),
    ('E12_cifar2', 'cnn_calib', 'CNN: calibrated logits'),
    ('E11_cifar', 'cnn_centere', 'CNN: center error'),
    ('E11_cifar', 'cnn_centere_pixcenter', 'CNN: center error + inputs'),
    ('E11_cifar', 'cnn_softmax', 'CNN: balanced softmax'),
    ('E11_cifar', 'cnn_softmax_bp', 'CNN: balanced softmax, BP'),
    ('E11_cifar', 'cnn_softmax_imb07', 'CNN: imbalanced softmax'),
    ('E11_cifar', 'cnn_softmax_imb07_bp', 'CNN: imbalanced softmax, BP'),
    ('E11_cifar', 'mlp', 'MLP: plain DFA'),
    ('E11_cifar', 'mlp_bp', 'MLP: BP'),
    ('E12_cifar2', 'mlp_calib', 'MLP: calibrated logits'),
    ('E11_cifar', 'mlp_centere', 'MLP: center error'),
    ('E11_cifar', 'mlp_pixcenter', 'MLP: center inputs'),
    ('E11_cifar', 'mlp_std', 'MLP: standardized inputs'),
    ('E12_cifar2', 'mlp_centere_pixcenter', 'MLP: center error + inputs'),
]
CONDITIONS = [(exp, tag) for exp, tag, _ in TABLE_CONDITIONS] + [
    ('E11_cifar', 'cnn_softmax_imb07_centere')]
RUN_KEYS = ['experiment', 'tag', 'run', 'seed', 'fb_seed', 'layer']


def selected_layer(meta):
    """CNN: final hidden block (including its fully connected block); MLP: layer 1."""
    if meta['arch'] == 'cnn':
        return len(meta['widths'])
    if meta['arch'] == 'mlp':
        return 1
    raise ValueError(f"Unsupported CIFAR architecture: {meta['arch']}")


def load_records(results_root):
    return pd.read_csv(Path(results_root) / 'cifar_reporting/trajectories.csv',
                       float_precision='round_trip')


def summarize(records):
    """Per-run extrema, followed by cross-run aggregation by the table caller."""
    rows = []
    for keys, run in records.groupby(RUN_KEYS, sort=True):
        if run.step.duplicated().any() or not {0, 300, 3000, 4000}.issubset(run.step):
            raise ValueError(f'Incomplete or duplicated CIFAR trajectory: {keys}')
        rows.append(dict(zip(RUN_KEYS, keys),
                         maxcos=float(run.loc[run.step <= 4000, 'cosine'].max()),
                         minp=float(run.loc[run.step <= 300, 'participation'].min()),
                         acc_3000=float(run.loc[run.step == 3000, 'accuracy'].iloc[0])))
    return pd.DataFrame(rows)


def trajectory(records, experiment, tag):
    """Pointwise mean and population SD, preserving the figure's existing convention."""
    rows = records[(records.experiment == experiment) & (records.tag == tag)]
    if rows.empty:
        raise ValueError(f'Missing CIFAR figure records: {experiment}/{tag}')
    wide = rows.pivot(index='step', columns='run', values='cosine')
    return wide.index.to_numpy(), wide.mean(axis=1).to_numpy(), wide.std(axis=1, ddof=0).to_numpy()
