#!/usr/bin/env python3
"""Submission reporting: the learning cost of collapse, a closed-form null for the reduced model, the self-termination
check, and the frozen-readout reference. Reads existing records only; trains nothing.

Also reports learning-rate robustness, centered geometry, mean-activity growth under error centering, and
imbalanced readout calibration. Writes compact records and summary.json in results/submission_20260923/
and generated submission_*.tex tables in paper/tables/.
Usage: PYTHONPATH=src python3 scripts/submission_reporting.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    """Load a sibling script as a module (the scripts directory is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Table 17's learning-time criterion and number formatting, shared rather than re-implemented.
_REVIEW = _script('review_reporting')
pm, sustained_crossing = _REVIEW.pm, _REVIEW.sustained_crossing

RES = ROOT / 'results'
DEST = RES / 'submission_20260923'
TABLES = ROOT / 'paper' / 'tables'
E1 = RES / 'E1_headline'

# Conditions of the cost-of-collapse comparison, in display order.
COST = [('base', 'Plain DFA'), ('frzbias', 'Frozen hidden biases'), ('pixcenter', 'Centered inputs'),
        ('pixcenter_frzbias', 'Centered inputs + frozen biases'), ('center_e', 'Centered output error'),
        ('priorbias', 'Prior-bias initialization'), ('outlr10', r'Faster readout ($10\times$)'),
        ('aligned', 'Pre-aligned DFA'), ('bp', 'Backpropagation')]


def first_step(df, column, threshold):
    """First logged step at which a column reaches a threshold."""
    hit = df[df[column] >= threshold]
    return float(hit.step.iloc[0]) if len(hit) else np.nan


def learning_times():
    rows = []
    for tag, _ in COST:
        for f in sorted((E1 / tag).glob('seed*_fb*.csv')):
            if f.name.startswith('reduced_'):
                continue
            d = pd.read_csv(f)
            m = json.loads(f.with_suffix('.meta.json').read_text())
            rows.append(dict(tag=tag, seed=m['seed'], fb_seed=m.get('fb_seed'),
                             time=sustained_crossing(d, .9 * m['prior_loss']),
                             acc50=first_step(d, 'probe_acc', .5), acc80=first_step(d, 'probe_acc', .8),
                             accuracy=float(d.loc[d.step == 3000, 'probe_acc'].iloc[0]) if (d.step == 3000).any()
                             else np.nan))
    return pd.DataFrame(rows)


def kappa_hat(row, layer):
    """Initialization estimate of Eq. (5) for one run and layer (sigmoid effective slope for every readout)."""
    C = int(row.C) if pd.notna(row.C) else 10
    slope = .25 if C <= 2 else (.5 - 1 / C) / np.log(C - 1)
    h_in = row.xbar2 if layer == 1 else row[f'H0_l{layer - 1}']
    h_top = row[f"H0_l{int(row.n_layers)}"]
    return row.fb_scale * row.ebar0 * (h_in + 1) / (slope * (h_top + 1) * (row.out_lr_mult / row.hid_lr_mult))


def null_points():
    """The paper's 450 reduced-model points with the closed-form null p = min(1, 1.5/kappa_hat) added."""
    model = pd.read_csv(RES / 'reduced_crossseed.csv')
    per = pd.concat([pd.read_csv(RES / e / 'summary_per_run.csv').assign(exp=e)
                     for e in sorted(model.exp.unique())], ignore_index=True)
    m = model.merge(per, on=['exp', 'tag', 'seed', 'fb_seed'], how='left', validate='many_to_one')
    if m.ebar0.isna().any():
        raise ValueError('reduced-model points without a matching run summary')
    k = np.array([kappa_hat(r, int(r.layer)) for _, r in m.iterrows()])
    m['kappa_hat'] = k
    m['p_null'] = np.where(k > 0, np.minimum(1., 1.5 / np.where(k > 0, k, 1.)), 1.)
    return m[['exp', 'tag', 'seed', 'fb_seed', 'layer', 'head', 'act', 'C', 'p_meas', 'p_model', 'kappa_hat', 'p_null']]


SWEEPS = [('gain', r'^g[0-9.]+$'), ('output bias', r'^q[0-9.]+$'), ('learning rate', r'^lr'), ('width', r'^w[0-9]+$'),
          ('batch size', r'^b[0-9]+$'), ('class count', r'^C[0-9]+$')]


def within_sweeps(pts):
    """Reduced-model correlation over setting-by-layer means within each single-parameter sweep of E2."""
    import re
    out = {}
    e2 = pts[pts.exp == 'E2_master']
    for name, pattern in SWEEPS:
        g = e2[e2.tag.str.match(pattern)].groupby(['tag', 'layer'])[['p_meas', 'p_model']].mean()
        out[name] = (float(np.corrcoef(g.p_meas, g.p_model)[0, 1]), len(g))
    return out


def agreement(df, pred):
    """Pearson r and mean absolute error, pooled over points and over condition means (as in Table 9)."""
    out = {}
    for name, g in [('pooled', df), ('conditions', df.groupby(['exp', 'tag'], as_index=False)[['p_meas', pred]].mean())]:
        out[name + '_r'] = float(np.corrcoef(g.p_meas, g[pred])[0, 1])
        out[name + '_mae'] = float(np.abs(g.p_meas - g[pred]).mean())
    return out


def main():
    DEST.mkdir(exist_ok=True)
    summary = {}

    # 1. Learning cost of collapse (E1 cohort: 15 DFA trajectories, 5 BP networks).
    t = learning_times()
    t.to_csv(DEST / 'learning_times.csv', index=False)
    per1 = pd.read_csv(E1 / 'summary_per_run.csv')
    rows = []
    for tag, label in COST:
        g, s = t[t.tag == tag], per1[per1.tag == tag]
        summary[f'time_{tag}'] = float(g.time.mean())
        summary[f'acc50_{tag}'] = float(g.acc50.median())
        summary[f'acc80_{tag}'] = float(g.acc80.median())
        summary[f'cos3_{tag}'] = float(s.maxcos_l3.mean())
        rows.append([label, len(g), pm(s.maxcos_l3, 3), pm(g.time, 0), f'{g.acc50.median():.0f}',
                     f'{g.acc80.median():.0f}', pm(g.accuracy, 3)])
    header = ['Condition', '$n$', r'Peak $\cos_3$', 'Learning time', r'50\% acc.', r'80\% acc.', 'Accuracy']
    body = [' & '.join(map(str, r)) + r'\\' for r in rows]
    TABLES.joinpath('submission_cost.tex').write_text('\n'.join([
        '% Generated by scripts/submission_reporting.py.',
        r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
        r'\caption{Learning cost of collapse in the MNIST baseline cohort. Learning time is the first probe loss below '
        r'90\% of the constant-predictor loss that stays below it for 50 updates (the criterion of '
        r'Table~\ref{tab:review_timing}); mean $\pm$ standard deviation. Accuracy columns give the median first step '
        r'reaching 50\% and 80\% probe accuracy; the last column is accuracy at step 3000. Peak cosine uses the full '
        r'trajectory, as in Table~\ref{tab:E1}. Centering prevents collapse and speeds discrimination but lowers '
        r"mean-activity energy, which slows the readout's fit to the class prior; the sigmoid loss then "
        r'stays high although accuracy rises.}\label{tab:submission_cost}',
        r'\begin{tabular}{l' + 'r' * (len(header) - 1) + '}', r'\toprule', ' & '.join(header) + r'\\\midrule',
        *body, r'\bottomrule\end{tabular}\end{table}', '']))

    # 2. Self-termination: shared-to-input-dependent error ratio 10 updates into training.
    for tag in ['base', 'bp', 'bp_outx12', 'aligned']:
        summary[f'ebar_ratio10_{tag}'] = float(per1.loc[per1.tag == tag, 'ebar_ratio_10'].mean())
    summary['ebar0_bp_outx12'] = float(per1.loc[per1.tag == 'bp_outx12', 'ebar0'].mean())

    # 3. Reduced model versus the closed-form null on the same 450 points.
    pts = null_points()
    pts.to_csv(DEST / 'null_points.csv', index=False)
    model, null = agreement(pts, 'p_model'), agreement(pts, 'p_null')
    summary.update({f'model_{k}': v for k, v in model.items()})
    summary.update({f'null_{k}': v for k, v in null.items()})
    summary['null_points'] = int(len(pts))
    summary['null_conditions'] = int(pts.groupby(['exp', 'tag']).ngroups)
    # The closed form's derivation assumes tanh units, sigmoid outputs and C > 2; compare there as well.
    valid = pts[(pts['head'] == 'sigmoid_bce') & (pts['act'] == 'tanh') & (pts['C'] > 2)]
    vmodel, vnull = agreement(valid, 'p_model'), agreement(valid, 'p_null')
    summary.update({f'valid_model_{k}': v for k, v in vmodel.items()})
    summary.update({f'valid_null_{k}': v for k, v in vnull.items()})
    summary['valid_points'] = int(len(valid))
    summary['valid_conditions'] = int(valid.groupby(['exp', 'tag']).ngroups)
    sweeps = within_sweeps(pts)
    for name, (r, n) in sweeps.items():
        summary['sweep_r_' + name.replace(' ', '_')] = r
    single = [r for name, (r, n) in sweeps.items() if name != 'class count']
    summary['sweep_r_min'], summary['sweep_r_max'] = float(min(single)), float(max(single))
    rows = []
    for label, a, b, n in ((f"All {summary['null_conditions']} settings", model, null, summary['null_points']),
                           (f"{summary['valid_conditions']} tanh, sigmoid, $C>2$ settings", vmodel, vnull,
                            summary['valid_points'])):
        for predictor, v in (('Reduced model', a), ('Closed-form estimate', b)):
            rows.append(rf"{label} & {predictor} & {n} & {v['pooled_r']:.2f} & {v['pooled_mae']:.3f} & "
                        rf"{v['conditions_r']:.2f} & {v['conditions_mae']:.3f}\\")
            label = ''
    sweep_text = ', '.join(f"{name} {r:.2f}" for name, (r, n) in sweeps.items())
    TABLES.joinpath('submission_null.tex').write_text('\n'.join([
        '% Generated by scripts/submission_reporting.py.',
        r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
        r'\caption{Participation minima predicted by the reduced model and by the closed-form initialization '
        r'estimate, $\min(1,1.5/\hat\kappa_\ell)$ with $\hat\kappa_\ell$ from Equation~\eqref{eq:kappa}, on '
        r'condition--seed--layer points. The closed form uses the sigmoid effective slope; the second block restricts '
        r'both predictors to the settings its derivation covers. Condition means average seeds and layers, as in '
        r'Table~\ref{tab:review_model_stats}. Within single-parameter sweeps, the reduced-model '
        rf'correlation over setting-by-layer means is: {sweep_text}; the class-count sweep includes the two-class '
        r'setting, where the model predicts almost no drive.}\label{tab:submission_null}',
        r'\begin{tabular}{llrrrrr}', r'\toprule',
        r'Settings & Predictor & Points & $r$ & MAE & $r$, means & MAE, means\\\midrule',
        *rows, r'\bottomrule\end{tabular}\end{table}', '']))

    # 4. Frozen-feature reference: readout continuation from the initial checkpoint, and feature amplitude.
    rev = RES / 'review_20260919'
    r = pd.read_csv(rev / 'readouts.csv')
    end = r[r.step == r.step.max()]
    for ck in ['initial', 'collapse']:
        for ro in ['raw', 'scaled']:
            summary[f'readout_{ck}_{ro}'] = 100 * float(end[(end.checkpoint == ck) & (end.readout == ro)].accuracy.mean())
    dec = pd.read_csv(rev / 'decoders.csv')
    rms = dec[dec.layer == 3].groupby('checkpoint').rms.mean()
    summary['rms3_initial'], summary['rms3_collapse'] = float(rms['initial']), float(rms['collapse'])
    summary['rms3_fold'] = float(rms['initial'] / rms['collapse'])
    summary['rms3_rate_fold'] = float((rms['initial'] / rms['collapse']) ** 2)

    # 5. Global standardization versus pixels / 255: layer-1 gate concentration and peak cosine.
    e2 = pd.read_csv(RES / 'E2_master' / 'summary_per_run.csv')
    for tag, key in [('pre_div255', 'raw'), ('pre_standardize', 'std')]:
        s = e2[e2.tag == tag]
        summary[f'l1_cos_{key}'] = float(s.maxcos_l1.mean())
        summary[f'l1_minp_{key}'] = float(s.minp_w300_l1.mean())

    # 6. Shared and input-dependent error through the baseline plateau (Figure 1g).
    runs = [pd.read_csv(f) for f in sorted((E1 / 'base').glob('seed*_fb*.csv')) if not f.name.startswith('reduced_')]
    traj = pd.concat(runs).groupby('step')[['ebar_norm', 'etil_norm']].mean()
    onset, exit_ = per1.loc[per1.tag == 'base', ['plateau_onset', 'plateau_exit']].mean()
    band = traj[(traj.index >= onset) & (traj.index <= exit_)]
    summary['ebar_100'] = float(traj.loc[100, 'ebar_norm'])
    summary['etil_plateau_min'], summary['etil_plateau_max'] = float(band.etil_norm.min()), float(band.etil_norm.max())
    summary['etil_3000'] = float(traj.loc[3000, 'etil_norm'])

    # 7. The original DFA protocol of Nokland (2016): zero initialization, RMSprop, batch 64 (E19).
    summary.update(nokland())

    # 8. Sign-error feedback (E20) and the feedback-gain sweep (E2): collapse depth against learning time.
    summary.update(sign_and_gain())

    # 8b. Centered geometry of the top layer (E1) and mean-activity growth under error centering (E10).
    summary.update(centered_geometry())
    summary.update(centering_instability())

    # 8c. Learning-rate robustness (E21, with the E1 runs at 1e-3) and a calibrated imbalanced softmax (E22).
    summary.update(learning_rate_robustness())
    summary.update(imbalanced_calibration())

    # 9. Values quoted from other records: default-ridge conditioner accuracy, the MNIST CNN's deepest-block participation
    # (one matched BP run), bias-corrected Adam with BP, and the rescaled initial-feature readout.
    e4 = pd.read_csv(RES / 'E4_cures' / 'summary_per_run.csv')
    summary['cond_default_acc'] = float(e4.loc[e4.tag == 'base__whiten', 'acc_3000'].mean())
    e3 = pd.read_csv(RES / 'E3_generality' / 'summary_per_run.csv')
    for tag in ('cnn', 'cnn_bp'):
        summary[f'{tag}_minp3'] = float(e3.loc[e3.tag == tag, 'minp_w300_l3'].mean())
    groups = json.loads((RES / 'revision_summary.json').read_text())['groups']
    for key, name in (('maxcos', 'cos'), ('minp', 'minp')):
        summary[f'adam_bp_{name}'] = groups['E13_adam/adam0.001_bp'][key]['mean']
        summary[f'adam_dfa_{name}'] = groups['E13_adam/adam0.001_base'][key]['mean']

    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


SIGN = [('sign', 'Sign'), ('sign_prior', 'Sign, prior bias'), ('sign_center_e', 'Sign, centered')]
GAINS = ('g0.03', 'g0.1', 'g0.3', 'g1.0', 'g3.0', 'g10.0')


def trajectory_stats(csv):
    """Collapse and learning-time statistics of one run."""
    d = pd.read_csv(csv)
    m = json.loads(csv.with_suffix('.meta.json').read_text())
    return dict(tag=m['tag'], seed=m['seed'], cos3=float(d.cos_l3.max()), cos3_end=float(d.cos_l3.iloc[-1]),
                p3=float(d[d.step <= 300].p_l3.min()), p3_end=float(d.p_l3.iloc[-1]),
                time=sustained_crossing(d, .9 * m['prior_loss']), acc50=first_step(d, 'probe_acc', .5),
                acc80=first_step(d, 'probe_acc', .8), accuracy=float(d.probe_acc.iloc[-1]), steps=int(d.step.max()))


def sign_and_gain():
    """Sign-error feedback (E20) and the gain sweep of E2: summary keys and a generated table."""
    out, body = {}, []
    runs = [trajectory_stats(f) for f in sorted((RES / 'E20_sign').glob('*/seed*_fb*.csv'))]
    if runs:
        r = pd.DataFrame(runs)
        r.to_csv(DEST / 'sign_runs.csv', index=False)
        for tag, label in SIGN:
            g = r[r.tag == tag]
            for col in ('cos3', 'cos3_end', 'p3', 'p3_end', 'time', 'accuracy'):
                out[f'{tag}_{col}'] = float(g[col].mean())
            out[f'{tag}_acc80_reached'] = int(g.acc80.notna().sum())
            body.append(' & '.join([label, str(len(g)), pm(g.cos3, 3), pm(g.cos3_end, 3), pm(g.p3, 3), pm(g.p3_end, 3),
                                    pm(g.time, 0), pm(g.accuracy, 3)]) + r'\\')
        header = ['Condition', '$n$', r'Peak $\cos_3$', r'Final $\cos_3$', r'Min.\ $p_3$', r'Final $p_3$',
                  'Learning time', 'Accuracy']
        TABLES.joinpath('submission_sign.tex').write_text('\n'.join([
            '% Generated by scripts/submission_reporting.py.',
            r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
            r'\caption{Sign-error feedback in the MNIST baseline: hidden layers receive $\operatorname{sign}(e)$ through the '
            r'baseline feedback, and the readout keeps its gradient. Participation minima use the first 300 updates; final '
            r'values and accuracy are at step 3000. Learning time uses the criterion of Table~\ref{tab:review_timing}. '
            r'Plain sign feedback and prior-bias initialization give identical hidden updates, because '
            r'$\operatorname{sign}(e)=\mathbf 1-2y$ does not depend on the readout. Means $\pm$ standard deviations over '
            r'three seed pairs.}\label{tab:submission_sign}',
            r'\begin{tabular}{l' + 'r' * (len(header) - 1) + '}', r'\toprule', ' & '.join(header) + r'\\\midrule',
            *body, r'\bottomrule\end{tabular}\end{table}', '']))
    runs = [trajectory_stats(f) for tag in GAINS for f in sorted((RES / 'E2_master' / tag).glob('seed*_fb*.csv'))
            if not f.name.startswith('reduced_')]
    if runs:
        r = pd.DataFrame(runs)
        r.to_csv(DEST / 'gain_runs.csv', index=False)
        for tag in GAINS:
            g = r[r.tag == tag]
            key = 'gain_' + tag[1:].replace('.', 'p')
            out[key + '_p3'] = float(g.p3.mean())
            out[key + '_cos3'] = float(g.cos3.mean())
            out[key + '_time'] = float(g.time.mean())
            out[key + '_time_reached'] = int(g.time.notna().sum())
    return out


LR_ARMS = [('dfa', 'base', 'DFA'), ('prior', 'priorbias', 'DFA, prior bias'), ('bp', 'bp', 'BP')]
LR_RATES = (0.001, 0.003, 0.01, 0.03, 0.1)


def run_record(csv):
    """Learning statistics of one run; a run whose loss becomes non-finite or exceeds its initial value at the end
    counts as diverged."""
    d = pd.read_csv(csv)
    loss = d.probe_loss.to_numpy()
    diverged = (not np.isfinite(loss).all()) or loss[-1] > loss[0]
    return dict(acc50=first_step(d, 'probe_acc', .5), acc80=first_step(d, 'probe_acc', .8),
                p3=float(d[d.step <= 300].p_l3.min()), accuracy=float(d.probe_acc.iloc[-1]), diverged=bool(diverged))


def steps_text(value):
    """An update count for a table cell, or a dash when the criterion was not reached."""
    return '--' if not np.isfinite(value) else f'{value:.0f}'


def learning_rate_robustness():
    """Times to 50% and 80% probe accuracy for DFA, DFA with prior bias and BP across plain-SGD rates. The 1e-3 rows
    reuse the E1 runs with seed pairs (0,0), (1,1), (2,2); BP there has feedback seed 0."""
    out, rows, records = {}, [], []
    for lr in LR_RATES:
        for arm, e1tag, label in LR_ARMS:
            if lr == 0.001:
                files = [E1 / e1tag / (f'seed{s}_fb{s}.csv' if arm != 'bp' else f'seed{s}_fb0.csv') for s in range(3)]
            else:
                files = sorted((RES / 'E21_lr' / f'{arm}_lr{lr:g}').glob('seed*_fb*.csv'))
            files = [f for f in files if f.exists()]
            if not files:
                continue
            g = pd.DataFrame([run_record(f) for f in files])
            records.append(g.assign(lr=lr, rule=arm, run=[f.stem for f in files]))
            ok = g[~g.diverged]
            key = f'lr_{arm}_{lr:g}'.replace('.', 'p')
            out[key + '_diverged'] = int(g.diverged.sum())
            out[key + '_acc50'] = float(ok.acc50.median()) if len(ok) else float('nan')
            out[key + '_acc80'] = float(ok.acc80.median()) if len(ok) and ok.acc80.notna().any() else float('nan')
            out[key + '_accuracy'] = float(ok.accuracy.mean()) if len(ok) else float('nan')
            out[key + '_p3'] = float(ok.p3.mean()) if len(ok) else float('nan')
            rows.append(' & '.join([rf'${lr:g}$', label, f'{len(g)}', f'{int(g.diverged.sum())}',
                                    pm(ok.p3, 4) if len(ok) else '--', steps_text(out[key + '_acc50']),
                                    steps_text(out[key + '_acc80']), pm(ok.accuracy, 3) if len(ok) else '--']) + r'\\')
    pd.concat(records).to_csv(DEST / 'lr_runs.csv', index=False)
    for lr in (0.001, 0.1):
        key = f'{lr:g}'.replace('.', 'p')
        out[f'lr_saving80_{key}'] = 100 * (1 - out[f'lr_prior_{key}_acc80'] / out[f'lr_dfa_{key}_acc80'])
    header = [r'$\eta$', 'Rule', '$n$', 'Diverged', r'Min.\ $p_3$', r'50\% acc.', r'80\% acc.', 'Accuracy']
    TABLES.joinpath('submission_lr.tex').write_text('\n'.join([
        '% Generated by scripts/submission_reporting.py.',
        r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
        r'\caption{Plain-SGD learning rate in the MNIST baseline, with equal hidden and readout rates. Columns give '
        r'diverged runs (non-finite loss, or final loss above the initial loss), the participation minimum within 300 '
        r'updates, the median first update reaching 50\% and 80\% probe accuracy, and accuracy after 3,000 updates, '
        r'over non-diverged runs with seed pairs $(0,0)$, $(1,1)$ and $(2,2)$. The $10^{-3}$ rows reuse the headline '
        r'runs with initialization seeds 0--2 (BP feedback seeds are unused). Prior bias sets all sigmoid biases to '
        r'$\operatorname{logit}(1/C)$. Probes are logged every ten updates at $10^{-3}$ and every two at larger rates. '
        r'Each new run completes 3,000 updates.}\label{tab:submission_lr}',
        r'\begin{tabular}{ll' + 'r' * (len(header) - 2) + '}', r'\toprule', ' & '.join(header) + r'\\\midrule',
        *rows, r'\bottomrule\end{tabular}\end{table}', '']))
    return out


def imbalanced_calibration():
    """Class-imbalanced softmax (p0 = 0.7) with and without a readout calibrated to the sampling prior."""
    out, records = {}, []
    groups = {'imb': (E1 / 'softmax_imb07', [f'seed{s}_fb{s}.csv' for s in range(3)]),
              'imb_bp': (E1 / 'softmax_imb07_bp', [f'seed{s}_fb0.csv' for s in range(3)]),
              'calib': (RES / 'E22_imbcalib' / 'softmax_imb07_calib', None),
              'calib_bp': (RES / 'E22_imbcalib' / 'softmax_imb07_calib_bp', None)}
    for key, (folder, names) in groups.items():
        files = [folder / n for n in names] if names else sorted(folder.glob('seed*_fb*.csv'))
        files = [f for f in files if f.exists()]
        if not files:
            continue
        rec = []
        for f in files:
            d = pd.read_csv(f)
            bal = d.set_index('step').probe_bal_acc
            rec.append(dict(cos3=float(d.cos_l3.max()), p3=float(d[d.step <= 300].p_l3.min()),
                            ebar0=float(d.ebar_norm.iloc[0]), bal450=float(bal.loc[:450].iloc[-1]),
                            bal1000=float(bal.loc[:1000].iloc[-1]), bal_end=float(bal.iloc[-1])))
        g = pd.DataFrame(rec)
        records.append(g.assign(condition=key, run=[f.stem for f in files]))
        for col in g.columns:
            out[f'imbcal_{key}_{col}'] = float(g[col].mean())
    if records:
        pd.concat(records).to_csv(DEST / 'imbalance_calibration_runs.csv', index=False)
    return out


CENTERED = [('base', 'Plain DFA'), ('bp', 'Backpropagation'), ('aligned', 'Pre-aligned DFA'),
            ('priorbias', 'Prior-bias initialization'), ('pixcenter_frzbias', 'Centered inputs + frozen biases'),
            ('center_e', 'Centered output error')]


def centered_geometry(window=500):
    """Top-layer centered variance E||h - hbar||^2 (relative to initialization) and centered effective rank: minima
    within the first `window` updates for each E1 run, then mean and standard deviation across runs."""
    out, body, records = {}, [], []
    for tag, label in CENTERED:
        runs = []
        for f in sorted((E1 / tag).glob('seed*_fb*.csv')):
            if f.name.startswith('reduced_'):
                continue
            d = pd.read_csv(f, usecols=['step', 'Eh2_l3', 'H_l3', 'rank_c_l3'])
            d = d[d.step <= window]
            var = (d.Eh2_l3 - d.H_l3).to_numpy()
            runs.append(dict(var_min=float(np.nanmin(var) / var[0]), rank0=float(d.rank_c_l3.iloc[0]),
                             rank_min=float(np.nanmin(d.rank_c_l3.to_numpy()))))
            records.append(dict(condition=tag, source=f.relative_to(ROOT).as_posix(), window=window, **runs[-1]))
        g = pd.DataFrame(runs)
        out[f'centered_var_min_{tag}'] = float(g.var_min.mean())
        out[f'centered_rank_min_{tag}'] = float(g.rank_min.mean())
        out[f'centered_rank0_{tag}'] = float(g.rank0.mean())
        body.append(' & '.join([label, str(len(g)), pm(g.var_min, 3), pm(g.rank0, 1), pm(g.rank_min, 1)]) + r'\\')
    pd.DataFrame(records).to_csv(DEST / 'centered_geometry_runs.csv', index=False)
    header = ['Condition', '$n$', r'Min.\ variance / initial', 'Initial rank', 'Min.\ rank']
    TABLES.joinpath('submission_centered.tex').write_text('\n'.join([
        '% Generated by scripts/submission_reporting.py.',
        r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
        r'\caption{Centered geometry of top-layer activity in the MNIST baseline cohort. The centered variance is '
        r'$\E_x\|h_3-\hb_3\|^2$, the amplitude of activity differences across inputs, and its minimum within the first '
        rf'{window} updates is given relative to initialization. The centered effective rank is the exponential of the '
        rf'entropy of the normalized centered covariance spectrum; its minimum uses the same {window}-update window. '
        r'Means $\pm$ standard deviations across runs.}'
        r'\label{tab:submission_centered}',
        r'\begin{tabular}{l' + 'r' * (len(header) - 1) + '}', r'\toprule', ' & '.join(header) + r'\\\midrule',
        *body, r'\bottomrule\end{tabular}\end{table}', '']))
    return out


def centering_instability():
    """Top-layer mean-activity energy ||hbar_L||^2 at initialization and at the last update, under error centering with
    raw pixels (eta 3e-4) and with per-pixel input centering (eta 1e-3), for linear, ReLU and GELU units (E10)."""
    out, records = {}, []
    for act in ('linear', 'relu', 'gelu'):
        for tag, key in ((f'{act}_lr3e4_centere', 'raw'), (f'{act}_lr1e3_centere_pixcenter', 'centered')):
            h0, hend, hmax = [], [], []
            for f in sorted((RES / 'E10_nonsat' / tag).glob('seed*_fb*.csv')):
                d = pd.read_csv(f, usecols=['step', 'H_l3'])
                h0.append(float(d.H_l3.iloc[0])); hend.append(float(d.H_l3.iloc[-1])); hmax.append(float(d.H_l3.max()))
                records.append(dict(activation=act, inputs=key, source=f.relative_to(ROOT).as_posix(),
                                    final_step=int(d.step.iloc[-1]), H0=h0[-1], Hend=hend[-1], Hmax=hmax[-1]))
            if h0:
                out[f'instab_{act}_{key}_H0'] = float(np.mean(h0))
                out[f'instab_{act}_{key}_Hend'] = float(np.mean(hend))
                out[f'instab_{act}_{key}_Hmax'] = float(np.mean(hmax))
                out[f'instab_{act}_{key}_Hend_tex'] = power_of_ten(np.mean(hend))
    pd.DataFrame(records).to_csv(DEST / 'centering_instability_runs.csv', index=False)
    return out


def power_of_ten(value):
    """LaTeX math for a positive number: 3.5\\times10^{14}, or the plain value below 1000."""
    if value < 1000:
        return f'{value:.1f}'
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10 ** exponent
    if round(mantissa, 1) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f'{mantissa:.1f}\\times10^{{{exponent}}}'


NOKLAND = [('dfa_zero', 'DFA, zero initialization'), ('bp_fanin', 'BP, fan-in initialization'),
           ('dfa_xavier', 'DFA, Xavier initialization')]


def nokland():
    """Collapse and learning-time statistics for every E19 run, a generated table and summary keys."""
    rows, out = [], {}
    exp = RES / 'E19_nokland'
    for f in sorted(exp.glob('*/seed*_fb*.csv')):
        d = pd.read_csv(f)
        m = json.loads(f.with_suffix('.meta.json').read_text())
        chance = float(np.max(m['prior'])) + .05
        left = d[(d.step > 0) & (d.probe_acc > chance)]
        rows.append(dict(tag=m['tag'], lr=m['lr'], seed=m['seed'], cos3=float(d.cos_l3.max()),
                         sat3=float(d.sat90_l3.max()), chi3=float(d.chi_l3.min()),
                         p3=float(d[d.step <= 300].p_l3.min()),
                         chance=float(left.step.iloc[0]) if len(left) else np.nan,
                         time=sustained_crossing(d, .9 * m['prior_loss']),
                         acc80=first_step(d, 'probe_acc', .8), accuracy=float(d.probe_acc.iloc[-1])))
    if not rows:
        return out
    r = pd.DataFrame(rows)
    r.to_csv(DEST / 'nokland_runs.csv', index=False)
    body = []
    for lr in sorted(r.lr.unique(), reverse=True):
        for arm, label in NOKLAND:
            g = r[r.tag == f'{arm}_lr{lr:g}']
            if not len(g):
                continue
            key = f'nokland_{arm}_{"fast" if lr >= 1e-3 else "slow"}'
            for col in ('cos3', 'sat3', 'chi3', 'p3', 'chance', 'time', 'acc80', 'accuracy'):
                out[f'{key}_{col}'] = float(g[col].mean())
            body.append(' & '.join([label, rf'$10^{{{int(np.round(np.log10(lr)))}}}$', str(len(g)), pm(g.cos3, 3),
                                    pm(g.sat3, 2), pm(g.chance, 0), pm(g.time, 0), pm(g.acc80, 0),
                                    pm(g.accuracy, 3)]) + r'\\')
    header = ['Protocol', 'Rate', '$n$', r'Peak $\cos_3$', 'Saturated', 'Leaves chance', 'Learning time',
              r'80\% acc.', 'Accuracy']
    TABLES.joinpath('submission_nokland.tex').write_text('\n'.join([
        '% Generated by scripts/submission_reporting.py.',
        r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}',
        r'\caption{Reconstruction of the original DFA protocol \citep{nokland2016}: $3\times800$ tanh MLP, logistic outputs with binary '
        r'cross-entropy, inputs in $[0,1]$, RMSprop and minibatches of 64, with zero initialization and uniform feedback '
        r'for DFA and fan-in initialization for BP; the last rows keep Xavier initialization and Gaussian feedback. '
        r'Saturated is the largest fraction of top-layer units with $|\bar h_{3i}|>0.9$. Leaves chance is the first '
        r'update with probe accuracy above the majority-class rate plus 0.05; learning time uses the criterion of '
        r'Table~\ref{tab:review_timing}; the 80\% column is the first update reaching that accuracy; the last column is '
        r'accuracy after 6,000 updates. At zero initialization, layers above the first are input-independent until their '
        r'weights grow, so their peak cosine is one by construction. Means $\pm$ standard deviations over three seed '
        r'pairs.}\label{tab:submission_nokland}',
        r'\begin{tabular}{ll' + 'r' * (len(header) - 2) + '}', r'\toprule', ' & '.join(header) + r'\\\midrule',
        *body, r'\bottomrule\end{tabular}\end{table}', '']))
    return out


if __name__ == '__main__':
    main()
