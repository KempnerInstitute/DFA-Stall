"""Tests for the teacher-student ladder: update equivalence with cmc.rules.step and the plateau detector."""
import numpy as np
import pandas as pd

from cmc.ladder import LadderConfig, Net, grads, rungs, selfcheck, train
from cmc.ladder_analysis import classify, find_plateaus


def test_dfa_update_matches_rules_step():
    dev = selfcheck()
    assert max(dev) < 1e-5


def test_bp_grad_matches_finite_difference():
    cfg = LadderConfig(D=8, C=3, widths="5,4", student_act="tanh", head="sigmoid_bce", bias=1)
    rng = np.random.default_rng(0)
    net = Net(cfg, rng)
    X = rng.standard_normal((6, cfg.D))
    T = np.eye(cfg.C)[rng.integers(0, cfg.C, 6)]
    loss, dW, _, _, _, _, _ = grads(net, cfg, X, T, None, "bp")
    eps, i, j = 1e-6, 2, 3
    net.W[0][i, j] += eps
    lp, *_ = grads(net, cfg, X, T, None, "bp")
    net.W[0][i, j] -= 2 * eps
    lm, *_ = grads(net, cfg, X, T, None, "bp")
    assert abs((lp - lm) / (2 * eps) - dW[0][i, j]) < 1e-5


def test_plateau_detector_finds_a_flat_window():
    step = np.unique(np.geomspace(1, 1e5, 300).astype(int))
    loss = np.where(step < 300, 3.0 - 1.0 * step / 300, np.where(step < 20000, 2.0, 2.0 * (20000 / step) ** 0.6))
    df = pd.DataFrame(dict(step=step, t=step / 500.0, test_loss=loss, const_loss=2.0))
    wins = [w for w in find_plateaus(df) if w["interior"]]
    assert len(wins) == 1
    w = wins[0]
    assert 250 <= w["step0"] <= 1500 and 12000 <= w["step1"] <= 25000
    assert abs(w["loss"] - 2.0) < 0.05 and w["drop_after"] > 0.5


def test_classify_separates_the_two_signatures():
    ours = dict(in_prior_band=True, interior=True, loss_over_const=1.0, wa_ratio=0.1, hcos_plateau=0.99)
    theirs = dict(in_prior_band=False, interior=True, loss_over_const=0.1, wa_ratio=0.97, hcos_plateau=0.0)
    assert classify(ours) == "common-mode collapse"
    assert classify(theirs) == "Refinetti post-alignment"


def test_rung0_runs_and_stays_finite():
    kw = dict(rungs()["rung0_scalar_K2M2_dfa"])
    kw.update(steps=2000, probe_n=256, n_log=40, seed=0)
    rows, meta = train(LadderConfig(**{k: v for k, v in kw.items() if k != "rung"}, rung="t"))
    assert np.isfinite([r["test_loss"] for r in rows]).all()
    assert meta["const_loss"] > 0
