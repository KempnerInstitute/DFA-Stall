"""Unit tests for the reduced common-mode collapse model: structural invariances that the theory requires."""
import math, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmc.models import MLP
from cmc.reduced import (Init, ReducedConfig, extract_init, gate_moments, kappa_closed_form, participation,
                         sigma_prime_eff, simulate, summarize)

D, L, C = 150, 3, 10

def make_init(q=0.5, mu_spread=0.24, H0=35.0, seed=0, prior_loss=None):
    """A synthetic initial condition with the baseline's numbers: d=150 x 3 tanh, C=10, ||xbar||^2 = 35."""
    rng = np.random.default_rng(seed)
    pi = np.full(C, 1.0 / C)
    pl = prior_loss if prior_loss is not None else float(-(pi * np.log(pi) + (1 - pi) * np.log(1 - pi)).sum())
    return Init(H0=H0, mu0=[rng.standard_normal(D) * mu_spread for _ in range(L)], sigma0=[0.31] * L,
                zbar0=np.full(C, math.log(q / (1 - q))), q=[1.44] + [1.0] * (L - 1), r=[1.0] * L, Lam0=1.1,
                lam0=[0.5] * L, widths=[D] * L, C=C, tbar=pi, sigma_x2=0.067, prior_loss=pl)

def make_B(g=1.0, seed=3):
    rng = np.random.default_rng(seed)
    return [g * rng.standard_normal((D, C)) for _ in range(L)]

def run(eta=1e-3, g=1.0, q=0.5, steps=600, substeps=1, escape=False, **kw):
    init = make_init(q=q, **{k: v for k, v in kw.items() if k in ("mu_spread", "H0", "seed")})
    cfg = ReducedConfig(eta_hid=eta, eta_out=eta, g=g, steps=steps, substeps=substeps, escape=escape,
                        **{k: v for k, v in kw.items()
                           if k in ("bias_channel", "drive_scale", "freeze_out", "onset_min_len")})
    return simulate(init, cfg, make_B(g))

def minp(df): return [float(df[f"p_l{l+1}"].min()) for l in range(L)]

# ---------------- quadrature and closed forms ----------------
def test_participation_and_gate_quadrature():
    hb, ga, u = gate_moments(np.zeros(8), 0.0)
    assert np.allclose(hb, 0.0) and np.allclose(ga, 1.0) and np.allclose(u, 1.0)
    assert abs(participation(u) - 1.0) < 1e-12
    hb, ga, u = gate_moments(np.array([0.0]), 1.0)          # E sech^2 under N(0,1), by direct quadrature
    x, w = np.polynomial.hermite.hermgauss(200)
    ref = ((1 - np.tanh(math.sqrt(2) * x) ** 2) * w).sum() / math.sqrt(math.pi)
    assert abs(float(ga[0]) - ref) < 1e-7

def test_sigma_prime_eff_is_theory_a_constant():
    assert abs(sigma_prime_eff(10) - 0.182) < 5e-4          # (1/2 - 1/C)/|logit(1/C)|, judge adjudication D1
    assert sigma_prime_eff(10) > 2.0 * 0.09                 # and it is twice pi(1-pi) at C = 10

def test_closed_form_kappa_matches_the_integrated_dose():
    """The exact dose integral (sigma'_eff) must reproduce the online dose kappa_l(t) the network logs."""
    df = run(steps=1200)
    for l in range(L):
        cf, dose = df.attrs["kappa_cf"][l], float(df[f"kappa_l{l+1}"].iloc[-1])
        assert abs(cf - dose) / dose < 0.05

def test_closed_form_kappa_matches_the_dose_when_the_mean_error_rotates():
    """A softmax head under an imbalanced prior turns ebar away from ehat_0 as it decays.

    Both the dose and its closed form are projections on the fixed direction ehat_0, so they must still agree;
    using the norm of zbar(0) - zstar instead of its component along ehat_0 breaks this by several fold."""
    rng = np.random.default_rng(1)
    pi = np.array([0.4] + [0.6 / (C - 1)] * (C - 1))
    init = make_init(prior_loss=float(-(pi * np.log(pi)).sum()))
    init.tbar = pi; init.zbar0 = rng.standard_normal(C) * 0.3
    cfg = ReducedConfig(eta_hid=1e-3, eta_out=3e-2, g=1.0, head="softmax_ce", steps=4000, escape=False)
    df = simulate(init, cfg, make_B())
    assert float(df.ebar_norm.iloc[-1]) < 1e-3 * float(df.ebar_norm.iloc[0])      # the head has absorbed the prior
    for l in range(L):
        cf, dose = df.attrs["kappa_cf"][l], float(df[f"kappa_l{l+1}"].iloc[-1])
        assert abs(cf - dose) / abs(dose) < 0.05, (l, cf, dose)

# ---------------- the invariances ----------------
def test_depth_is_independent_of_the_learning_rate():
    """eta multiplies the drift and the prior clock alike, so collapse depth is eta-free (duration is not).

    Time is measured in units of 1/eta (steps ∝ 1/eta) and the Euler sub-step is held at eta*dt = 3e-4 so that
    the comparison is between trajectories of the ODE, not between discretizations of it."""
    out = {}
    for eta in (3e-4, 1e-3, 3e-3):
        steps = int(round(0.6 / eta)); sub = max(1, int(round(eta / 3e-4)))
        out[eta] = minp(run(eta=eta, steps=steps, substeps=sub))
    ref = out[1e-3]
    for eta, v in out.items():
        for a, b in zip(v, ref):
            assert abs(a - b) / b < 0.01, f"eta={eta}: min p {v} vs {ref}"

def test_no_drive_when_the_mean_error_vanishes():
    """q = 1/C makes ebar_0 = 0 exactly: no drift, so participation never leaves its initial value."""
    df = run(q=1.0 / C, steps=400)
    for l in range(L):
        p = df[f"p_l{l+1}"].values
        assert abs(p.max() - p.min()) < 1e-9
        assert abs(float(df[f"kappa_l{l+1}"].iloc[-1])) < 1e-12
        assert abs(float(df[f"mu_spread_l{l+1}"].iloc[-1]) - float(df[f"mu_spread_l{l+1}"].iloc[0])) < 1e-12

def test_no_drive_when_presynaptic_means_and_the_bias_channel_vanish():
    """H_0 = 0 with zero-mean units and frozen hidden biases kills both factors of the common-mode term."""
    init = make_init(); init.H0 = 0.0; init.mu0 = [np.zeros(D) for _ in range(L)]
    cfg = ReducedConfig(eta_hid=1e-3, eta_out=1e-3, g=1.0, steps=400, escape=False, bias_channel=0.0)
    df = simulate(init, cfg, make_B())
    for l in range(L):
        assert abs(float(df[f"p_l{l+1}"].min()) - 1.0) < 1e-9
        assert abs(float(df[f"H_l{l+1}"].max())) < 1e-12

def test_depth_is_monotone_in_the_feedback_gain():
    depth = [minp(run(g=g, steps=900))[L - 1] for g in (0.1, 0.3, 1.0, 3.0)]
    assert all(a > b for a, b in zip(depth, depth[1:])), depth

def test_depth_is_monotone_in_the_mean_error():
    """|q - 1/C| is ||ebar_0||/sqrt(C) for the one-versus-rest sigmoid head."""
    depth = [minp(run(q=q, steps=900))[L - 1] for q in (0.12, 0.2, 0.35, 0.5, 0.7)]
    assert all(a > b for a, b in zip(depth, depth[1:])), depth

# ---------------- the escape state ----------------
def test_escape_produces_an_exit_and_the_collapse_only_model_does_not():
    with_escape = run(steps=2000, escape=True)
    without = run(steps=2000, escape=False)
    assert with_escape.attrs["onset"] is not None and with_escape.attrs["exit"] is not None
    assert without.attrs["exit"] is None
    assert with_escape.attrs["exit"] > with_escape.attrs["onset"]
    lam = with_escape["Lambda_l3"].values                    # starvation first, cascade growth afterwards
    assert lam.min() < 0.3 * lam[0] and lam[-1] > lam.min()

def test_frozen_head_makes_the_collapse_unbounded():
    """With the head frozen the mean error never decays, so the dose diverges (the classical Hebbian runaway)."""
    free = run(steps=900); frozen = run(steps=900, freeze_out=True)
    assert frozen[f"kappa_l{L}"].iloc[-1] > 3.0 * free[f"kappa_l{L}"].iloc[-1]
    assert minp(frozen)[L - 1] < minp(free)[L - 1]
    assert all(math.isinf(k) for k in frozen.attrs["kappa_cf"])      # no finite asymptotic dose to report
    assert all(math.isfinite(k) for k in free.attrs["kappa_cf"])

def test_onset_is_the_start_of_a_band_that_is_actually_held():
    """notes/PROTOCOL.md (and cmc.analysis.plateau) ask for >= 50 steps inside the band, not a single touch."""
    df = run(steps=2000, escape=True)
    onset, Lprior = df.attrs["onset"], df.attrs["prior_loss"]
    assert onset is not None
    held = df[(df.step >= onset) & ((df.loss_model - Lprior).abs() / Lprior < 0.03)]
    assert float(held.step.max()) - onset >= 50
    strict = run(steps=2000, escape=True, onset_min_len=10 ** 6)     # a band that is never held that long
    assert strict.attrs["onset"] is None and strict.attrs["exit"] is None

# ---------------- reading the initial state off a real network ----------------
def test_extract_init_reproduces_the_networks_own_statistics():
    torch.manual_seed(0)
    model = MLP(20, [16, 16], 5, act="tanh")
    X = torch.rand(512, 20); y = torch.randint(0, 5, (512,))
    init = extract_init(model, X, y, 5)
    with torch.no_grad():
        z, pres, acts = model.forward_blocks(X, detach=False)
    assert abs(init.H0 - float((X.mean(0) ** 2).sum())) < 1e-5
    assert np.allclose(init.mu0[0], pres[0].mean(0).numpy(), atol=1e-6)
    assert abs(init.sigma0[1] - float(pres[1].std(0).mean())) < 1e-6
    assert np.allclose(init.zbar0, z.mean(0).numpy(), atol=1e-5)
    wvar = float(model.hidden[0].weight.detach().var())
    assert abs(init.q[0] - 20 * wvar) < 1e-6 and abs(init.r[0] - 16 * wvar) < 1e-6

def test_summary_reports_the_closed_form_and_integrated_dose():
    s = summarize(run(steps=600))
    assert set(s) >= {"minp_l1", "maxcos_l3", "kappa_l2", "kappa_cf_l2", "onset", "exit"}
    assert 0.0 < s["minp_l3"] < 1.0 and 0.0 < s["maxcos_l3"] <= 1.0
    assert abs(kappa_closed_form(1.0, 1.2649, 35.0, 215.0, 10) - 1.2649 * 36.0 / (0.182 * 216.0)) < 1e-3
