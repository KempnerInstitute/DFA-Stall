import math, os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmc.models import MLP, act_deriv
from cmc.rules import head_forward, make_feedback, step
from cmc.run import Config
from cmc.analysis import plateau, ebar0_closed_form
from cmc.diagnostics import participation

def test_rowwise_alignment_identity():
    M = torch.ones(3, 2); assert abs(F.cosine_similarity(M, M, dim=1).mean().item() - 1.0) < 1e-6

def test_ebar0_closed_form_sigmoid():
    torch.manual_seed(0); C = 10; cfg = Config(head="sigmoid_bce")
    z = torch.zeros(4096, C); y = torch.randint(0, C, (4096,))
    _, p, e, t = head_forward(z, y, cfg, C); meas = e.mean(0).norm().item()
    assert abs(meas - ebar0_closed_form("sigmoid_bce", C)) < 0.03 * ebar0_closed_form("sigmoid_bce", C) + 0.02

def test_ebar0_softmax_balanced_is_zero():
    C = 10; cfg = Config(head="softmax_ce"); z = torch.zeros(10000, C); y = torch.arange(10000) % C
    _, p, e, t = head_forward(z, y, cfg, C); assert e.mean(0).norm().item() < 1e-6

def test_bp_rule_matches_autograd_and_dfa_teaching_formula():
    torch.manual_seed(0); C = 10; x = torch.randn(64, 20); y = torch.randint(0, C, (64,))
    cfg = Config(rule="dfa", width=16, depth=2, head="sigmoid_bce", lr=0.0)
    m = MLP(20, [16, 16], C, act="tanh"); fb = make_feedback(m, C, cfg, "cpu")
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    res = step(m, x, y, C, cfg, fb, opt, state={})
    # explicit replica of the DFA teaching signal for layer 1
    with torch.no_grad():
        z, pres, acts = m.forward_blocks(x, detach=False); _, p, e, t = head_forward(z, y, cfg, C)
        delta1 = (e @ fb["B"][0].t()) * (1 - acts[0] ** 2)
        gw_expected = delta1.t() @ x / 64
    assert torch.allclose(m.hidden[0].weight.grad, gw_expected, atol=1e-6)
    # BP rule equals autograd of the mean loss
    cfg_bp = Config(rule="bp", width=16, depth=2, head="sigmoid_bce", lr=0.0); m2 = MLP(20, [16, 16], C); opt2 = torch.optim.SGD(m2.parameters(), lr=0.0)
    step(m2, x, y, C, cfg_bp, {}, opt2, state={}); g1 = m2.hidden[0].weight.grad.clone()
    for p_ in m2.parameters(): p_.grad = None
    z, _, _ = m2.forward_blocks(x, detach=False); lv, *_ = head_forward(z, y, cfg_bp, C); lv.mean().backward()
    assert torch.allclose(g1, m2.hidden[0].weight.grad, atol=1e-6)

def test_projected_step_of_bp_is_one():
    g = torch.randn(30, 20); assert abs(((g * g).sum() / g.norm() ** 2).item() - 1.0) < 1e-6

def test_participation_bounds():
    assert abs(participation(torch.ones(300)) - 1.0) < 1e-9
    u = torch.zeros(300); u[:30] = 1.0; assert abs(participation(u) - 0.1) < 1e-6

def test_plateau_detector():
    import pandas as pd
    steps = np.arange(0, 1510, 10); L = np.where(steps < 100, 7 - 0.04 * steps, 3.25); L = np.where(steps > 500, 3.25 - 0.004 * (steps - 500), L)
    acc = np.where((steps >= 100) & (steps <= 500), 0.11, 0.6)
    df = pd.DataFrame(dict(step=steps, probe_loss=L, probe_acc=acc)); pl = plateau(df, 3.2508, [0.1] * 10)
    assert pl is not None and 90 <= pl["onset"] <= 110 and pl["exit"] is not None and 550 <= pl["exit"] <= 650

def test_centered_delta_has_zero_batch_mean():
    torch.manual_seed(1); C = 10; x = torch.randn(64, 20); y = torch.randint(0, C, (64,))
    cfg = Config(rule="dfa", width=16, depth=2, center="delta", lr=0.0); m = MLP(20, [16, 16], C); fb = make_feedback(m, C, cfg, "cpu")
    res = step(m, x, y, C, cfg, fb, torch.optim.SGD(m.parameters(), lr=0.0), state={})
    assert res["deltas"][0].mean(0).abs().max().item() < 1e-6

def test_nokland_protocol_options():
    """Zero and fan-in initializations, fan-out-scaled uniform feedback and RMSprop (Nokland, 2016). At zero
    initialization deeper layers receive no weight gradient, and their bias gradient is exactly B_l ebar."""
    from cmc.run import apply_init, make_optimizer
    torch.manual_seed(0); C = 10
    m = MLP(20, [16, 8], C, act="tanh"); apply_init(m, "uniform_fanin")
    for layer in list(m.hidden) + [m.out]:
        bound = 1 / math.sqrt(layer.in_features)
        assert float(layer.weight.detach().abs().max()) <= bound and float(layer.bias.detach().abs().max()) <= bound
    cfg = Config(rule="dfa", fb_dist="uniform_fanout", optimizer="rmsprop", head="sigmoid_bce", lr=0.0)
    fb = make_feedback(m, C, cfg, "cpu")
    for b, w in zip(fb["B"], m.widths):
        assert b.shape == (w, C) and float(b.abs().max()) <= 1 / math.sqrt(w)
    assert isinstance(make_optimizer(m, cfg), torch.optim.RMSprop)
    apply_init(m, "zero")
    assert all(float(p.detach().abs().max()) == 0.0 for p in m.parameters())
    x = torch.rand(32, 20); y = torch.randint(0, C, (32,))
    step(m, x, y, C, cfg, fb, torch.optim.SGD(m.parameters(), lr=0.0), state={})
    ebar = (0.5 - F.one_hot(y, C).float()).mean(0)
    assert float(m.hidden[1].weight.grad.abs().max()) == 0.0
    assert torch.allclose(m.hidden[1].bias.grad, fb["B"][1] @ ebar, atol=1e-6)

def test_sign_feedback_teaching_signal():
    """Sign-error feedback: with sigmoid outputs sign(e) = 1 - 2y for every input, and it replaces e in the hidden
    teaching signal while the readout keeps its true gradient."""
    torch.manual_seed(0); C = 10; x = torch.randn(64, 20); y = torch.randint(0, C, (64,))
    cfg = Config(rule="dfa", fb_signal="sign", head="sigmoid_bce", lr=0.0)
    m = MLP(20, [16, 16], C, act="tanh"); fb = make_feedback(m, C, cfg, "cpu")
    step(m, x, y, C, cfg, fb, torch.optim.SGD(m.parameters(), lr=0.0), state={})
    with torch.no_grad():
        z, pres, acts = m.forward_blocks(x, detach=False); _, p, e, t = head_forward(z, y, cfg, C)
        s = torch.sign(e)
        assert torch.equal(s, 1 - 2 * F.one_hot(y, C).float())
        delta1 = (s @ fb["B"][0].t()) * (1 - acts[0] ** 2)
    assert torch.allclose(m.hidden[0].weight.grad, delta1.t() @ x / 64, atol=1e-6)
    assert torch.allclose(m.out.bias.grad, e.mean(0), atol=1e-6)
