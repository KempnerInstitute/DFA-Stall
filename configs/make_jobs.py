#!/usr/bin/env python3
"""Generate job files (one `cmc.run` argument line per run) for each experiment family."""
import itertools, os
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS_HEAD = [(s, f) for s in range(5) for f in range(3)]        # 5 inits x 3 feedback draws = 15 trajectories
SEEDS_3 = [(s, s) for s in range(3)]                             # 3 (init, feedback) pairs
def w(name, lines):
    p = os.path.join(HERE, name); open(p, "w").write("\n".join(lines) + "\n"); print(f"{name}: {len(lines)} jobs")
def job(exp, tag, s, f, **kw):
    parts = [f"--exp {exp} --tag {tag} --seed {s} --fb_seed {f}"] + [f"--{k} {v}" for k, v in kw.items()]
    return " ".join(parts)

# ---- E1 headline: 15 trajectories per condition, 3000 steps
E1 = {
 "base": {}, "bp": dict(rule="bp"), "fa": dict(rule="fa"), "aligned": dict(rule="aligned"),
 "center_delta": dict(center="delta"), "center_e": dict(center="e"), "gated": dict(center="gated"),
 "pixcenter": dict(preprocess="pixcenter"), "pixcenter_frzbias": dict(preprocess="pixcenter", freeze_bias=1), "frzbias": dict(freeze_bias=1),
 "priorbias": dict(out_bias_q=0.1), "outlr10": dict(out_lr_mult=10), "frzout": dict(freeze_out=1, steps=1500),
 "softmax": dict(head="softmax_ce", preprocess="standardize"), "softmax_imb07": dict(head="softmax_ce", preprocess="standardize", p0=0.7),
 "softmax_imb07_bp": dict(head="softmax_ce", preprocess="standardize", p0=0.7, rule="bp"), "bp_outx12": dict(rule="bp", out_w_scale=12),
}
lines = []
for tag, kw in E1.items():
    kw = dict(kw); steps = kw.pop("steps", 3000)
    for s, f in SEEDS_HEAD:
        if kw.get("rule") == "bp" and f > 0: continue          # BP has no feedback draw
        lines.append(job("E1_headline", tag, s, f, steps=steps, **kw))
w("E1_headline.txt", lines)

# ---- E2 master sweep: 3 pairs each; depth measures; 1500 steps (3000 for slow)
lines = []
for g in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"g{g}", s, f, fb_scale=g, steps=(3000 if g < 0.3 else 1500)))
for q in [0.1, 0.2, 0.35, 0.5, 0.7, 0.85]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"q{q}", s, f, out_bias_q=q, steps=1500))
for lr in [3e-4, 1e-3, 3e-3]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"lr{lr}", s, f, lr=lr, steps=int(1500 * 1e-3 / lr)))
for d in [100, 300, 900, 2000]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"w{d}", s, f, width=d, steps=1500))
for C in [2, 3, 5, 10]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"C{C}", s, f, classes=C, steps=1500))
for b in [32, 128, 512, 2048]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"b{b}", s, f, batch=b, steps=1500))
for pre in ["div255", "standardize", "pixcenter", "pixstd"]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"pre_{pre}", s, f, preprocess=pre, steps=1500))
for rho in [0.3, 1.0, 3.0, 10.0]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"outlr{rho}", s, f, out_lr_mult=rho, steps=1500))
for hm in [0.25, 1.0, 4.0]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"hidlr{hm}", s, f, hid_lr_mult=hm, steps=int(1500 / min(hm, 1))))
for p0 in [0.0, 0.3, 0.5, 0.7, 0.9]:
    for s, f in SEEDS_3: lines.append(job("E2_master", f"smimb{p0}", s, f, head="softmax_ce", preprocess="standardize", p0=p0, steps=1500))
w("E2_master.txt", lines)

# ---- E3 generality: 3 pairs; with BP twin and cure per family
lines = []
fam = {
 "depth6": dict(depth=6), "relu": dict(act="relu"), "relu_lr3e4": dict(act="relu", lr=3e-4, steps=3000), "gelu": dict(act="gelu"), "linear": dict(act="linear"),
 "cnn": dict(arch="cnn", steps=1500), "multilabel": dict(head="multilabel"), "mse_s1": dict(head="mse", target_scale=1.0), "mse_s1_ctr": dict(head="mse", target_scale=1.0, target_center=1),
 "mse_s5": dict(head="mse", target_scale=5.0), "fashion": dict(dataset="fashion"), "fashion_softmax": dict(dataset="fashion", head="softmax_ce", preprocess="standardize"),
 "fa": dict(rule="fa"), "fa_depth6": dict(rule="fa", depth=6), "w900_depth6": dict(width=900, depth=6),
}
for tag, kw in fam.items():
    steps = kw.pop("steps", 2000)
    for s, f in SEEDS_3:
        lines.append(job("E3_generality", tag, s, f, steps=steps, **kw))
        lines.append(job("E3_generality", tag + "_center", s, f, steps=steps, center="delta", **kw))
        if kw.get("rule") != "fa" and f == 0: lines.append(job("E3_generality", tag + "_bp", s, 0, steps=steps, rule="bp", **{k: v for k, v in kw.items() if k != "rule"}))
w("E3_generality.txt", lines)

# ---- E4 cures bake-off (3 pairs, 3000 steps, base protocol and two hard settings)
lines = []
cures = {"none": {}, "center_delta": dict(center="delta"), "center_e": dict(center="e"), "gated": dict(center="gated"), "priorbias": dict(out_bias_q=0.1),
         "outlr10": dict(out_lr_mult=10), "muon": dict(muon=1), "bn": dict(batchnorm=1, center="none"), "whiten": dict(whiten=1), "aligned": dict(rule="aligned")}
settings = {"base": {}, "depth6": dict(depth=6), "cnn": dict(arch="cnn", steps=1500)}
for sname, skw in settings.items():
    for cname, ckw in cures.items():
        if sname == "cnn" and cname in ("whiten", "bn", "muon"): continue
        for s, f in SEEDS_3:
            kw = dict(skw); kw.update(ckw); steps = kw.pop("steps", 3000)
            lines.append(job("E4_cures", f"{sname}__{cname}", s, f, steps=steps, **kw))
w("E4_cures.txt", lines)

# ---- E5 recovery-law arbitration: decoupled hidden lr x output lr x gain, 2 pairs
lines = []
for hm, om, g in itertools.product([0.25, 1.0, 4.0], [0.3, 1.0, 3.0, 10.0], [0.3, 1.0, 3.0]):
    for s, f in SEEDS_3[:2]:
        steps = int(min(6000, max(1500, 3000 / (hm * om * g) ** 0.5 * 1.5)))
        lines.append(job("E5_recovery", f"h{hm}_o{om}_g{g}", s, f, hid_lr_mult=hm, out_lr_mult=om, fb_scale=g, steps=steps))
w("E5_recovery.txt", lines)
