"""Physical training conditions, independent of sweep labels and logging."""
from dataclasses import asdict
import json
from .run import Config

IGNORED = {"exp", "tag", "out_root", "seed", "fb_seed", "steps", "log_every", "heavy_every",
           "probe_n", "snapshots", "device", "threads", "save_snapshots", "anatomy", "measure_updates"}


def condition_key(meta, include_seeds=False):
    values=asdict(Config())
    values.update({k:v for k,v in meta.items() if k in values})
    values={k:v for k,v in values.items() if k not in IGNORED}
    values["classes"]=meta.get("C", values["classes"])
    if values["out_bias_q"]==.5 or values["head"]=="softmax_ce":values["out_bias_q"]=-1.
    if values["center"]!="gated":values.pop("gate_thr")
    if not values["whiten"]:values.pop("whiten_lam")
    if values["head"]!="multilabel":values.pop("extra_bits")
    if not values["shift_step"]:values.pop("shift_p0")
    if include_seeds:values.update(seed=meta["seed"],fb_seed=meta["fb_seed"])
    return json.dumps(values,sort_keys=True)
