"""Checks of the scientific identity and the executable measurement pipeline."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("measure_residual", ROOT / "scripts" / "measure_residual.py")
residual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(residual)


def test_mean_and_full_update_decomposition():
    g = torch.Generator().manual_seed(71)
    error = torch.randn(31, 4, generator=g, dtype=torch.float64) + 0.4
    gamma = torch.sigmoid(torch.randn(31, 7, generator=g, dtype=torch.float64))
    feedback = torch.randn(7, 4, generator=g, dtype=torch.float64)
    h = torch.randn(31, 5, generator=g, dtype=torch.float64) + 2.0
    teacher, lead, r = residual.teaching_parts(gamma, error, feedback)
    delta = gamma * (error @ feedback.T)
    cov = (delta - delta.mean(0)).T @ (h - h.mean(0)) / len(h)
    assert torch.allclose(teacher, lead + r, atol=1e-14, rtol=1e-14)
    assert torch.allclose(delta.T @ h / len(h), cov + torch.outer(lead + r, h.mean(0)), atol=1e-14, rtol=1e-14)


def test_centered_error_can_leave_a_residual_but_constant_gates_do_not():
    error = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    feedback = torch.ones(1, 1, dtype=torch.float64)
    teacher, lead, r = residual.teaching_parts(torch.tensor([[1.0], [0.25]]), error, feedback)
    assert lead.item() == 0.0
    assert r.item() == teacher.item() == 0.375
    _, _, r_constant = residual.teaching_parts(torch.ones(2, 1), error, feedback)
    assert r_constant.item() == 0.0


def synthetic_data():
    g = torch.Generator().manual_seed(42)
    return dict(Xtr=torch.rand(60, 5, generator=g), ytr=torch.arange(60) % 3,
                Xval=torch.rand(30, 5, generator=g), yval=torch.arange(30) % 3, C=3)


def test_measurement_matches_runner_and_records_seed_provenance(monkeypatch, tmp_path):
    from cmc import run
    data = synthetic_data()
    monkeypatch.setattr(residual.D, "load", lambda *args, **kwargs: data)
    cfg = run.Config(seed=3, fb_seed=7, width=8, depth=2, batch=16, probe_n=24,
                     steps=2, threads=1, log_every=1, heavy_every=1,
                     save_snapshots=0, out_root=str(tmp_path))
    frame, metadata = residual.run_condition("base", cfg, sample_steps=(0, 1, 2))
    reference, _ = run.main(cfg)
    measured = frame.groupby("step").ebar_norm.first().to_numpy()
    assert np.allclose(measured, [row["ebar_norm"] for row in reference], atol=1e-7, rtol=1e-7)
    assert frame.decomposition_error.max() < 1e-6
    assert set(frame.seed) == {3} and set(frame.fb_seed) == {7}
    assert metadata["probe_seed"] == 1002
    assert metadata["config"]["fb_seed"] == 7
    repeat, repeat_meta = residual.run_condition("base", cfg, sample_steps=(0, 1, 2))
    pd.testing.assert_frame_equal(frame, repeat)
    assert metadata == repeat_meta


def test_command_writes_measurements_summary_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(residual.D, "load", lambda *args, **kwargs: synthetic_data())
    monkeypatch.setattr(residual, "CONFIGS", {"base": {"width": 8, "depth": 2, "probe_n": 24}})
    monkeypatch.setattr(residual, "SAMPLE_STEPS", (0, 1, 2))
    assert residual.main(["--output-dir", str(tmp_path), "--threads", "1"]) == 0
    frame = pd.read_csv(tmp_path / "residual_ratio.csv")
    summary = json.loads((tmp_path / "residual_ratio.json").read_text())
    metadata = json.loads((tmp_path / "residual_ratio.meta.json").read_text())
    assert len(frame) == 6 and set(frame.step) == {0, 1, 2}
    assert np.isclose(summary["base_init_max"], frame.loc[frame.step.eq(0), "ratio"].max(), atol=1e-14, rtol=1e-14)
    assert len(summary["base_init_by_layer"]) == 2
    assert metadata["runs"]["base"]["config"]["seed"] == 0
    assert metadata["runs"]["base"]["sample_steps"] == [0, 1, 2]
    assert "scripts/measure_residual.py" in metadata["source_sha256"]
