# Paper code release, 24 September 2026

This release contains the implementation and analyses used in **Common-Mode
Collapse and Recovery in Direct Feedback Alignment**, by Varun Reddy, Houman
Safaai and Bernardo L. Sabatini. The project grew out of the original DFA stall
experiments; this release records the pipeline used for the revised paper.

The code covers the baseline and causal controls, reduced model, optimizer and
architecture comparisons, activation and feedback controls, readout calibration,
learning-rate sweeps, class-prior shifts, sign-feedback tests, and teacher–student
comparison. It retains the original experiment identifiers so records can be
matched to the manuscript and reproduction instructions.

## What is included

- Training and analysis code, complete available configurations and seed choices.
- Tests of numerical identities, learning rules, diagnostics and reporting.
- Run metadata, compact summaries, source mappings and numerical validation inputs.
- Selected CIFAR trajectories needed to reproduce and cross-check the geometry
  figure and table without retraining.

## What must be regenerated

Public image datasets must be downloaded. Full training trajectories and unit
snapshots must be regenerated for analyses that depend on them. The release does
not promise that every figure can be rebuilt from compact summaries alone.

The current checkout excludes manuscript sources, PDFs, rendered figures,
presentations, internal review notes and cluster-specific launchers. It starts
from the validated code supplement, with a portable data directory and public
documentation. Scientific Python modules, experiment configurations, tests and
reference records are unchanged from that supplement. `release.json` identifies
the exact source archive and revision.

## Original project history

The repository also retains Varun Reddy's seven commits, ending at
`8c21518db68b6beec360a196fa7e12d581af827d`. Their authorship, timestamps and code
changes are preserved. Automated co-author trailers were removed from the commit
messages, changing the commit IDs while leaving Varun as their sole credited
author. A merge connects this history to `main` while retaining the release
files exactly. The original implementation and
its figures remain accessible in those historical revisions. The private
manuscript development history is not included.

## Validation

Run `python scripts/reproduce_submission.py` on a fresh checkout. This validates
the file manifest and runs the 50 numerical and reporting tests without image
datasets or a GPU. The focused tables and CIFAR geometry figure can also be
regenerated from the included records, as described in the README.
