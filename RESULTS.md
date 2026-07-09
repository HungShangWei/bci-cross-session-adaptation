# Cross-session re-adaptation: extended results

Extends the primary benchmark (`adaptation_curve.png`, BNCI2014_001 / BCI IV-2a) with
(1) replication on two more MOABB motor-imagery datasets, (2) a readout ablation, and
(3) a diagnostic decomposition of the adaptation curve. All runs reuse the identical
no-leakage protocol and decoders from `moabb_cross_session_adaptation.py`; the extra
datasets are driven by `run_extra.py` / `run_extra_ckpt.py`. Every number below is
computed from the saved `results_*.pkl` by `diag.py` (zero recompute).

## Headline: accuracy and calibration recover on different timescales

The central observation from the primary benchmark — after cross-session drift a decoder's
**accuracy can recover while its calibration does not** — replicates on all three datasets.
At a small adaptation budget (5 target trials) the continuous-time CfC decoder shows a large
negative-log-likelihood (NLL) spike while Cohen's κ moves comparatively little. The binding
constraint of the recovery (the slowest-recovering dimension of D_rec = max{D_acc, D_cal})
is **D_cal on every dataset**.

Pre-registered thresholds: γ = 0.85 (accuracy), NLL margin = 0.15, self-reference at the
largest budget. Wilcoxon signed-rank tests contrast budget 5 vs the fully-adapted budget,
paired over subjects, pre-specified (no multiple-comparison search).

| Dataset (n) | drift? | CfC κ @0→@5 | CfC NLL @0→@5 | D_acc / D_cal | binding | decoupling (CfC) |
|---|---|---|---|---|---|---|
| IV-2a / BNCI2014_001 (9) | yes | 0.51 → 0.41 | 0.61 → 0.94 | 20 / 40 | D_cal | ΔNLL=+0.49, p=0.004 |
| IV-2b / BNCI2014_004 (9) | yes | 0.39 → 0.46 | 0.63 → 0.73 | 10 / 80 | D_cal | ΔNLL=+0.35, p=0.004 |
| Lee2019_MI (54) | weak | 0.75 → 0.61 | 0.30 → 0.62 | 10 / 40 | D_cal | ΔNLL=+0.37, p≈2e-10 |

On IV-2b the effect is cleanest: the smallest adaptation *raises* CfC's accuracy while
*worsening* its calibration — the two axes move in opposite directions. On Lee2019 (n=54)
the decoupling is the most statistically robust (p ≈ 2×10⁻¹⁰), even though its accuracy
axis is near-saturated (see caveats).

Practical implication: an accuracy-only benchmark would report "recovered" at budgets where
the probability estimates are in fact worse than before adaptation. This motivates reporting
calibration (NLL/Brier) and idle/no-intent rejection as separate recovery dimensions rather
than a single accuracy score.

## Readout ablation: no superiority is claimed for CfC

CfC's ranking depends on an arbitrary implementation choice — how the continuous-time state
sequence is pooled into a decision (mean-pool vs last-step). Swapping the pre-registered
mean-pool readout for last-step drops CfC's ceiling κ on IV-2a from **0.563 to 0.327**
(2nd place to last). Because the ranking is readout-sensitive, this repository makes **no
claim that CfC is the better decoder**; CfC is used only as a probe that exhibits the
accuracy/calibration decoupling clearly. (`curve_iv2a_last.png`, `binding_constraint_iv2a_last.*`.)

## The diagnostic points to a direction

Across datasets the binding constraint is consistently D_cal — calibration recovery, not
representation transfer, is the bottleneck after drift. Under the framework this indicates
method development should target a calibration-constrained re-adaptation objective rather
than accuracy alone. The benchmark identifies which axis binds, and that axis selects the
next step.

## Caveats (read these)

- **D_FAH is not computed.** None of these datasets contains an idle / no-intent stream, so
  the false-activation dimension of D_rec cannot be measured here. Only {D_acc, D_cal} of the
  three dimensions are available.
- **Lee2019 accuracy axis is near-saturated.** Session-1→session-2 transfer on Lee2019 is
  near-trivial for the classical pipeline: EA+TS-LR reaches κ ≈ 1.0 at budget 0 for all 54
  subjects. The accuracy-efficiency axis therefore carries little signal on this dataset, and
  the EA+TS-LR curve is **not used as evidence**. The decoupling finding rests on the deep
  decoders (CfC, EEGNet), whose accuracy is not saturated (CfC κ@0 = 0.75, 8/54 subjects
  below 0.5). The EA κ ≈ 1.0 most likely reflects easy session-to-session transfer and/or
  high-dimensional tangent-space overfitting at 62 channels (tangent dim ≈ 1953 vs ~100
  trials); it is flagged here rather than featured.
- **Preliminary, single-run results** intended as proof-of-work, not a peer-reviewed
  evaluation. Budgets, decoders, and hyper-parameters are fixed at the values in
  `moabb_cross_session_adaptation.py`.
- **No claim of a new architecture or a standard vocabulary.** The contribution is a
  reproducible evaluation protocol and diagnostic, demonstrated on public data.

## Files

- Primary: `adaptation_curve.png` (+ `binding_constraint_orig.*`, `diagnostics_ci_orig.csv`)
- IV-2b: `curve_bnci2b_mean.png`, `results_bnci2b_mean.pkl`, `binding_constraint_bnci2b_mean.*`, `diagnostics_ci_bnci2b_mean.csv`
- Lee2019 (n=54): `curve_lee2019_mean_n54.png`, `results_lee2019_mean_ckpt.pkl`, `binding_constraint_lee2019_mean_ckpt.*`, `diagnostics_ci_lee2019_mean_ckpt.csv`
- Readout ablation: `curve_iv2a_last.png`, `results_iv2a_last.pkl`, `binding_constraint_iv2a_last.*`
- Reproduce any table: `PKL=<file>.pkl python3 diag.py`
