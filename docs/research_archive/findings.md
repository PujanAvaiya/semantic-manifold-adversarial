# Consolidated findings — evidence, not promises

**Evidence cutoff:** 2026-11-02. Numbers are drawn from the supplied run summaries and synthetic result CSVs. All findings below are preliminary and must be reported with their specific evaluation setting.

## F1 — Dimension accuracy and tangent orientation can diverge

Four score networks (one original and three independent training seeds), 250 untouched synthetic test points per model and noise scale, torus with true intrinsic dimension 2:

| σ | Correct estimated-rank fraction | Oracle-rank tangent-projector error | Estimated-rank tangent-projector error |
|---|---:|---:|---:|
| 0.030 | 0.0% | 0.47040 | 0.70715 |
| 0.128 | 21.7% | 0.45007 | 0.60871 |
| 0.342 | 64.0% | 0.48266 | 0.46951 |

Increasing σ improved rank recovery substantially but did not improve oracle-rank tangent orientation. **The oracle-rank evaluation uses privileged synthetic labels, not a deployable method.** This illustrates a known mathematical distinction; it is not a claimed new theorem.

## F2 — Learned-score discrepancy and smoothing bias compete

The torus experiment separates an estimate of learned-score operator discrepancy from reference-smoothed geometry bias:

| σ | Mean learned/reference operator discrepancy (Frobenius / √8) | Mean smoothed-reference / true-tangent discrepancy |
|---|---:|---:|
| 0.030 | 0.33587 | 0.00049 |
| 0.128 | 0.26140 | 0.00861 |
| 0.342 | 0.12745 | 0.08698 |

These diagnostics have different definitions/scales and should **not** be summed as an additive causal attribution. The experiment motivates noise-scale selection based on geometric usefulness rather than dimension alone.

Reference numerical quadrature passes its grid audit at **11 of 12** family × scale conditions. Sphere σ=0.030 **failed** the grid convergence tolerance (maximum relative operator discrepancy 0.113); precise reference-based claims in that setting require higher resolution.

## F3 — Conditional spectral bounds can be mathematically sound but uninformative

Use reference conditional error bounds involving the learned eigengap, a justified upper bound on score-operator error, and smoothing bias. Under the *pointwise synthetic reference* bound, informative fractions across all three noise scales (and requiring correct rank and grid eligibility) were:

- Circle: **17.27%**
- Ellipse: **21.67%**
- Sphere: **5.93%**
- Torus: **0.03%**

Under the tested conservative FIT/CAL synthetic operator and smoothing envelopes, **0%** of tested certificates were non-vacuous. This does **not** disprove spectral perturbation theory. These empirical envelopes are fitted using synthetic reference labels and are **not deployable on unknown real-data geometry**.

## F4 — Independent-model agreement is not a universal reliability score

In held-out-model-seed random-forest ablations, adding independent-model response improved tangent-error prediction MAE on only **6 of 16** model-family folds. Improvements were not consistent across the circle, ellipse, sphere and torus. Supervised ablations still used true synthetic tangent errors for fitting: no label-free MNIST guarantee follows.

## F5 — Neighbor proxy helps on simple geometries, not all shapes

A sample-only kNN reference proxy showed mean within-scale Spearman associations with true tangent error of approximately +0.780 circle, +0.812 ellipse, +0.620 sphere, and −0.109 torus. The proxy uses observed data coordinates, not analytic parameter angles, but validation against true errors remains synthetic.

## F6 — Earlier selection comparison, with strict scope

A noise-calibrated tangent-scale selector yielded lower **proxy risk** on one modest MNIST calibration evaluation: approximately 0.2174 versus 0.3411 at matching 83/100 acceptance (about 36.3% lower proxy risk). These are **not** measured real-MNIST tangent-orientation errors. In separate synthetic oracle-rank geometry tests the circle showed a 9.7% paired error reduction compared with a fixed-scale LHSD-derived **spectral-head** baseline; the sphere improvement (~3%) was inconclusive. The full official LHSD method was not necessarily reimplemented or benchmarked in each comparison; fixed-scale spectral-head values should be labeled precisely.

## F7 — Existing adversarial work, separate from transfer

The user's earlier MNIST classifier/targeted PDPGD benchmark reported ~98.58% CNN test accuracy and a digit 1→7 targeted example at step 59, L1 ≈15.28, L2 ≈1.83, ~243 nonzero pixels. This is **not evidence of transfer to unseen black-box classifiers**.

## Open hypotheses (not proven)

1. Score-derived tangent-candidate gradients align better across independent classifiers than full, normal and rank-matched controls.
2. Reliability-aware geometry makes target-query-free transfer attacks stronger at matched distortion and semantic validity.
3. A geometry-conditioned generator can reduce sensitivity to the training surrogate.

The alignment and transfer notebooks are *planned/run on Colab separately*. Until real result CSVs are imported, do not claim improvement or significance.
