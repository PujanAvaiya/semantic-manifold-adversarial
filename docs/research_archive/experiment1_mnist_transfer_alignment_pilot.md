# Experiment 1 — MNIST tangent-candidate gradient alignment (24-image pilot)

**Status:** completed PILOT, 24 sampled MNIST digit-1 images; **not** a successful transfer attack. Summary transcribed from user-supplied Colab tables and figures on 2026-11-02. The original per-image output CSVs have not yet been imported. This file is a research record and is not a preprint or a claim of novel SOTA.

## Question

Do pretrained score-derived pixel-space tangent-*candidate* filters improve agreement of input-loss gradients across independently trained classifiers, relative to full-space gradients, local PCA and dimension-matched random subspaces? Does alignment predict attack transfer?

## Protocol visible in reported results

- Four classifier architectures/instances: CNN-A, CNN-B, MLP, tiny ResNet.
- 24 sampled digit-1 MNIST images; pairwise gradient measures use 144 image/model-pair rows per single-scale method (and 432 for aggregate random controls).
- Pretrained MNIST score filter at diffusion timestep t = 5 or 100.
- Gradient comparison methods include unfiltered (full), score-direct (filter applied directly to each gradient), effective-dimension PCA, random effective-dimension baseline, candidate-score banks and normal/residual controls.
- One-step surrogate-trained, **target-feedback-free generation**, scored against held-out classifiers. Tables report target-margin change and ASR. Each listed attack achieved L2 ~2.0; achieved L∞ varies by method.
- Paired 95% bootstrap intervals cluster comparisons by 24 sampled images as reported in the notebook.

## Gradient-alignment findings

| Measure | Mean |
|---|---:|
| Full gradient cosine | ~0.193 (plot estimate) |
| Direct score filter, t=5 | 0.3458 |
| Effective-rank PCA t=5 | 0.2649 |
| Effective-rank random t=5 | 0.2028 |
| Direct score filter, t=100 | 0.5262 |

Paired image-bootstrap mean differences:

- score-direct t=5 minus PCA-effective t=5: cosine **+0.08081**, 95% CI **[+0.05910,+0.10125]**.
- score-direct t=5 minus random-effective t=5: cosine **+0.14293**, 95% CI **[+0.11253,+0.17411]**.
- score-direct t=100 minus PCA-effective t=100: cosine **+0.07608**, 95% CI **[-0.05310,+0.19894]** (inconclusive).

Caveat: **score-derived tangent-candidate ≠ verified true tangent** in real MNIST. Filtered cosine can rise even when useful loss-gradient magnitude is reduced.

## Retained gradient energy

The supplied plot shows direct score filtering retains ~18% at t=5 but only ~2.5% at t=100. The pairwise table reports mean source-energy fractions 0.2007 (direct t5) and 0.0293 (direct t100), with minor differences attributable to aggregation. Thus high normalized cosine may coexist with tiny retained absolute gradient contribution.

## One-step transfer: no supported improvement in actual fooling

| Attack (same achieved L2 ≈ 2.0) | Mean target margin increase | Transfer ASR |
|---|---:|---:|
| Full-space gradient | **1.4059** | 0% |
| Score-direct t=5 | 1.1903 | 0% |
| PCA-effective t=5 | 0.7685 | 0% |
| Random-effective t=5 | 0.6838 | 0% |
| Score-direct t=100 | 0.5935 | 0% |

One normal-direction variant gave ~0.35% ASR (1 out of 288 evaluations). Most methods were exactly zero. These 288 evaluations share only 24 images and repeated models, so **not 288 independent images**.

Paired average target-margin difference score-direct t5 minus full: **−0.21555**, 95% CI **[−0.27967,−0.14630]**. Score-direct t5 minus PCA-effective t5: **+0.42179**, CI **[+0.34204,+0.49937]**. The conventional full-space method still caused a greater average target margin increase.

Comparison is matched in *achieved L2*, **not L∞**: e.g. table reports full mean L∞ 0.4355, score-direct t5 0.7409, PCA-effective t5 0.7116. Hence matched perceptual distortion, L∞ budget and semantic preservation were **not established**. No semantic-valid attack success can be claimed.

## Interpretation and decision

**Supported:** score-derived filtering can increase normalized cross-model gradient agreement compared with matched-rank PCA/random controls on this 24-image pilot. This is a directional-alignment observation.

**Unsupported:** geometry-conditioned attacks improve black-box fooling or semantic preservation. Observed ASR is essentially zero, and the full-space attack led in mean target-margin increase.

**Proposed single follow-up gate (not yet completed):** with the existing frozen models and geometry, run a *predeclared*, matched L2 + L∞ bounded attack comparison with multiple steps (e.g., 10–20), a small fixed budget sweep, and larger held-out images. Assess paired target fooling and valid-semantic fooling, not only cosine. Do not tune method or noise scale on held-out targets. If still no advantage, report the negative transfer result and retain geometry conditioning only as a PhD hypothesis, not a preliminary success.

## Reproduction

Notebook: [11_mnist_tangent_transfer_alignment.ipynb](../../notebooks/research_archive/11_mnist_tangent_transfer_alignment.ipynb). Pending original CSV import from Colab Drive: PAIRWISE_GRADIENT_ALIGNMENT.csv, PAIRED_IMAGE_BOOTSTRAP_CI95.csv, TRANSFER_ONE_STEP_TARGET_FEEDBACK_FREE_GENERATION.csv (verify actual exported filename). No claims here depend on unprovided row-level files.
