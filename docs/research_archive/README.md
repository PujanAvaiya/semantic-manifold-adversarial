# Research archive — score geometry → black-box adversarial transfer

**Status (2026-11-02):** consolidated research record. This section extends, and does not replace, the existing latent-VAE roadmap in this repository.

## The scientific question

Can independently learned, **reliability-aware local data geometry** improve the semantic validity and cross-model transfer of surrogate-trained, target-query-free adversarial examples?

The work has two distinct stages:

1. **Completed / preliminary geometry investigation**: synthetic manifolds, pretrained score-Jacobian geometry on MNIST, fixed-versus-calibrated noise-scale comparisons, reliability diagnostics, score-model error versus smoothing bias, and conditional spectral error-bound feasibility.
2. **Proposed security investigation**: independent classifier-gradient alignment, actual black-box transfer at matched perturbation budgets, then a geometry-conditioned adversarial generator. There are **no positive transfer results yet** from the new alignment notebook.

## Navigation

- [Findings and limitations](findings.md): grounded observations, sample counts, and caveats
- [Notebook index](notebook_index.md): available notebook lineage and pending historical files
- [Proposed experiment and proposal plan](proposal_plan.md): exact go/no-go tests
- [Reproduction and data provenance](reproduction.md): requirements, paths, run protocols

## Repository rules

- Preserve original code, study stages and prior results; no silent replacement of historical notebooks.
- Commit *code and small aggregate evidence* only. Exclude raw data, pretrained weights, credentials, personal information, Drive dumps and enormous per-point CSVs.
- Where an experiment has only a smoke test or a draft notebook, mark it **NOT EXECUTED ON REAL DATA**.
- Ground-truth synthetic geometry may be used for audit and supervised diagnostics; it is **not available on real MNIST**.
- Estimated-rank and oracle-rank tangent errors are different metrics. A good estimate of intrinsic dimension is **not** a tangent-orientation guarantee.
- Publication/novelty claims require comparison with current prior work. These notebooks do not establish state-of-the-art performance.

## Importing additional notebooks from another Gmail/Colab account

Google Colab/Drive files are account-scoped. GitHub access cannot automatically enumerate another Google account. Export **File → Download → Download .ipynb**, and download result CSVs from that account's Drive. Use a clean filename and retain the original in a private archive; add selected notebook copies only after reviewing outputs, secrets, and licensing. See the notebook index for known missing artifacts.

## Proposed contribution (not an achieved result)

Given clean image x, random variable ξ, frozen tangent estimate P̂_T(x), and reliability score U_T(x), investigate a generator G_φ(x, ξ, P̂_T, U_T) trained on surrogate classifiers and evaluated only on held-out target models. Whether tangent constraints help transfer is an **empirical hypothesis**. Do not assume tangency means semantic preservation.
