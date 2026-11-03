# Proposal integration and final experiments

## Working title

**Geometry-Reliable Adversarial Generation for Surrogate-Robust Black-Box Transfer**

## Main hypothesis

Data geometry learned independently of a victim classifier, when screened for reliability, may suppress surrogate-specific attack directions and improve black-box transfer while preserving semantic validity. This is testable and may fail.

## Completed preliminary work

- Frozen score-model Jacobian and candidate tangent directions in MNIST pixel space.
- Oracle-grounded synthetic geometry and rank/orientation separation.
- Model-versus-smoothing diagnostics and quadrature audit.
- Fit/calibration proxy selection and conservative error-bound feasibility.

## Three remaining experiments (stop after these)

### A. Cross-model alignment and one-step transfer (currently running externally)

Frozen MNIST pretrained score model; independent classifier CNN-A, CNN-B, MLP, small ResNet; fixed score-noise scales t=5 and t=100. For an input x compare full gradient alignment, score-filtered gradient, estimated tangent bank, normal residual, same-rank random subspace and local PCA. Use paired image-level confidence intervals. Target model gradients for **offline scientific alignment only**; no use in attack generation/tuning. Supplement cosine with retained gradient norm, and matched-L2 actual target attack effect.

Notebook: notebooks/research_archive/06_mnist_tangent_transfer_alignment.ipynb

**Stop/go:** If higher cosine is not accompanied by actual transfer benefit under matched budgets, do not motivate generator performance using cosine alone.

### B. Matched black-box transfer

Compare PGD, random subspace, local PCA, fixed score geometry, and frozen reliability-filtered score geometry. Cross-architecture test matrix, multiple model seeds, matched budgets and semantic-label checks; report success conditional on accepted examples **and** all examples with a predefined abstention fallback. Freeze hyperparameters without consulting held-out targets.

### C. Simple geometry-conditioned generator pilot

Compare capacity/training-matched G_φ(x, ξ) and G_φ(x, ξ, P̂_T, U_T). Same surrogate supervision, perturbation budget, and semantic checks; held-out classifiers only for evaluation. Do not introduce a new EBM or new manifold geometry estimator before these tests.

**Stop/go:** A well documented negative result still supports a bounded PhD proposal; there is no requirement to prove a generator works before writing the proposal.

## Proposal structure

1. Motivation: transferable black-box vulnerabilities and surrogate dependence.
2. Gap: data-geometry evidence may be uncertain; tangent≠semantic and geometric rank≠orientation.
3. Preliminary evidence: F1–F6 with explicit limits.
4. Proposed three research objectives: reliable geometry; relationship to transfer; generator conditioning.
5. Experimental protocol and baselines, held-out model seeds, uncertainty and semantic-valid success.
6. Expected (not achieved) contributions and failure-contingency plan.

## Research ethics/reproduction

Use only permitted models/datasets and non-production test classifiers. Never publish secrets, raw personal images, private Drive files, pretrained checkpoints, or any victim credentials.
