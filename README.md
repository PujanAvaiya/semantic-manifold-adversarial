# Semantic Manifold Adversarial

Research code and documentation for studying adversarial perturbations through the geometry of learned latent representations.

The project starts from an MNIST variational autoencoder (VAE) and a separate victim classifier, then studies whether local latent geometry—especially tangent/normal structure—can guide targeted perturbations that generalize beyond a single source example and eventually transfer across models.

## Research principle

The project is intentionally staged. We do **not** add adaptive geometry, self-supervised retrieval, probabilistic direction models, or ensembles until simpler experiments show why they are needed.

The authoritative eight-week plan is in [`docs/research_roadmap.md`](docs/research_roadmap.md). The notebook-to-stage mapping is in [`docs/experiments.md`](docs/experiments.md).

## Eight-week plan

```text
Week 1  TNAR/ManiFool geometry, Jacobians, tangent/normal decomposition
   ↓
Week 2  Run current attack across many digit 1 samples + diagnostics
   ↓
Week 3  Fix only diagnosed problems:
        adaptive neighbors / adaptive r / target alignment / relinearization
   ↓
Week 4  Remove labels and test self-supervised semantic neighbors
   ↓
Week 5  Deterministic transfer baselines + simple sampling around a*
   ↓
        promising?
        ├── no  → do not build a learned distribution
        └── yes ↓
Week 6  Derive and test q_psi(a | z, t)
   ↓
Week 7  Surrogate ensemble + unseen architectures
   ↓
Week 8  Ablations, plots, failure analysis, determine actual contribution
```

## Current baseline

The original exploratory notebook uses MNIST, a convolutional VAE with a 16-dimensional latent space, and a separate convolutional classifier. It contains reconstruction checks, latent extraction, KNN and cluster diagnostics, interpolation/smoothness experiments, targeted latent optimization, local PCA tangent-space experiments, sparse tangent experiments, boundary-crossing analysis, and warm-start experiments.

The exact original notebook is being kept as the historical research log. The current GitHub connector does not expose a direct local-file upload action, so `archive/README.md` records the pending exact notebook copy rather than silently replacing it with a modified export. Stable code lives in `src/`, while notebooks are clean experiment entry points.

## Repository layout

```text
semantic-manifold-adversarial/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── models/
│   │   ├── vae.py
│   │   └── victim_cnn.py
│   ├── latent_geometry/
│   │   ├── latent_extraction.py
│   │   ├── local_pca.py
│   │   ├── tangent_space.py
│   │   └── manifold_metrics.py
│   ├── attacks/
│   │   ├── latent_attack.py
│   │   ├── tangent_attack.py
│   │   ├── sparse_tangent_attack.py
│   │   └── pdpgd.py
│   └── evaluation/
│       ├── metrics.py
│       └── visualization.py
├── notebooks/
│   ├── 01_train_vae.ipynb
│   ├── 02_validate_latent_space.ipynb
│   ├── 03_geometry_jacobian_tangent_normal.ipynb
│   ├── 04_multi_source_attack_diagnostics.ipynb
│   ├── 05_adaptive_geometry_ablation.ipynb
│   ├── 06_self_supervised_neighbor_retrieval.ipynb
│   ├── 07_deterministic_transfer_baselines.ipynb
│   ├── 08_sampling_around_optimal_direction.ipynb
│   ├── 09_learned_direction_distribution.ipynb
│   ├── 10_surrogate_ensemble_transfer.ipynb
│   ├── 11_unseen_architecture_transfer.ipynb
│   └── 12_ablation_failure_analysis.ipynb
├── docs/
│   ├── problem_formulation.md
│   ├── vae_and_latent_space.md
│   ├── manifold_geometry.md
│   ├── attack_method.md
│   ├── research_roadmap.md
│   ├── experiments.md
│   └── results.md
├── figures/
├── checkpoints/
│   └── README.md
└── archive/
    └── README.md
```

## Stage gates

Three gates prevent unnecessary complexity:

1. **Week 2 → Week 3:** only failures observed in the multi-source diagnostics justify adaptive geometry.
2. **Week 4:** label-free semantic neighborhoods must be good enough to support the geometry before the project can claim reduced supervision.
3. **Week 5 → Week 6:** implement `q_psi(a | z,t)` only if deterministic transfer plus simple sampling indicates that a useful transferable direction distribution exists.

## Setup

```bash
git clone https://github.com/PujanAvaiya/semantic-manifold-adversarial.git
cd semantic-manifold-adversarial
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

MNIST is downloaded by `torchvision` when the training/validation notebooks are run.

## Reproducibility policy

- Large model checkpoints and generated datasets are not committed directly to Git.
- Stable model and geometry code belongs in `src/`.
- Notebooks contain experiment configuration, execution, diagnostics, plots, and interpretation—not duplicate model definitions.
- Every experiment states its baseline, changed variable, metrics, stopping rule, and failure cases.
- Per-sample results should be saved in machine-readable form before aggregate plotting.
- The exact original exploratory notebook must remain unchanged when copied into `archive/`.
- The final research contribution will be determined from the ablations and failure analysis, not selected in advance.

## Status

Repository foundation is in place. The roadmap is now evidence-driven and staged; experiments should proceed in the order defined in `docs/research_roadmap.md`.