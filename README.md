# Semantic Manifold Adversarial

Research code and documentation for studying adversarial perturbations through the geometry of learned latent representations.

The project starts from an MNIST variational autoencoder (VAE) and a separate victim classifier, then studies whether local latent geometry—especially PCA-estimated tangent spaces—can guide targeted, sparse, and boundary-aware adversarial perturbations.

## Current research questions

1. Does the VAE learn a latent representation with meaningful local class geometry?
2. Can a local same-class tangent space be estimated from nearby latent codes?
3. Can targeted perturbations constrained to that tangent space cross a classifier decision boundary?
4. Can the tangent coefficients be made sparse while retaining attack success?
5. Do useful directions generalize across multiple source samples instead of only one example?
6. Can geometry-derived warm starts improve later constrained optimization such as PDPGD-style attacks?
7. Do the discovered directions or perturbations transfer across victim models?

## Current baseline

The original exploratory notebook uses MNIST, a convolutional VAE with a 16-dimensional latent space, and a separate convolutional classifier. The notebook contains reconstruction checks, latent extraction, KNN and cluster diagnostics, interpolation/smoothness experiments, targeted latent attacks, local PCA tangent-space attacks, sparse tangent attacks, boundary-crossing analysis, and warm-start experiments.

The original notebook is preserved under `archive/` and should be treated as the historical research log. Stable code is gradually moved into `src/`, while the notebooks under `notebooks/` are intended to become reproducible experiments.

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
│   ├── 03_single_source_tangent_attack.ipynb
│   ├── 04_multi_source_tangent_generalization.ipynb
│   ├── 05_boundary_warm_start_pdpgd.ipynb
│   └── 06_transferability_experiments.ipynb
├── docs/
│   ├── problem_formulation.md
│   ├── vae_and_latent_space.md
│   ├── manifold_geometry.md
│   ├── attack_method.md
│   ├── experiments.md
│   └── results.md
├── figures/
├── checkpoints/
│   └── README.md
└── archive/
    └── VAE_on_MNIST_original.ipynb
```

## Experimental storyline

```text
Train VAE
   ↓
Validate reconstruction and latent structure
   ↓
Estimate local same-class tangent spaces
   ↓
Run a single-source tangent-constrained targeted attack
   ↓
Test multi-source generalization
   ↓
Use geometry-derived boundary warm starts
   ↓
Evaluate transferability across victim models
```

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
- Notebooks should contain experiment configuration, execution, plots, and interpretation—not duplicate model definitions.
- Each experiment notebook should state its research question, mathematical objective, setup, metrics, result, and interpretation.
- The original exploratory notebook is preserved unchanged in `archive/` as a research record.

## Status

Repository restructuring is in progress. The current codebase is being extracted from the original exploratory notebook into reusable modules and cleaner experiment notebooks.
