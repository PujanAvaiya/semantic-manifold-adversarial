# Archive

This directory is reserved for the original exploratory research notebook before modularization.

The source notebook supplied for this restructuring is `VAE_on_MNIST (5)(1).ipynb` (97 cells). It contains the historical sequence of experiments: VAE setup/training/loading, reconstruction checks, latent diagnostics, victim CNN work, direct latent optimization, tangent-space PCA experiments, sparse tangent experiments, decision-boundary searches, and reconstruction-cost analysis.

The original notebook should remain a historical record. Clean reproducible experiments belong in `notebooks/`, and stable reusable code belongs in `src/`.

**Migration note:** the GitHub connector used for this restructuring can create UTF-8 repository files but does not expose a direct local-file upload action. The exact 1.4 MB notebook attachment therefore still needs to be copied into this directory as `VAE_on_MNIST_original.ipynb`; the restructuring work intentionally does not replace it with a modified copy.
