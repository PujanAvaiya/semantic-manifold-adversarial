# Experiment Roadmap

## Notebook 01 — Train VAE

Purpose: reproduce the convolutional VAE baseline and save a checkpoint.

Record:
- random seed
- device
- epochs
- optimizer and learning rate
- beta
- train/test reconstruction metrics
- checkpoint path

## Notebook 02 — Validate Latent Space

Purpose: establish whether the learned representation is suitable for geometric experiments.

Reproduce the archived checks for reconstruction quality, latent KNN behavior, class centroids/radii, class separation, silhouette behavior, small-transform stability, local smoothness, interpolation, and latent variance/collapse.

## Notebook 03 — Single-Source Tangent Experiment

Purpose: reproduce one clean source-to-target experiment while separating geometry estimation from optimization code.

Record the chosen source image, source/target classes, neighborhood size, tangent dimension, explained variance, optimization hyperparameters, success/failure, latent norm, image MSE, and boundary location.

## Notebook 04 — Multi-Source Tangent Generalization

Purpose: move beyond a single successful example.

For many correctly classified source examples:
1. encode the example;
2. estimate its own local same-class tangent basis;
3. run the same fixed experimental protocol;
4. collect success rate and perturbation statistics;
5. compare tangent directions across examples.

This notebook is the next major research step.

## Notebook 05 — Boundary Warm Start + PDPGD

Purpose: compare a geometry-derived near-boundary initialization against a standard initialization under the same downstream optimization objective.

Do not claim improvement until both initializations use identical budgets, stopping conditions, and evaluation metrics.

## Notebook 06 — Transferability

Purpose: evaluate whether geometry-derived perturbations/directions remain effective across separately trained victim architectures or seeds.

Report both success and failure cases. Keep the VAE/generator fixed when the goal is to isolate classifier transfer.
