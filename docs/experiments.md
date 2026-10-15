# Experiment Notebook Map

The authoritative research schedule is in [`research_roadmap.md`](research_roadmap.md). This file maps that eight-week plan to reproducible notebooks.

The first two notebooks are prerequisites that establish the VAE representation. Notebooks 03–12 follow the staged research plan. Later notebooks are intentionally conditional: do not implement them early merely because a placeholder exists.

## Prerequisite — Notebook 01: Train VAE

`01_train_vae.ipynb`

Purpose: reproduce the convolutional VAE baseline and save a checkpoint.

Record random seed, device, epochs, optimizer, learning rate, beta, train/test reconstruction metrics, latent dimension, and checkpoint path.

## Prerequisite — Notebook 02: Validate Latent Space

`02_validate_latent_space.ipynb`

Purpose: establish whether the learned representation is suitable for geometric experiments.

Reproduce reconstruction quality, latent KNN behavior, class centroids/radii, separation, silhouette behavior, small-transform stability, interpolation, local smoothness, and latent variance/collapse checks.

---

## Week 1 — Notebook 03: Geometry, Jacobians, Tangent/Normal Decomposition

`03_geometry_jacobian_tangent_normal.ipynb`

Study TNAR/ManiFool geometry, Jacobians, local linearization, tangent/normal decomposition, decision-boundary normals, and the relationship between PCA-estimated and Jacobian-derived local geometry.

This notebook is primarily understanding and numerical verification—not algorithm modification.

## Week 2 — Notebook 04: Multi-Source Current-Attack Diagnostics

`04_multi_source_attack_diagnostics.ipynb`

Run the current fixed attack across many correctly classified digit `1` samples without changing the method.

Collect sample-level and aggregate diagnostics: success, optimization steps, margins, perturbation norms, reconstruction distortion, neighbor distances/purity, singular values, tangent dimension, tangent/normal components, boundary-crossing location, and representative success/failure cases.

**Rule:** no adaptive neighbors, adaptive `r`, target alignment, or relinearization in this notebook.

## Week 3 — Notebook 05: Adaptive Geometry Ablations

`05_adaptive_geometry_ablation.ipynb`

Test only the modifications justified by Notebook 04:
- adaptive neighbors;
- adaptive tangent dimension `r`;
- target alignment;
- relinearization / tangent re-estimation.

Each intervention must first be tested independently against the fixed Week 2 baseline.

## Week 4 — Notebook 06: Self-Supervised Semantic Neighbors

`06_self_supervised_neighbor_retrieval.ipynb`

Remove ground-truth labels from neighborhood construction and test semantic/representation-based neighbor retrieval.

Compare label-free versus labeled neighborhoods on semantic purity, local subspace stability, attack success, distortion, and failure modes.

## Week 5A — Notebook 07: Deterministic Transfer Baselines

`07_deterministic_transfer_baselines.ipynb`

Establish transfer performance of the deterministic optimized tangent-coordinate solution `a*` before introducing any learned distribution.

Test separate seeds and different victim architectures while clearly separating source-model success from transfer success.

## Week 5B — Notebook 08: Sampling Around `a*`

`08_sampling_around_optimal_direction.ipynb`

Test simple stochastic exploration around the deterministic optimum before training a distribution model.

For example, evaluate controlled perturbations of the form `a = a* + noise` and measure source success, transfer success, distortion, direction diversity, and sensitivity to sampling scale.

**Decision gate:** Notebook 09 is entered only if Notebook 08 provides evidence that a useful transferable region/distribution exists.

## Week 6 — Notebook 09: Learned Direction Distribution — Conditional

`09_learned_direction_distribution.ipynb`

Only if Week 5 is promising, derive and implement a conditional distribution

`q_psi(a | z, t)`

and compare it against deterministic `a*`, random tangent sampling, and simple local sampling around `a*`.

Do not implement this stage merely to complete the notebook sequence.

## Week 7A — Notebook 10: Surrogate Ensemble

`10_surrogate_ensemble_transfer.ipynb`

Evaluate optimization using multiple surrogate classifiers. Measure whether an ensemble improves generalization beyond single-surrogate geometry.

## Week 7B — Notebook 11: Unseen Architecture Transfer

`11_unseen_architecture_transfer.ipynb`

Hold victim architectures out of optimization and evaluate transfer to unseen architectures and unseen random initializations.

Produce a transfer matrix that identifies whether architecture and/or initialization was seen during optimization.

## Week 8 — Notebook 12: Ablations and Failure Analysis

`12_ablation_failure_analysis.ipynb`

Stop adding components. Perform the final ablations, plots, failure taxonomy, limitations analysis, and contribution identification.

The final contribution must be inferred from the experimental evidence rather than selected in advance.

---

# Research discipline

For every experimental notebook:

1. state the research question before running code;
2. define the baseline and the single variable being changed;
3. fix seeds and evaluation rules where possible;
4. record both successful and failed examples;
5. save machine-readable per-sample results, not only plots;
6. distinguish observation from interpretation;
7. add a component to the final method only if its ablation supports keeping it.
