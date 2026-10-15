# Problem Formulation

## Goal

Study whether adversarial perturbations can be constructed and understood through the geometry of a learned latent representation instead of treating every input pixel as an unrelated optimization variable.

The current baseline uses MNIST, a convolutional variational autoencoder (VAE), and a separate victim CNN.

For an input image `x`, the VAE encoder produces a latent mean

\[
z_0 = E(x) \in \mathbb{R}^{16}.
\]

The decoder reconstructs

\[
\hat{x}=D(z_0).
\]

A targeted latent perturbation searches for `delta` such that

\[
\operatorname{argmax} C(D(z_0+\delta))=t,
\]

where `C` is the victim classifier and `t` is the desired target class.

## Tangent-space restriction

The central geometric hypothesis is that a useful perturbation can be restricted to a locally estimated same-class tangent space. If

\[
B\in\mathbb{R}^{d\times r}
\]

contains local PCA directions, then

\[
\delta_T = B a
\]

(or equivalently `a @ B.T` in the row-vector implementation), so optimization occurs in the lower-dimensional coordinate vector `a` rather than the full latent space.

## Baseline objective

The exploratory notebook combines three pressures:

\[
J = L_{margin} + \lambda \frac{\|\delta\|_2}{s_z} + \beta\,\mathrm{MSE}(D(z_0+\delta), D(z_0)),
\]

where the targeted margin term is

\[
L_{margin}=\max\left(\max_{k\neq t} f_k(x')-f_t(x')+\kappa,0\right).
\]

The image-space MSE is used only as a local semantic-preservation proxy; it is not treated as a proof that semantics are unchanged.

## Research progression

The repository separates the project into: latent-space validation, single-source tangent experiments, multi-source generalization, boundary-aware warm starts, and cross-model transfer evaluation.
