# Reproduction: arXiv:2208.03923

Paper: **Adversarial robustness of VAEs through the lens of local geometry** by Asif Khan and Amos Storkey.

This reproduction is inserted before the TNAR/ManiFool study because it directly tests Jacobian-based local geometry on the same kind of VAE representation already used by this project.

## Core equation being reproduced

For a stochastic encoder

\[
f(x)=\mu(x)+\epsilon\odot\sigma(x),\qquad \epsilon\sim\mathcal N(0,I),
\]

the paper defines the expected stochastic pullback metric

\[
\hat G_x
=J_\mu(x)^T J_\mu(x)
+J_\sigma(x)^T J_\sigma(x).
\]

The local fixed-norm optimisation

\[
\max_\eta \eta^T\hat G_x\eta
\quad\text{s.t.}\quad \|\eta\|_2=\eta_0
\]

reduces to the eigenproblem

\[
\hat G_x\eta=\lambda\eta.
\]

The dominant eigenvector is therefore the locally maximally sensitive direction under this quadratic approximation.

The paper describes a one-step corruption

\[
x_c=x+\delta\lambda\eta.
\]

## Phase A — core mechanism on our existing ConvVAE

The existing trained VAE is held fixed to test the local geometric mechanism.

Implementation: `src/latent_geometry/pullback_metric.py`

Notebook: `notebooks/reproductions/2208_03923_core_pullback_attack.ipynb`

For several MNIST test samples:

1. compute `J_mu` and `J_sigma` with autograd;
2. form the 784 x 784 pullback metric;
3. verify symmetry and positive-semidefinite behaviour numerically;
4. eigendecompose the metric;
5. measure the spectral radius and the paper/code anisotropy score;
6. perturb along the dominant and top-five eigendirections;
7. measure latent displacement and reconstruction MSE as the step grows;
8. compare the dominant direction with random directions under matched input-space L2 norm.

### Positive reproduction criteria

The core mechanism is considered reproduced on this VAE only if the dominant pullback direction consistently causes stronger local latent/reconstruction change than matched random directions over multiple samples, and if the effect increases sensibly with perturbation magnitude.

A visually dramatic reconstruction from one image is not sufficient evidence.

## Important implementation detail: delta grid

The paper prose states that 40 logarithmic steps are used over a range beginning at `0.01`, while the authors' public implementation uses

```python
np.logspace(-1, 1, 40)
```

which is `0.1` to `10`. The figure values near `0.5233` and `0.7443` are consistent with the public-code grid. Figure-level reproduction uses the public-code grid and records the discrepancy.

## Phase B — paper-faithful beta-VAE sweep

If Phase A works, the next reproduction step is to train paper-style beta-VAEs and test whether increasing beta suppresses the dominant pullback eigenvalue and directional bias.

The supplementary material specifies for MNIST/FashionMNIST an MLP encoder with hidden widths `256, 256, 512, 32`, a 32-dimensional Gaussian latent distribution, an inverse-style decoder, tanh activations with batch normalisation, and Adam with learning rate `0.003`.

The authors also sweep beta over 50 logarithmically spaced values between `0.01` and `10` for robustness analysis.

Phase B results are kept separate from the ConvVAE experiment because the architectures and evaluation questions differ.

## Why this comes before TNAR/ManiFool

This paper lets us verify three ingredients directly on our current setup:

- encoder Jacobians;
- a local metric induced by the VAE representation;
- eigendirections of local sensitivity.

Once these are working, TNAR gives us the tangent/normal decomposition perspective and ManiFool gives us the curved-manifold/relinearisation perspective. That ordering makes the Week 1 geometry work empirical rather than purely theoretical.
