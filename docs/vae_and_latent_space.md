# VAE and Latent Space

## Baseline architecture

The current MNIST baseline uses a convolutional VAE with latent dimension 16.

Encoder:

```text
1x28x28
  -> Conv2d(1, 32, 3, stride=2, padding=1) + ReLU
  -> Conv2d(32, 64, 3, stride=2, padding=1) + ReLU
  -> 64x7x7
  -> separate linear heads for mu and logvar
  -> z in R^16
```

Decoder:

```text
z in R^16
  -> Linear(16, 64*7*7)
  -> ConvTranspose2d(64, 32, 4, stride=2, padding=1) + ReLU
  -> ConvTranspose2d(32, 1, 4, stride=2, padding=1) + Sigmoid
  -> 1x28x28
```

The training objective is reconstruction BCE plus KL divergence:

\[
L_{VAE}=L_{recon}+\beta D_{KL}(q_\phi(z|x)\|p(z)),
\]

with `beta = 1` in the recorded baseline.

## Why the latent mean is used for geometry

For geometry experiments, the notebook uses the deterministic posterior mean `mu` rather than a fresh sampled `z`. This removes sampling noise when computing neighborhoods, class centroids, PCA directions, interpolation paths, and attack trajectories.

## Validation performed in the exploratory notebook

The archived notebook contains checks for reconstruction MSE, KNN classification in latent space, class centroids/radii, inter-class separation, silhouette behavior, invariance under small image transformations, local decoder smoothness, latent interpolation, and variance/collapse diagnostics.

These checks should be reproduced in `notebooks/02_validate_latent_space.ipynb` before claims about manifold structure are made.
