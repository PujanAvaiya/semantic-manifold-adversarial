# Attack Method

## 1. Direct latent baseline

The direct baseline optimizes an additive latent perturbation

\[
z' = z_0 + \delta.
\]

The decoded image `D(z')` is evaluated by the victim classifier. The objective combines a targeted margin loss, normalized latent movement, and reconstruction-relative image MSE.

## 2. Tangent-constrained variant

Instead of allowing arbitrary `delta` in the full 16-dimensional latent space, estimate a local basis `B` and optimize coefficients `a`:

\[
\delta_T = B a.
\]

The code stores examples as row vectors, so the implementation is

```python
delta_tangent = a @ B.T
z_adv = z0 + delta_tangent
```

This guarantees that the constructed perturbation lies in the span of the estimated local basis.

## 3. Sparse tangent variant

The exploratory notebook also studies sparsity in tangent coordinates. After a gradient update, a hard-thresholding operator can keep only the largest-magnitude `K` coefficients:

\[
a \leftarrow H_K(a).
\]

That enforces

\[
\|a\|_0 \le K.
\]

Sparsity of `a` means few local tangent directions are active. It does **not** necessarily mean that the final decoded image changes only a few pixels or that the full latent perturbation is coordinate-sparse.

## 4. Boundary-aware warm start

Once a successful tangent direction is found, the exploratory notebook searches along

\[
z(\alpha)=z_0+\alpha\delta_T
\]

and finds the last point still predicted as the source class before the target-class boundary is crossed. A dense scan followed by binary search produces a near-boundary starting point.

The purpose of the later `05_boundary_warm_start_pdpgd.ipynb` experiment is to test whether this geometry-derived start improves a subsequent constrained optimizer.

## PDPGD status

A reusable PDPGD implementation has **not** yet been migrated from the current project materials. The repository therefore keeps an explicit integration point in `src/attacks/pdpgd.py` rather than inventing code that was not part of the recorded baseline.
