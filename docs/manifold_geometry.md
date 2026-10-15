# Local Manifold Geometry

## Local same-class neighborhood

For a source latent code `z0`, the baseline first restricts the reference set to training examples with the same true class. It then selects the nearest `K` latent means by Euclidean distance.

If those neighbors are collected into `Z_local`, the neighborhood is centered by

\[
X_{local}=Z_{local}-\bar z_{local}.
\]

## PCA / SVD tangent estimate

The exploratory notebook applies SVD:

\[
X_{local}=U S V^T.
\]

The leading right-singular vectors are treated as local directions of variation. A tangent basis is formed from the smallest number of components needed to explain a chosen fraction of local variance, subject to a maximum dimension.

In the baseline configuration:

- nearest same-class neighbors: 200
- target cumulative local variance: 80%
- maximum tangent dimension: 8
- latent dimension: 16

The resulting basis

\[
B\in\mathbb{R}^{16\times r}
\]

has approximately orthonormal columns, checked through

\[
\|B^TB-I\|.
\]

## Interpretation

This is a local linear approximation, not a proof that the data manifold is globally linear or that every vector in the PCA span preserves semantic identity. The basis is used as an experimentally motivated search subspace.

## Week 1 geometry study targets

Before changing the baseline attack, the project will explicitly study four geometric views side by side:

1. **Local PCA geometry** — the empirical tangent-like subspace currently used by the code.
2. **Jacobian geometry** — local linear maps induced by encoder/decoder/classifier functions.
3. **Tangent/normal decomposition** — decompose a vector into components parallel and orthogonal to the estimated local tangent space.
4. **TNAR/ManiFool perspective** — study how those methods motivate manifold-aware directions and decision-boundary geometry, then compare their assumptions with the current latent-PCA construction.

For an orthonormal tangent basis `B`, a latent vector `v` can be decomposed numerically as

\[
v_{tan}=BB^T v,
\]

and

\[
v_{normal}=v-v_{tan}.
\]

The Week 1 notebook must verify this decomposition numerically and compare it with Jacobian-derived local information before any adaptive algorithm is introduced.

## Questions that must remain separate

- **Data-manifold tangent:** which directions represent local variation of plausible samples?
- **Data-manifold normal:** which directions leave that estimated local variation?
- **Classifier boundary normal:** which direction locally changes the decision score fastest?

These objects may interact, but they should not be treated as interchangeable without measurement.

## Recognition pattern

If you see: **nearby same-class latent points + local covariance/PCA/SVD**, think: **estimate a local tangent-like subspace**.

If you see: **delta = B a**, think: **optimize coordinates inside that estimated local subspace instead of all latent coordinates independently**.

If you see: **BB^T v**, think: **project `v` onto the estimated tangent space**.

If you see: **v - BB^T v**, think: **normal component relative to that estimated tangent space**.
