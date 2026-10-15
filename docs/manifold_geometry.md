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

## Recognition pattern

If you see: **nearby same-class latent points + local covariance/PCA/SVD**, think: **estimate a local tangent-like subspace**.

If you see: **delta = B a**, think: **optimize coordinates inside that estimated local subspace instead of all latent coordinates independently**.
