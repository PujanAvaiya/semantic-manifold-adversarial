# Results

This file is intentionally a structured results log rather than a place for unverified conclusions.

## Baseline status

The original exploratory notebook contains successful latent/tangent experiments and extensive diagnostic plots. Those results are being migrated into reproducible numbered notebooks before they are summarized here as final project claims.

## Reporting template

For each experiment, record:

| Field | Value |
|---|---|
| Dataset | |
| VAE checkpoint / seed | |
| Victim checkpoint / seed | |
| Source class | |
| Target class | |
| Number of evaluated samples | |
| Local neighbors K | |
| Tangent dimension | |
| Explained local variance | |
| Attack/optimization success rate | |
| Median latent L2 | |
| Median image MSE | |
| Sparsity metric | |
| Transfer setting | |

## Required visual evidence

For representative success and failure cases, save:

- original image;
- deterministic VAE reconstruction;
- changed/optimized decoded image;
- absolute difference map;
- optimization curves;
- tangent/PCA diagnostics when relevant.

Generated figures should be written under `figures/generated/` locally and only selected publication-quality figures should be committed.

## Interpretation rule

Keep three statements separate:

1. **Observed:** what the experiment directly measured.
2. **Inferred:** the geometric interpretation suggested by the measurement.
3. **Not established:** stronger claims that would require additional controls or datasets.
