# Compact evidence tables

These are small CSV snapshots from the supplied experiments, imported without fabricated values. See [findings](../../docs/research_archive/findings.md) for interpretation and limitations.

| CSV | Role |
|---|---|
| [numerical_quadrature_convergence.csv](numerical_quadrature_convergence.csv) | Reference smoothed-score numerical grid audit; sphere at σ=0.030 **failed** |
| [torus_failure_decomposition.csv](torus_failure_decomposition.csv) | Torus error components at multiple noise scales and independently trained models |
| [multiseed_ablation.csv](multiseed_ablation.csv) | Held-out model seed uncertainty-predictor comparisons |
| [independent_model_diagnostics.csv](independent_model_diagnostics.csv) | Cross-model response/consistency diagnostic summaries |
| [proposal_feasibility_summary.csv](proposal_feasibility_summary.csv) | Conditional eigenspace error-bound feasibility by family |
| [orientation_experiment1_fresh.csv](orientation_experiment1_fresh.csv) | Fresh dimension/orientation descriptive experiment |
| [response_ridge_ablation.csv](response_ridge_ablation.csv) | Response feature incremental ridge comparison |
| [risk_coverage_paired_ci95.csv](risk_coverage_paired_ci95.csv) | Paired uncertainty estimates for risk/coverage diagnostic |

## Important limitations

- These tables cover synthetic geometry or MNIST proxy evaluations. Do not interpret proxy risk as unknown real-MNIST tangent error.
- True synthetic tangent and smoothing references used for audit cannot be used for deployment to unknown images.
- Full ~8 MB pointwise tables, model checkpoints, generated images, and Google Drive runtime contents are **not** included.
- No meaningful adversarial-transfer success claim has yet been established by this archive. New transfer result CSVs are **pending** completion in Colab.
