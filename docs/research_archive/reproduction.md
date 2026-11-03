# Reproduction and provenance

## Notebook lineage

The code falls into two historically separate lines:

1. **Latent VAE / targeted optimization:** baseline VAE, same-class neighbor PCA, encoder Jacobians and PDPGD. Existing historical root notebooks and docs in the repository remain unchanged.
2. **Pixel-space EBM and score geometry:** synthetic Gaussian-smoothed reference score, score Jacobian or Hessian analysis, matrix-free spectral filtering, LHSD-style spectral head comparisons, MNIST scale selection, reliability/uncertainty experiments and adversarial transfer alignment.

Do **not** conflate the two tangent representations. VAE latent PCA directions are in latent coordinates; score-derived tangent candidates are in ambient pixel space.

## Source models and output locations

- Some MNIST score notebooks fetch the publicly available Hugging Face model **harveymannering/mnist-ddpm**. An Internet connection and compatible diffusers version are required; this is not a checkpoint shipped by this repository.
- Earlier synthetic checkpoints and per-point CSVs are under **MyDrive/synthetic_tangent_ground_truth**.
- Multiseed follow-up uses **MyDrive/Geometry_Reliability_Followup_Independent_Seeds**.
- Spectral bound study uses **MyDrive/Spectral_Error_Bound_Feasibility**.
- MNIST alignment notebook writes to **MyDrive/MNIST_Tangent_Transfer_Alignment_Exp1**.
- These are external, account-specific Google Drive folders, **not tracked by GitHub**. A different Google account may have separate copies.

## Running a notebook

Download/open the specific notebook via Colab; mount your own Drive, install declared dependencies, run the pilot configuration before the full configuration. Preserve FIT, CAL and held-out TEST splits and classifier-model separation.

## Provenance and safety audit

- Archive only notebooks and **small** aggregate CSVs. Large raw tables and weights should be kept in user-controlled Drive or a dedicated release/artifact store.
- Prior exploratory notebooks may include saved outputs, absolute Google Drive paths or runtime state. Archived GitHub copies should have cell outputs cleared, runtime-specific settings documented, and secrets checked.
- Do not claim smoke-test statistics as actual experiments.
- Synthetic ground-truth rank and tangent projectors are for validation only.
- Critical numerical caveat: sphere σ=0.030 reference quadrature did not pass coarse/fine comparison.
- Exact original historical VAE and external-Gmail notebook copies should not be rewritten or inferred from summaries.

## Next import

Download historical notebooks and CSVs from the other Google account and add them as original dated snapshots under archive/imports/**after review**. The notebooks available in this user's ChatGPT Library are indexed separately; some original checkpoints/results may remain in Google Drive only.
