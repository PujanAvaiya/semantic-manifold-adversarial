# Research notebook index (inventory snapshot)

**Status as of 2026-11-02.** GitHub import copies have notebook outputs/execution state cleared. The originals remain in the user's file library and, where applicable, in Google Drive. This inventory does **not** imply that every file was fully executed or that its results are validated.

## Reviewed research notebooks committed on this branch

| # | Topic | Source notebook | Sanitized GitHub copy |
|---:|---|---|---|
| 1 | Synthetic tangent ground truth | Ground_Truth_Tangent_Recovery_Cross_Manifold_Extended.ipynb | [Open notebook](../../notebooks/research_archive/01_synthetic_tangent_recovery.ipynb) |
| 2 | Noise-scale selection | Noise_Aware_Tangent_Selector_Colab.ipynb | [Open notebook](../../notebooks/research_archive/02_noise_aware_tangent_selector.ipynb) |
| 3 | Comparison to fixed spectral head | LHSD_vs_Calibrated_Geometry_Benchmark.ipynb | [Open notebook](../../notebooks/research_archive/03_lhsd_derived_vs_calibrated.ipynb) |
| 4 | Orientation uncertainty | Orientation_Reliability_Experiments_1_2_3_Colab.ipynb | [Open notebook](../../notebooks/research_archive/04_orientation_reliability_diagnostics.ipynb) |
| 5 | Independent-seed reliability | Geometry_Reliability_Followup_Independent_Seeds_Colab.ipynb | [Open notebook](../../notebooks/research_archive/05_independent_seed_followup.ipynb) |
| 6 | Spectral error bound feasibility | Score_Geometry_Error_Bound_Feasibility_Colab.ipynb | [Open notebook](../../notebooks/research_archive/06_spectral_error_bound_feasibility.ipynb) |
| 7 | MNIST pixel geometry | MNIST_Digit1_Final_Geometry.ipynb | [Open notebook](../../notebooks/research_archive/07_mnist_digit1_score_geometry.ipynb) |
| 8 | MNIST direction extraction | Task2_MNIST_Pixel_Space_Directions_Colab.ipynb | [Open notebook](../../notebooks/research_archive/08_mnist_pixel_space_directions.ipynb) |
| 9 | MNIST calibration | MNIST_Calibrated_UCB_Task2_Integrated_Colab.ipynb | [Open notebook](../../notebooks/research_archive/09_mnist_calibrated_task2.ipynb) |
| 10 | Dimension stress test | MNIST_ID_Stress_Test.ipynb | [Open notebook](../../notebooks/research_archive/10_dimension_stress_benchmark.ipynb) |
| 11 | Transfer hypothesis test — real run pending | MNIST_Tangent_Directions_Transferability_Experiment_Colab.ipynb | [Open notebook](../../notebooks/research_archive/11_mnist_tangent_transfer_alignment.ipynb) |
| 12 | Black-box baseline | MNIST_HSJA_TtBA_BlackBox_Benchmark_Colab.ipynb | [Open notebook](../../notebooks/research_archive/12_hsja_ttba_blackbox_benchmark.ipynb) |
| 13 | VAE PCA attack | MNIST_1_to_7_PCA_32D_20PC_Colab.ipynb | [Open notebook](../../notebooks/research_archive/13_vae_pca_attack_pilot.ipynb) |
| 14 | CIFAR VAE pilot | CIFAR10_VAE_Tangent_Attack.ipynb | [Open notebook](../../notebooks/research_archive/14_cifar_vae_tangent_attack.ipynb) |
| 15 | EBM sampling | mnist_ebm_langevin_colab.ipynb | [Open notebook](../../notebooks/research_archive/15_mnist_ebm_langevin.ipynb) |
| 16 | EBM versus PCA | clean_mnist_ebm_vs_pca_decisive_colab.ipynb | [Open notebook](../../notebooks/research_archive/16_clean_mnist_ebm_vs_pca.ipynb) |

## Existing notebooks from the original GitHub research structure

The repository already contains numbered notebooks under [notebooks/](../../notebooks/) and docs for earlier latent-VAE work. They are not replaced by this import. Some prior notebooks are scaffolds rather than completed experimental runs.

## Located in ChatGPT Library; not yet incorporated into the curated branch

**64 additional notebook filenames** are discoverable in the Library. Many are variant versions and experiments; original output retention, code provenance, redundancy and access rights need review before public import. In particular, the large historical VAE/PDPGD notebooks remain source artifacts, not silently rewritten substitutes.

### Adversarial / VAE (15)

- Cross_Manifold_Transfer_Test_Colab(1).ipynb
- Cross_Manifold_Transfer_Test_Colab.ipynb
- Cross_Manifold_Transfer_Test_No_Upload_Colab.ipynb
- MNIST_1_to_7_PCA_32D_20PC_Colab(1).ipynb
- MNIST_HSJA_TtBA_BlackBox_Benchmark_Colab(1).ipynb
- PDPGD (1).ipynb
- PDPGD.ipynb
- VAE_on_MNIST (1).ipynb
- VAE_on_MNIST (2).ipynb
- VAE_on_MNIST (3).ipynb
- VAE_on_MNIST (4).ipynb
- VAE_on_MNIST (5)(1).ipynb
- VAE_on_MNIST (5).ipynb
- VAE_on_MNIST.ipynb
- cifar_low_cost_stability_colab.ipynb

### Score / EBM geometry (26)

- LHSD_pretrained_score_comparison_FIXED(1).ipynb
- LHSD_pretrained_score_comparison_FIXED.ipynb
- LHSD_pretrained_score_comparison_colab.ipynb
- Untitled8_experiment5_dimension_estimation.ipynb
- Untitled8_experiment5_score_probe_SVD.ipynb
- adaptive_lhsd_mnist_colab(1).ipynb
- adaptive_lhsd_mnist_colab.ipynb
- auto_scale_ebm_dimension.ipynb
- conformal_ebm_final_checkpoint_colab(1).ipynb
- conformal_ebm_final_checkpoint_colab.ipynb
- dimension_accuracy_colab_FIXED.ipynb
- dimension_accuracy_stress_test_colab.ipynb
- ebm_sigma_bias_dimension_estimators.ipynb
- energy_geometry_conditions_mnist_colab(1).ipynb
- energy_geometry_conditions_mnist_colab.ipynb
- finite_data_repeated_seed_benchmark(1).ipynb
- finite_data_repeated_seed_benchmark.ipynb
- finite_data_repeated_seed_benchmark_with_torus_controls.ipynb
- frozen_ebm_adaptive_geometry_research_colab.ipynb
- highd_pixel_ebm_geometry.ipynb
- mnist_ebm_langevin_colab(1).ipynb
- mnist_frozen_ebm_paper_comparison_colab.ipynb
- residual_ebm_two_variants_colab.ipynb
- residual_ebm_two_variants_teacher_fixed_colab(1).ipynb
- residual_ebm_two_variants_teacher_fixed_colab.ipynb
- torus_rank_final_colab.ipynb

### Calibration / validation (11)

- Geometry_Reliability_Followup_Independent_Seeds_Colab(1).ipynb
- Ground_Truth_Tangent_Recovery_Colab(1).ipynb
- Ground_Truth_Tangent_Recovery_Colab(2).ipynb
- Ground_Truth_Tangent_Recovery_Colab.ipynb
- MNIST_Original_vs_Denoised_CALIBRATED.ipynb
- MNIST_Reliability_Calibration_Standalone.ipynb
- MNIST_Scale_Selection_Repair_Colab.ipynb
- Noise_Aware_Tangent_Selector_Colab(1).ipynb
- calibration.ipynb
- noise_scale_selector_research_colab.ipynb
- proposal_paper_style_benchmark_colab.ipynb

### Exploratory / original (12)

- MNIST_Original_vs_Denoised_Colab.ipynb
- Simple_MNIST (1).ipynb
- Simple_MNIST.ipynb
- Simple_MNIST_fixed (1).ipynb
- Simple_MNIST_fixed.ipynb
- Task2_MNIST_Pixel_Space_Directions_Colab(1).ipynb
- Untitled5 (1).ipynb
- Untitled5.ipynb
- Untitled8.ipynb
- Untitled8_experiment4_movement.ipynb
- Untitled8_fixed_geometry.ipynb
- final_pixel_manifold_decision.ipynb

## Python scripts still requiring separate source-code review (36)

These scripts are available in the Library but were not automatically published. The runnable notebooks above contain corresponding code for the selected experiments.

- Ground_Truth_Tangent_Recovery.py
- MNIST_Digit1_Final_Geometry.py
- MNIST_HSJA_TtBA_BlackBox_Benchmark_Colab.py
- MNIST_Original_vs_Denoised.py
- MNIST_Reliability_Calibration_Addon.py
- Noise_Aware_Tangent_Selector.py
- append_torus_controls_to_existing_colab.py
- auto_scale_ebm_dimension.py
- cifar_low_cost_stability.py
- conformal_ebm_final_gate.py
- dimension_accuracy_stress_benchmark.py
- ebm_geometry_research_benchmark.py
- ebm_sigma_bias_dimension_estimators.py
- energy_geometry_conditions_mnist.py
- experiment4_manifold_movement.py
- experiment5_estimated_dimension.py
- experiment5_score_probe_svd.py
- final_pixel_manifold_decision.py
- finite_data_repeated_seed_benchmark.py
- geometry_evaluation_fixed.py
- highd_ebm_geometry_diagnostic(1).py
- highd_ebm_geometry_diagnostic.py
- highd_matrix_free_power_audit(1).py
- highd_matrix_free_power_audit.py
- highd_pixel_ebm_geometry.py
- lhsd_adaptive_mnist.py
- lhsd_pretrained_mnist_bridge.py
- lhsd_pretrained_mnist_bridge_fixed.py
- mnist_frozen_ebm_paper_comparison.py
- multiscale_ebm_decisive.py
- noise_scale_analysis.py
- noise_scale_projector_benchmark.py
- proposal_paper_style_benchmark.py
- residual_ebm_two_variants.py
- residual_ebm_two_variants_teacher_fixed.py
- torus_rank_final.py

## Files held outside this repository

- Raw per-point tables (e.g. full multiseed 8MB diagnostics), trained models and checkpoints, runtime caches and raw MNIST data are excluded from this branch.
- Colab files that exist **only** in another Gmail's Google Drive cannot be enumerated from here; use that account's Download .ipynb and upload the exports for review.
- Real transfer alignment output from the currently running new notebook is **not yet imported**. Do not report it as a completed experimental result.
