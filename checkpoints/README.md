# Checkpoints

Model weights are intentionally not committed to normal Git history.

Expected local filenames for the current MNIST baseline are:

```text
checkpoints/
├── vae_model.pth
└── victim_cnn.pth
```

The original exploratory notebook saved these weights during the experiment. Reproducible training/loading will be moved into the numbered notebooks. If checkpoints later need to be shared, prefer a release artifact, Git LFS, or a model registry rather than committing large binary files directly.
