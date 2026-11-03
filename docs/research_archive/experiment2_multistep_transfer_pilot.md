# Experiment 2 — Multi-step transfer pilot: fixed score bank versus full gradient

Status: PILOT, 24 eligible MNIST digit-1 images, four independently trained classifier instances, 12 iterative surrogate-gradient steps. Results copied from figures and a truncated bootstrap table supplied on 2026-11-02. Full row-level results have not been imported and exact statistics must be checked before publication.

## Tested question and controls
Full-space PGD was compared with a finite frozen score-filtered probe bank (t=5), PCA bank, random bank, and 50/50 normalized direction hybrid (full gradient plus fixed score bank). Each attack uses surrogate gradients; target models are used for evaluation only. Constraint pairs are (L2=2,Linf=.40), (3,.55), (4,.70) in [-1,1] pixels. These are caps, not proof of exactly matched achieved distortion. Geometry bank is NOT an actual certified tangent projector or exact manifold retraction.

## Approximate ASR from the budget curve
| Method | L2<=3, Linf<=.55 | L2<=4, Linf<=.70 |
|---|---:|---:|
| Full-space PGD | ~12% | ~35% |
| Fixed score-filtered bank t5 | ~0% | ~0% |
| PCA bank t5 | ~0% | ~0% |
| Random bank t5 | ~1% | ~10% |
| Hybrid (full+score bank) | ~5% | ~27% |

Mean target margin increases at the largest budget were approximately full 4.7, hybrid 4.05, score bank 1.8, PCA bank 1.3 and random bank 1.25 (plot readings, not CSV-exact). Full-space PGD had the largest average attack effect.

At (L2=3,Linf=.55), the full attack heatmap shows successful transfers concentrated on tiny ResNet as the target: CNN-A→ResNet ~50%; CNN-B→ResNet ~42%; MLP→ResNet ~17%; CNN-A→CNN-B ~33%. Most other target pairs were 0. Results vary strongly by target, and each pair has only 24 sampled images.

## Paired bootstrap excerpts
At (L2=2,Linf=.4), score-bank-minus-full target-margin difference -0.9891, 95% image-bootstrap CI [-1.0951,-0.8975]; hybrid-minus-full -0.3127, CI [-0.3579,-0.2686].

IMPORTANT: The expensive direct-score filter is a different mechanism from the finite score bank: it reapplies the frozen original-image score filter to each new surrogate gradient, and was evaluated on only eight images with CNN-A as source. At (L2=4,Linf=.7), direct score filtering minus PCA bank gave ASR difference +0.2500 with image-bootstrap CI [+0.1250,+0.4167], and minus random bank +0.2083 CI [0,+0.375]. This result is from eight images and 24 source/target observations, so do not pool it with the 24-image attack curve or infer direct-score advantage against full-space PGD.

## Interpretation for proposal
Supported: multi-step attacks can cause nonzero black-box transfer; hard restriction to this fixed finite score-filtered bank is ineffective; hybrid mixing recovers much of the strength of unrestricted PGD but remains below full-space PGD.

Not supported: gradient alignment ensures attack transfer; true manifold-tangent attacks fail; direct score filtering fails; semantic validity is preserved; or any geometry method improves held-out target ASR against full-space PGD.

The next efficient analysis should compare direct filtering and full-space PGD on the identical eight-image subset, inspect bank rank and retained gradient energy, and evaluate achieved L2 and Linf per method before broadening the experiment. Semantic checks are still needed.

Data provenance: five screenshot plots and truncated paired-bootstrap data provided by the user; raw multistep CSVs pending. The Colab notebook is named MNIST_Multistep_Transfer_PGD_Colab.ipynb and available in the user's ChatGPT Library. 