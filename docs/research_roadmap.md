# Eight-Week Research Roadmap

This document is the authoritative research plan for the project. The order matters: each later stage should only be added when the earlier diagnostics justify it.

## Guiding principle

Do not add complexity because it is theoretically attractive. First run the simplest controlled experiment, measure where it fails, and only then introduce the smallest change that addresses the observed failure mode.

```text
understand geometry
        ↓
stress-test the current attack
        ↓
use diagnostics to identify failure modes
        ↓
fix only the failure modes that actually occur
        ↓
remove label dependence
        ↓
establish deterministic transfer baselines
        ↓
only then consider a learned distribution over directions
        ↓
ensemble/unseen-model transfer
        ↓
ablations + failure analysis + contribution statement
```

---

## Week 1 — Geometry foundations

### Goal
Understand the geometric assumptions behind the current method before changing the attack.

### Topics
- TNAR geometry
- ManiFool geometry
- Jacobians of encoder/decoder/classifier mappings
- local linearization
- tangent-space and normal-space decomposition
- relationship between PCA-estimated tangent directions and Jacobian-derived tangent directions
- decision-boundary normal versus data-manifold tangent directions

### Questions to answer
1. What exactly is the manifold in the current MNIST/VAE setting?
2. At a latent point `z`, what object is being approximated by the local PCA basis?
3. When should a perturbation lie in the tangent space, normal space, or contain both components?
4. When is the local linear approximation expected to break down?
5. How do TNAR and ManiFool motivate tangent/normal decompositions differently from the current implementation?

### Deliverable
A geometry notebook and notes containing equations, diagrams, and numerical sanity checks for Jacobians, tangent bases, normal components, orthogonality, and local linearization.

No algorithmic modification should be justified solely from Week 1 theory.

---

## Week 2 — Run the current attack across many digit 1 samples

### Goal
Determine whether the existing single-example behavior is representative or anecdotal.

### Protocol
Keep the current method fixed. For many correctly classified digit `1` inputs:

1. encode the source image;
2. verify the VAE reconstruction preserves the intended source semantics/prediction;
3. retrieve neighbors using the current fixed rule;
4. estimate the local tangent basis using the current fixed `r`/dimension rule;
5. run exactly the same attack objective and stopping criterion;
6. save all diagnostics for both successes and failures.

### Diagnostics
At minimum record:
- source confidence and target confidence before/after
- attack success/failure
- number of optimization steps
- latent perturbation norm
- decoded image-space distortion
- neighborhood distances
- neighborhood class purity under the current labeled setup
- singular values / explained variance
- selected tangent dimension `r`
- projection of useful/optimized direction into tangent and normal components
- decision margin along the path
- whether the path crosses the boundary and where
- reconstruction quality and semantic plausibility

### Deliverable
A sample-level results table plus aggregate plots and representative success/failure cases.

### Decision rule
Do **not** change neighbor selection, `r`, alignment, or relinearization during this week. The purpose is diagnosis.

---

## Week 3 — Fix only what Week 2 says is necessary

### Goal
Introduce adaptive geometry only when a measured failure mode motivates it.

### Candidate modifications

#### A. Adaptive neighbors
Use only if fixed-neighbor retrieval is unstable, contaminated, or geometrically inconsistent.

Possible diagnostics that justify it:
- sharp jumps in neighbor distance
- poor local reconstruction
- mixed local geometry
- unstable PCA basis across nearby `k`

#### B. Adaptive tangent dimension `r`
Use only if a fixed `r` discards important local variation or includes noisy directions.

Possible rules to test:
- cumulative explained variance threshold
- eigengap / singular-value gap
- stability of the estimated subspace

#### C. Target alignment
Use only if the target-relevant direction is systematically poorly represented by the current tangent basis.

Measure target alignment before adding a target-aware basis or ranking rule.

#### D. Relinarization / re-estimation along the path
Use only if long perturbation paths leave the validity region of the original local linearization.

Evidence should include curvature, basis drift, increasing reconstruction/local-linearity error, or attack failures correlated with path length.

### Ablation rule
Each modification must be evaluated independently before combinations are tested.

### Deliverable
A diagnostic-to-fix table:

| Observed failure | Evidence | Proposed fix | Ablation result | Keep? |
| --- | --- | --- | --- | --- |

---

## Week 4 — Remove labels from neighbor retrieval

### Goal
Test whether the method can construct useful semantic neighborhoods without using ground-truth class labels.

### Baseline
Week 2/3 same-class labeled-neighbor retrieval.

### New setting
Retrieve semantic neighbors using a self-supervised or representation-based similarity rule rather than labels.

Examples to investigate experimentally:
- encoder-feature nearest neighbors
- self-supervised image embeddings
- reconstruction/latent similarity with consistency filters

### Key comparison
Compare labeled versus label-free neighborhoods on:
- semantic purity
- local geometry stability
- tangent-subspace similarity
- attack success
- distortion
- failure modes

### Deliverable
Evidence for whether label-free semantic neighborhoods are sufficiently reliable for the downstream geometry.

---

## Week 5 — Deterministic transfer baselines and sampling around `a*`

### Goal
Before learning a distribution, establish what a deterministic optimized tangent coefficient vector already transfers.

Let `a*` denote the optimized tangent-coordinate solution for a source example/target.

### Stage 5A — Deterministic transfer baseline
Test the exact optimized direction/perturbation on:
- independently initialized versions of the same architecture
- at least one different victim architecture
- multiple source examples

Record white-box/source success separately from transfer success.

### Stage 5B — Simple sampling around `a*`
Before training any generative distribution over directions, test whether local stochastic exploration already improves transfer:

```text
a = a* + noise
```

or other simple controlled sampling in tangent-coordinate space.

Measure:
- transfer success distribution
- source-model success
- norm/distortion
- direction diversity
- sensitivity to sampling scale

### Decision gate for Week 6
Proceed to `q_psi(a | z, t)` **only if** Stage 5 indicates that a useful transferable region/distribution exists around or beyond the deterministic optimum.

If sampling provides no meaningful evidence of a transferable direction distribution, do not build a learned probabilistic model just because it is available.

---

## Week 6 — Conditional direction distribution `q_psi(a | z, t)` — conditional stage

### Entry condition
Week 5 must be promising.

### Goal
Model a distribution over useful tangent-coordinate perturbations conditioned on source latent representation `z` and target `t`:

`q_psi(a | z, t)`

### Questions
- What should `a` represent: complete perturbation, residual around a deterministic optimum, or a low-dimensional transferable direction?
- What training signal defines a good sample?
- How should source success, transfer success, norm, and diversity trade off?
- Is the learned distribution better than simple noise around `a*`?

### Required baselines
A learned model is only meaningful if compared with:
- deterministic `a*`
- isotropic/random tangent sampling
- simple local sampling around `a*`

### Deliverable
A derivation of the objective plus a controlled implementation only after the deterministic baselines are established.

---

## Week 7 — Surrogate ensemble and unseen architectures

### Goal
Measure whether the method captures transferable geometry rather than overfitting one classifier.

### Experiments
- optimize/evaluate with a surrogate ensemble
- hold out victim architectures completely from optimization
- test on unseen initialization seeds
- separate same-architecture transfer from cross-architecture transfer

### Report
For every target model distinguish:
- seen during optimization?
- architecture seen?
- initialization seen?
- source-model success
- unseen-model transfer success
- distortion and semantic metrics

### Deliverable
A transfer matrix across surrogate and held-out models.

---

## Week 8 — Ablations, plots, failure analysis, and contribution identification

### Goal
Stop adding components and determine what the experiments actually support.

### Mandatory ablations
Depending on which stages survive earlier testing:
- pixel/direct latent baseline versus tangent-constrained method
- fixed versus adaptive neighbors
- fixed versus adaptive `r`
- target alignment on/off
- one-shot tangent basis versus relinearization
- labeled versus self-supervised neighbors
- deterministic `a*` versus local sampling
- single surrogate versus surrogate ensemble
- learned `q_psi` versus simple sampling, if Week 6 was entered

### Failure analysis
Categorize failures rather than hiding them. Examples:
- poor VAE reconstruction
- bad neighborhood
- unstable tangent estimate
- target direction outside local tangent space
- excessive curvature / stale linearization
- source attack succeeds but transfer fails
- transfer requires excessive semantic change

### Final question
What is the smallest claim supported by the evidence?

Possible contributions must emerge from the results; they are not fixed in advance.

### Deliverable
- final plots
- ablation tables
- failure taxonomy
- reproducible experiment settings
- explicit contribution statement
- limitations and negative results

---

# Stage gates

The roadmap contains three important gates:

```text
Week 2 diagnostics
      ↓
Only observed problems enter Week 3

Week 4 label-free retrieval
      ↓
Must be competitive enough to justify a label-independent pipeline

Week 5 deterministic + simple sampling transfer
      ↓
Promising? ── no ──> skip learned q_psi
      │
     yes
      ↓
Week 6 q_psi(a | z,t)
```

This keeps the project evidence-driven and prevents the final method from becoming a collection of unmotivated components.