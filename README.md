# Quantum-Inspired Feature Selection via QUBO

A redundancy-aware feature selection method formulated as a Quadratic Unconstrained
Binary Optimization (QUBO) problem, solved with classical simulated annealing and
GPU-based quantum-inspired annealing (Fixstars Amplify AE), and benchmarked against
a standard classical baseline (`SelectKBest`).

## Motivation

Standard filter-based feature selection methods (e.g. mutual-information ranking)
select features purely by individual relevance to the target variable. They have
no mechanism to penalize **redundancy** — if two features are highly correlated
with each other, a relevance-only method may select both, wasting a feature
"slot" on near-duplicate information.

This project reformulates feature selection as a QUBO problem that jointly
optimizes for:
- **High relevance** — each selected feature should be informative about the target.
- **Low redundancy** — selected features should be minimally correlated with each other.

This is a natural fit for annealing-based solvers (quantum and quantum-inspired),
since QUBO is their native problem format.

## Problem Formulation

For each feature `i`, we define a binary decision variable `x_i` (1 = selected,
0 = not selected). The objective to **minimize** is:

```
H(x) = -Σ relevance_i * x_i + α * Σ redundancy_ij * x_i * x_j   (i < j)
```

Where:
- `relevance_i` — mutual information between feature `i` and the target label
- `redundancy_ij` — absolute Pearson correlation between features `i` and `j`
- `α` — tunable trade-off weight controlling how strongly redundancy is penalized
  (also implicitly controls how many features get selected, since higher α favors
  smaller, less-correlated sets)

This maps directly onto a QUBO matrix `Q`, with linear terms `Q[i,i] = -relevance_i`
and quadratic terms `Q[i,j] = α * redundancy_ij`.

## Dataset

`sklearn.datasets.load_breast_cancer` — 30 numeric features, binary classification
(malignant/benign), 569 samples. Chosen for its small, well-understood feature set
that makes QUBO solutions interpretable by inspection.

## Methodology

1. Compute relevance (mutual information) and redundancy (correlation matrix) for
   all 30 features.
2. Build the QUBO for a given `α`.
3. Solve with two independent annealing-based methods:
   - **Local classical simulated annealing** — `dimod.SimulatedAnnealingSampler`
   - **Cloud GPU-based quantum-inspired annealing** — Fixstars Amplify AE
4. Compare the selected feature set against a classical baseline
   (`SelectKBest` with mutual information) at matching `k`.
5. Evaluate both feature sets downstream with a `RandomForestClassifier`
   (70/30 train/test split, fixed random seed).

Real D-Wave QPU access was not available for this iteration (see Limitations);
Fixstars Amplify AE served as the quantum-inspired hardware benchmark.

## Results

| k | Classical (SelectKBest) | QUBO (redundancy-aware) | Classical Acc | QUBO Acc | Gap |
|---|---|---|---|---|---|
| 2 | worst perimeter, worst area | fractal dimension error, worst perimeter | 90.06% | 90.64% | +0.58% |
| 3 | worst radius, worst perimeter, worst area | mean radius, mean symmetry, worst area | 90.64% | 95.91% | +5.27% |
| 4 | mean concave points, worst radius, worst perimeter, worst area | mean perimeter, mean symmetry, concavity error, worst area | 91.81% | 97.66% | +5.85% |

**Key finding:** at every tested `k`, `SelectKBest` repeatedly stacks near-duplicate
"size" measurements (`worst radius`, `worst perimeter`, `worst area` are all highly
correlated tumor-size metrics), since it ranks by relevance alone. The QUBO
formulation's redundancy penalty consistently swaps in a feature from a different
measurement axis (e.g. `mean symmetry`, `concavity error`) instead — and this
diversification produces a growing accuracy advantage as `k` increases, since
classical selection wastes an increasing fraction of its budget on redundant
information.

Both annealing solvers (local simulated annealing and Fixstars Amplify AE)
converged on identical feature sets at `k=2`, supporting that the QUBO formulation
is stable and not an artifact of one solver's randomness.

## Limitations

- **Alpha sensitivity**: at `k=4`, several nearby `α` values (0.20–0.28) all produced
  4-feature solutions, but with substantially different feature sets. This indicates
  the QUBO's solution landscape is not sharply peaked near that region — results
  should be read as "a redundancy-aware formulation consistently outperforms
  relevance-only selection across sampled configurations," not as a claim of a
  single globally optimal α per k.
- **No real QPU results**: Fixstars Amplify AE is a GPU-based quantum-inspired
  solver, not physical quantum annealing hardware. Real D-Wave QPU access (via
  D-Wave Leap or Fixstars' D-Wave integration) is identified as future work.
- **Single dataset**: results are demonstrated on one dataset (30 features). Broader
  validation across datasets of varying size and correlation structure would
  strengthen the generality of the finding.

## Future Work

- Run the identical QUBO on real D-Wave QPU hardware once access is available, to
  compare physical quantum annealing results against the simulated/GPU-based runs.
- Systematically sweep `α` with multiple samples per value to characterize solution
  stability more rigorously.
- Extend to additional datasets to test generality of the redundancy-aware advantage.

## Setup

```bash
pip install dwave-ocean-sdk amplify scikit-learn numpy
```

```python
# See notebook for full code:
# 1. Load data + compute relevance/redundancy
# 2. Build QUBO
# 3. Solve with dimod.SimulatedAnnealingSampler (local, no token needed)
# 4. Solve with Fixstars Amplify AE (requires free AE Basic token from
#    https://amplify.fixstars.com)
# 5. Compare against SelectKBest baseline
```

## Tech Stack

- `dimod` / `dwave-ocean-sdk` — QUBO/BQM construction, local simulated annealing
- `amplify` (Fixstars) — GPU-based quantum-inspired annealing
- `scikit-learn` — classical baseline, mutual information, classifier evaluation
