# Integrated Gradients analysis

`run_final_mixed_ig.py` contains the audited base analysis workflow.

`run_final_mixed_ig_v2.py` replaces the numerical IG integrator while retaining
the model loading, sample selection, attention extraction, and evaluation
workflow of the base implementation.

## Manuscript-reported run

- split mode: mixed
- split seed: 42
- optimization seed: 42
- canonical structures: 413
- observations per structure: 1
- baseline: learned gene-identity embeddings
- quadrature: Gauss-Legendre
- quadrature points: 128
- checkpoint SHA256: `cb9434a0e1d2e28e0328ffcc1720da5cf2555555382b11bbc3e392c8a7c9126c`

A separate 64-point convergence run produced nearly identical agreement
statistics.

Derived outputs for the reported run are provided under
`results/integrated_gradients/`.
