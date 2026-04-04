# RI-MML

Resolvent-induced Muntz-Mittag-Leffler Petrov-Galerkin solver for scalar fractional initial-value problems of the form

`epsilon D_t^alpha u(t) + a(t) u(t) = f(t),   0 < alpha < 1,   u(0) = u0`.

## Scope

The current implementation supports a pointwise initial condition only. It does not implement history-dependent or prehistory-dependent initialization.

This matters for validation:

- `verification` is strong: the repository now includes closed-form exact tests and manufactured-solution convergence studies.
- `validation` is narrow: only literature cases compatible with the current solver assumptions are allowed to make quantitative claims.
- `limitations` are explicit: Lopez-style history-dependent initialization is not supported, and several literature comparisons remain qualitative or replay-only.

## Install

```bash
pip install -e .[benchmarks]
```

## Run The Full Validation Pipeline

```bash
python -m benchmarks
```

Artifacts are written to:

- `benchmarks/results/` for machine-readable CSV/JSON outputs
- `benchmarks/results/tables/` for paper-ready tables
- `benchmarks/report/validation_report.md` for the full report
- `benchmarks/report/paper_validation_summary.md` for the paper-style summary

## Minimal Solver Example

```python
import numpy as np

from rimml import FractionalProblem, SolverConfig, SpectralPetrovGalerkinSolver


problem = FractionalProblem(
    alpha=0.5,
    epsilon=0.1,
    interval_end=1.0,
    u0=1.0,
    coefficient=lambda x: 2.0 + 0.25 * x,
    forcing=lambda x: np.ones_like(x),
    reference_coefficient=2.0,
)

solver = SpectralPetrovGalerkinSolver(problem, SolverConfig(basis_size=6))
solution = solver.solve()

grid = np.linspace(0.0, 1.0, 200)
u_values = solution.evaluate(grid)
```
