# Caputo-Mainardi Silver Relaxation Benchmark (T10)

Source article: M. Caputo, F. Mainardi, 'A New Dissipation Model Based on Memory Mechanism', Pure and Applied Geophysics, 1971.

## Physical setup

This benchmark models stress relaxation in silver after a unit step strain in the fractional standard linear solid of Caputo-Mainardi.
For Table 1 material parameters we use Silver with delta = 0.55, alpha = 39.3, nu = 0.50, hence beta = alpha + delta = 39.85 and epsilon = 1 / beta.
The normalized relaxation law is the Mittag-Leffler response u(t) = E_{0.5}(-39.85 t^{0.5}), which combines a sharp initial transition with a long memory tail.

## Solver call

The spectral solver is used as a black box through:

```python
solve_fractional_ivp(
    ProblemConfig(
        alpha=0.5,
        epsilon=0.0250941028858218,
        interval_end=10,
        u0=1.0,
        coefficient=lambda x: np.ones_like(x),
        forcing=lambda x: np.zeros_like(x),
        reference_coefficient=1.0,
    ),
    SolverRunConfig(basis_size=N),
)
```

## Detailed results

| case_id | T | method | DoF | runtime_seconds | err_inf | err_L2 | condition_number | relative_residual_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T10 | 1.000000e+01 | L1 finite difference | 200 | 3.613000e-04 | 7.218260e-01 | 1.094223e-01 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 500 | 4.324000e-04 | 6.396996e-01 | 6.375689e-02 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 1000 | 4.422000e-04 | 5.685596e-01 | 4.146659e-02 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 2000 | 4.362000e-04 | 4.920052e-01 | 2.639112e-02 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 5000 | 8.601000e-04 | 3.883871e-01 | 1.400386e-02 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 10000 | 1.419400e-03 | 3.135284e-01 | 8.439496e-03 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 20000 | 3.038300e-03 | 2.458842e-01 | 4.976983e-03 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 50000 | 6.960100e-03 | 1.715615e-01 | 2.406209e-03 | nan | nan |
| T10 | 1.000000e+01 | L1 finite difference | 100000 | 1.681280e-02 | 1.276307e-01 | 1.363440e-03 | nan | nan |
| T10 | 1.000000e+01 | Spectral method | 4 | 1.594650e-02 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 6 | 2.263510e-02 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 8 | 3.699220e-02 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 10 | 4.731170e-02 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 12 | 5.413330e-02 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 16 | 7.418290e-02 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 20 | 9.631950e-02 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 24 | 1.158505e-01 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |
| T10 | 1.000000e+01 | Spectral method | 32 | 1.566095e-01 | 2.331468e-15 | 1.627293e-17 | 1.000000e+00 | 0.000000e+00 |

## Summary table

| comparison | method | DoF | runtime_seconds | err_inf | err_L2 |
| --- | --- | --- | --- | --- | --- |
| best Spectral method | Spectral method | 4 | 1.594650e-02 | 2.331468e-15 | 1.627293e-17 |
| best L1 finite difference | L1 finite difference | 100000 | 1.681280e-02 | 1.276307e-01 | 1.363440e-03 |
| closest runtime | Spectral method | 4 | 1.594650e-02 | 2.331468e-15 | 1.627293e-17 |
| closest runtime | L1 finite difference | 100000 | 1.681280e-02 | 1.276307e-01 | 1.363440e-03 |
| closest DoF | Spectral method | 32 | 1.566095e-01 | 2.331468e-15 | 1.627293e-17 |
| closest DoF | L1 finite difference | 200 | 3.613000e-04 | 7.218260e-01 | 1.094223e-01 |

## Key observations

- The characteristic initial-layer scale is epsilon^(1/nu) = epsilon^2 = 6.297140e-04, so a uniform L1 grid must spend many steps resolving the very start of the trajectory.
- Best Spectral method Linf error: 2.331468e-15 at DoF = 4, runtime = 1.594650e-02 s.
- Best L1 finite difference Linf error: 1.276307e-01 at DoF = 100000, runtime = 1.681280e-02 s.
- Closest-runtime comparison: Spectral method DoF = 4 gives Linf = 2.331468e-15, while L1 finite difference DoF = 100000 gives Linf = 1.276307e-01.
- Closest-DoF comparison: Spectral method DoF = 32 gives Linf = 2.331468e-15, while L1 finite difference DoF = 200 gives Linf = 7.218260e-01.

## Artifacts

- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\caputo_mainardi_T10_results.csv`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\tables\caputo_mainardi_T10_summary.csv`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T10_solution_full.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T10_solution_zoom_0p01.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T10_solution_zoom_0p002.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T10_solution_sqrt_t.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T10_accuracy.png`