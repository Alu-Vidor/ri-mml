# Caputo-Mainardi Silver Relaxation Benchmark (T1)

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
        interval_end=1,
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
| T1 | 1.000000e+00 | L1 finite difference | 200 | 2.933000e-04 | 4.920214e-01 | 2.639549e-02 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 500 | 2.966000e-04 | 3.886559e-01 | 1.405645e-02 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 1000 | 3.384000e-04 | 3.138375e-01 | 8.489766e-03 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 2000 | 4.070000e-04 | 2.462308e-01 | 5.026692e-03 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 5000 | 8.234000e-04 | 1.715734e-01 | 2.407367e-03 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 10000 | 1.369700e-03 | 1.276438e-01 | 1.364467e-03 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 20000 | 2.527200e-03 | 9.354008e-02 | 7.636849e-04 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 50000 | 5.704900e-03 | 6.106676e-02 | 3.590025e-04 | nan | nan |
| T1 | 1.000000e+00 | L1 finite difference | 100000 | 1.624550e-02 | 4.385247e-02 | 2.061770e-04 | nan | nan |
| T1 | 1.000000e+00 | Spectral method | 4 | 1.576770e-02 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 6 | 2.996530e-02 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 8 | 4.164870e-02 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 10 | 5.535000e-02 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 12 | 5.160600e-02 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 16 | 7.386130e-02 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 20 | 9.676110e-02 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 24 | 1.150711e-01 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |
| T1 | 1.000000e+00 | Spectral method | 32 | 1.538407e-01 | 2.775558e-15 | 1.453304e-17 | 1.000000e+00 | 0.000000e+00 |

## Summary table

| comparison | method | DoF | runtime_seconds | err_inf | err_L2 |
| --- | --- | --- | --- | --- | --- |
| best Spectral method | Spectral method | 4 | 1.576770e-02 | 2.775558e-15 | 1.453304e-17 |
| best L1 finite difference | L1 finite difference | 100000 | 1.624550e-02 | 4.385247e-02 | 2.061770e-04 |
| closest runtime | Spectral method | 4 | 1.576770e-02 | 2.775558e-15 | 1.453304e-17 |
| closest runtime | L1 finite difference | 100000 | 1.624550e-02 | 4.385247e-02 | 2.061770e-04 |
| closest DoF | Spectral method | 32 | 1.538407e-01 | 2.775558e-15 | 1.453304e-17 |
| closest DoF | L1 finite difference | 200 | 2.933000e-04 | 4.920214e-01 | 2.639549e-02 |

## Key observations

- The characteristic initial-layer scale is epsilon^(1/nu) = epsilon^2 = 6.297140e-04, so a uniform L1 grid must spend many steps resolving the very start of the trajectory.
- Best Spectral method Linf error: 2.775558e-15 at DoF = 4, runtime = 1.576770e-02 s.
- Best L1 finite difference Linf error: 4.385247e-02 at DoF = 100000, runtime = 1.624550e-02 s.
- Closest-runtime comparison: Spectral method DoF = 4 gives Linf = 2.775558e-15, while L1 finite difference DoF = 100000 gives Linf = 4.385247e-02.
- Closest-DoF comparison: Spectral method DoF = 32 gives Linf = 2.775558e-15, while L1 finite difference DoF = 200 gives Linf = 4.920214e-01.

## Artifacts

- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\caputo_mainardi_T1_results.csv`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\tables\caputo_mainardi_T1_summary.csv`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T1_solution_full.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T1_solution_zoom_0p01.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T1_solution_zoom_0p002.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T1_solution_sqrt_t.png`
- `C:\Users\danii\Desktop\Diss\ri-mml\benchmarks\results\plots\caputo_mainardi_T1_accuracy.png`