# Paper-Style Validation Summary

## Verified

- The solver reproduces closed-form Mittag-Leffler test problems on a fixed grid, with best/worst absolute L2 errors of `0.000e+00` / `1.086e-06` over the reported sweep.
- The solver demonstrates convergence on manufactured solutions across smooth, weakly singular, and trial-space-aligned families; best/worst absolute L2 errors in that suite are `1.707e-07` / `3.520e-04`.

## Quantitatively Compared

- A limited Kothari subset with published zero-initial conditions is compared quantitatively under model-reduction and PDF-digitization assumptions only, with moderate RMSE rather than strong one-to-one agreement.

## Qualitatively Compared

- Kothari replay scenarios with inferred initial state or inferred switching remain descriptive only (10 replay rows).
- Gomez and Wang are retained as external structural references and do not support strict quantitative validation claims.

## Not Supported / Limitations

- History-dependent initialization is not implemented; the current solver supports pointwise initial condition only.
- Lopez is therefore a limitation study, not an accuracy benchmark.
- Benchmarks based only on digitized curves remain approximate and should not be described as raw experimental validation.