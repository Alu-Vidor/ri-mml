# Validation Report

## What this report validates

- Verification of the RI-MML implementation on closed-form Mittag-Leffler test problems in the same mathematical setting as the solver.
- Verification of convergence behavior on manufactured solutions, including smooth, weakly singular, and trial-space-aligned families.
- A narrow quantitative literature comparison for Kothari zero-initial scenarios that can be replayed without inferred initial state fitting.

## What this report does not validate

- It does not validate history-dependent initialization. The current solver supports pointwise initial condition only: `u(0)=u0`.
- It does not claim one-to-one replication of Lopez, Wang, or Gomez models when their published setup differs structurally from the present solver.
- It does not claim validation against raw experimental data when curves were digitized from PDF figures.

## 1. Verification: Closed-Form Exact Problems

These cases reproduce analytic Mittag-Leffler solutions with fixed evaluation grids and deterministic solver settings.

| case_id | alpha | epsilon | l2_abs | l2_rel | l_inf |
| --- | --- | --- | --- | --- | --- |
| mlf_charge_a0p75 | 0.35 | 0.1 | 2.8718649553850324e-08 | 3.2032816229589975e-08 | 1.984295202595021e-07 |
| mlf_charge_a1p50 | 0.65 | 0.01 | 6.198326363377246e-10 | 5.206858853603327e-10 | 3.594371467130486e-09 |
| mlf_charge_small_eps | 0.92 | 0.0001 | 8.61325637410106e-12 | 1.0770248019674062e-11 | 4.597588976196221e-11 |
| mlf_discharge_a1p00 | 0.85 | 0.005 | 0.0 | 0.0 | 0.0 |

Best absolute L2 error across exact cases: `0.000e+00`. Worst absolute L2 error across exact cases: `1.086e-06`.

## 2. Verification: Manufactured Convergence

Manufactured solutions are grouped into smooth analytic, weak endpoint singularity, and trial-space-aligned families.

| family | case_id | best_l2_abs | best_l2_rel | best_l_inf | max_condition_number | max_relative_residual_norm |
| --- | --- | --- | --- | --- | --- | --- |
| smooth_analytic | smooth_a035_eps1e-2 | 1.707475289844927e-07 | 5.27078466513302e-07 | 2.1078477828173625e-06 | 1.2361594654047223 | 2.954387729890414e-16 |
| smooth_analytic | smooth_a075_eps1e-3 | 3.0242847447805974e-07 | 2.8580352749170693e-06 | 6.180566167371393e-06 | 1.2427033687242528 | 2.4621594653004984e-16 |
| trial_space_aligned | aligned_a045_eps1e-3 | 6.177641016016663e-05 | 9.105946209889068e-05 | 0.0011926199847824 | 1.0 | 0.0 |
| trial_space_aligned | aligned_a080_eps1e-4 | 1.1090580084355592e-06 | 3.3254613841277426e-06 | 1.8374065965699737e-05 | 1.0 | 0.0 |
| weak_endpoint_singularity | singular_a035_eps1e-2 | 0.0001275302891744 | 0.0001341849794845 | 0.0041993258383596 | 1.096332325420398 | 3.269947258262186e-16 |
| weak_endpoint_singularity | singular_a075_eps5e-4 | 1.9025940617267023e-06 | 3.3458706112484456e-06 | 2.875051404985229e-05 | 1.0980909494860378 | 3.744360109504184e-16 |

Maximum observed local convergence-rate indicators by family: smooth_analytic: 15.26, trial_space_aligned: 10.51, weak_endpoint_singularity: 10.76.

## 3. Quantitatively Compared Literature Cases

Only Kothari zero-initial scenarios remain in this section. They use published Table 5 parameters and pointwise zero initial state, but comparison still relies on digitized curves from PDF figures.

| benchmark_id | figure | mode | model_match_level | data_source | initialization_type | claim_level | duration_s | rmse | mae | e_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kothari_strict_zero_initial_T10 | fig6 | strict | reduced | digitized_pdf | pointwise | quantitative_validation | 10.0 | 0.2595615307301043 | 0.2432354646952258 | 0.4197892028173031 |
| kothari_strict_zero_initial_T20 | fig6 | strict | reduced | digitized_pdf | pointwise | quantitative_validation | 20.0 | 0.2030369727707899 | 0.1891395345717064 | 0.3830651364538324 |
| kothari_strict_zero_initial_T30 | fig6 | strict | reduced | digitized_pdf | pointwise | quantitative_validation | 30.0 | 0.1582949820220523 | 0.1419443728406273 | 0.2879468652009898 |
| kothari_strict_zero_initial_T40 | fig6 | strict | reduced | digitized_pdf | pointwise | quantitative_validation | 40.0 | 0.0963283817625248 | 0.0733097914000882 | 0.2979482713908477 |
| kothari_strict_zero_initial_T50 | fig6 | strict | reduced | digitized_pdf | pointwise | quantitative_validation | 50.0 | 0.1452016812863276 | 0.1073745132742303 | 0.4571298751287946 |

Best strict Kothari RMSE: `9.633e-02`.
Observed strict-mode errors remain moderate rather than negligible, so this section supports only a cautious quantitative comparison.

## 4. Qualitatively Compared Literature Cases

These cases are retained as external references only. They may still report numerical distances, but those distances are not promoted to strict validation claims.

| benchmark_id | mode | model_match_level | data_source | initialization_type | claim_level | rmse | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kothari_replay_fig3a | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.1158759653607435 | curve replay under inferred initial state; not a one-to-one replication of the published model |
| kothari_replay_fig3b | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.1178996603845718 | curve replay under inferred initial state; not a one-to-one replication of the published model |
| kothari_replay_fig5b | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.6962307348233974 | curve replay under inferred initial state; not a one-to-one replication of the published model |
| kothari_replay_fig7b | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.3002243940108962 | curve replay under inferred initial state; not a one-to-one replication of the published model |
| kothari_replay_long_cycle | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.1403987670868166 | piecewise replay with switching times inferred from figure geometry |
| kothari_replay_nonzero_initial_T10 | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.2946759742997288 | curve replay under inferred initial state; not validation accuracy |
| kothari_replay_nonzero_initial_T20 | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.1856666948890445 | curve replay under inferred initial state; not validation accuracy |
| kothari_replay_nonzero_initial_T30 | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.1180491866746351 | curve replay under inferred initial state; not validation accuracy |
| kothari_replay_nonzero_initial_T40 | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.1247337329157996 | curve replay under inferred initial state; not validation accuracy |
| kothari_replay_nonzero_initial_T50 | replay | reduced | digitized_pdf | inferred | qualitative_validation | 0.1571249782646963 | curve replay under inferred initial state; not validation accuracy |
| gomez_qualitative_reference | external_reference | qualitative | digitized_pdf | pointwise | qualitative_validation | nan | Gomez is retained only as a structural reference for the qualitative trend with varying gamma. It is not used for strict quantitative validation of the present RI-MML solver. |
| wang_external_reference | external_reference | qualitative | manual_table_transcription | history-dependent | qualitative_validation | nan | Wang is treated as an external reference only. The published model structure is richer than the present one-term RI-MML solver, so no strict quantitative validation claim is made. |

Best replay-only Kothari RMSE: `1.159e-01`. This number is descriptive only and should not be read as validation accuracy.

Published Wang metrics are preserved below as external context, not as a scorecard for the present solver.

| temperature_C | published_mae_mV | published_mre_percent | published_rmse_mV |
| --- | --- | --- | --- |
| 0.0 | 26.4 | 1.45 | 32.8 |
| 25.0 | 37.3 | 2.1 | 45.1 |
| 45.0 | 28.2 | 1.52 | 35.4 |

## 5. Limitations

| benchmark_id | validation_category | source_paper | model_match_level | data_source | initialization_type | claim_level | message | not_supported | supported_surrogate_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lopez_initialization_limit | limitations | Lopez-Villanueva et al. 2022 history-dependent initialization study | surrogate | manual_table_transcription | history-dependent | limitation_only | Current RI-MML solver supports only a pointwise initial condition u(0)=u0. It does not implement history-dependent initialization, so Lopez is included only to document a boundary of applicability. | ['history-dependent initialized operators', 'prehistory-dependent CPE state reconstruction', 'one-to-one replay of Lopez decay curves with the present solver'] | ['analytic decay families g1/g2/g3 as external references', 'multi-exponential surrogate g4 comparison'] |

## 6. Reproducibility

- Single-command entrypoint: `python -m benchmarks`.
- Deterministic settings are stored in `benchmarks/configs/verification_suite.json`.
- Machine-readable artifacts are written to `benchmarks/results/` and paper-ready tables to `benchmarks/results/tables/`.
- Benchmark claims are separated into verification, quantitative literature comparison, qualitative external reference, and limitations.
- For constant-coefficient reductions the discrete matrix can collapse to the identity, so `condition_number = 1` and zero linear residual are expected but not the primary diagnostics.
