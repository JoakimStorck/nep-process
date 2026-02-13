# Genopheno standard report

- life: `life.jsonl`

- pop: `pop.jsonl`

- world: `world.jsonl`


## Counts

| metric | value |
| --- | --- |
| agents_seen | 397 |
| births_life | 397 |
| deaths_life | 360 |
| steps_life | 0 |


## Lifespan

| stat | value |
| --- | --- |
| mean | 62.914 |
| p10 | 35.046 |
| p50 | 61.270 |
| p90 | 92.900 |


## Offspring

| stat | value |
| --- | --- |
| mean | 0.970 |
| p10 | 0.000 |
| p50 | 1.000 |
| p90 | 3.000 |
| share_zero | 0.446 |


## Maturity gating

| metric | value |
| --- | --- |
| matured_share | 0.952 |
| matured_n | 378 |

| offspring_after_mature stat | value |
| --- | --- |
| mean | 0.970 |
| p10 | 0.000 |
| p50 | 1.000 |
| p90 | 3.000 |
| share_zero | 0.446 |


## Phenotype → fitness (correlations)

| phenotype | corr_vs_R0m_or_R0 |
| --- | --- |
| E_repro_min | -0.141 |
| mobility | -0.111 |
| stress_per_drain | -0.076 |
| metabolism_scale | -0.057 |
| E_rep_min | +0.045 |
| repair_capacity | +0.041 |
| sociability | +0.040 |
| risk_aversion | -0.037 |
| frailty_gain | +0.036 |
| repro_rate | -0.035 |


## Phenotype selection differential (top group vs all)

| phenotype | Δ mean(top) - mean(all) |
| --- | --- |
| frailty_gain | +0.0063 |
| risk_aversion | -0.0062 |
| A_mature | +0.0055 |
| E_repro_min | -0.0053 |
| cold_aversion | +0.0033 |
| mobility | -0.0028 |
| susceptibility | -0.0025 |
| sociability | +0.0020 |
| E_rep_min | +0.0018 |
| repro_rate | -0.0016 |


## Lineage dominance

| agent_id | R0 | age | birth_t |
| --- | --- | --- | --- |
| 17 | 5 | 136.44 | 66.48 |
| 20 | 4 | 99.66 | 84.62 |
| 202 | 4 | 105.86 | 360.06 |
| 209 | 4 | 113.38 | 368.28 |
| 5 | 3 | 105.28 | 0.00 |
| 13 | 3 | 69.30 | 34.70 |
| 21 | 3 | 71.18 | 87.36 |
| 25 | 3 | 77.32 | 100.02 |
| 27 | 3 | 78.08 | 119.56 |
| 28 | 3 | 76.34 | 121.28 |

| agent_id | R0_mature | age | birth_t |
| --- | --- | --- | --- |
| 17 | 5 | 136.44 | 66.48 |
| 20 | 4 | 99.66 | 84.62 |
| 202 | 4 | 105.86 | 360.06 |
| 209 | 4 | 113.38 | 368.28 |
| 5 | 3 | 105.28 | 0.00 |
| 13 | 3 | 69.30 | 34.70 |
| 21 | 3 | 71.18 | 87.36 |
| 25 | 3 | 77.32 | 100.02 |
| 27 | 3 | 78.08 | 119.56 |
| 28 | 3 | 76.34 | 121.28 |


## Birth cohorts (drift + cohort fitness)

- bin_width: 250.0


### t ∈ [0, 250)  (n_births=123)

| metric | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| age | 69.24 | 39.07 | 69.70 | 99.31 |
| R0 | 1.19 | 0.00 | 1.00 | 3.00 |
| R0_mature | 1.19 | 0.00 | 1.00 | 3.00 |

- matured_share: 0.975


### t ∈ [250, 500)  (n_births=179)

| metric | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| age | 62.57 | 36.82 | 61.94 | 92.95 |
| R0 | 1.12 | 0.00 | 1.00 | 3.00 |
| R0_mature | 1.12 | 0.00 | 1.00 | 3.00 |

- matured_share: 0.961


### t ∈ [500, 750)  (n_births=95)

| metric | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| age | 51.08 | 34.21 | 48.73 | 70.14 |
| R0 | 0.45 | 0.00 | 0.00 | 1.00 |
| R0_mature | 0.45 | 0.00 | 0.00 | 1.00 |

- matured_share: 1.000
