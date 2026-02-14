# Genopheno standard report

- life: `life.jsonl`

- pop: `pop.jsonl`

- world: `world.jsonl`


## Counts

| metric | value |
| --- | --- |
| agents_seen | 345 |
| births_life | 345 |
| deaths_life | 275 |
| steps_life | 0 |


## Lifespan

| stat | value |
| --- | --- |
| mean | 61.084 |
| p10 | 30.872 |
| p50 | 60.520 |
| p90 | 90.652 |


## Offspring

| stat | value |
| --- | --- |
| mean | 0.965 |
| p10 | 0.000 |
| p50 | 1.000 |
| p90 | 2.000 |
| share_zero | 0.394 |


## Maturity gating

| metric | value |
| --- | --- |
| matured_share | 0.948 |
| matured_n | 327 |

| offspring_after_mature stat | value |
| --- | --- |
| mean | 0.965 |
| p10 | 0.000 |
| p50 | 1.000 |
| p90 | 2.000 |
| share_zero | 0.394 |


## Phenotype → fitness (correlations)

| phenotype | corr_vs_R0m_or_R0 |
| --- | --- |
| E_repro_min | -0.197 |
| metabolism_scale | -0.187 |
| repro_rate | +0.154 |
| stress_per_drain | -0.153 |
| A_mature | -0.152 |
| sense_strength | +0.135 |
| risk_aversion | +0.134 |
| E_rep_min | +0.126 |
| sociability | +0.122 |
| cold_aversion | -0.122 |


## Phenotype selection differential (top group vs all)

| phenotype | Δ mean(top) - mean(all) |
| --- | --- |
| A_mature | -0.1950 |
| frailty_gain | +0.0271 |
| risk_aversion | +0.0197 |
| sociability | +0.0123 |
| mobility | -0.0074 |
| E_repro_min | -0.0070 |
| sense_strength | +0.0064 |
| metabolism_scale | -0.0047 |
| cold_aversion | -0.0030 |
| repro_rate | +0.0028 |


## Lineage dominance

| agent_id | R0 | age | birth_t |
| --- | --- | --- | --- |
| 13 | 3 | 101.64 | 47.32 |
| 16 | 3 | 101.80 | 83.20 |
| 21 | 3 | 94.84 | 154.12 |
| 23 | 3 | 97.44 | 167.98 |
| 24 | 3 | 91.56 | 177.48 |
| 28 | 3 | 94.46 | 191.86 |
| 31 | 3 | 100.56 | 207.70 |
| 33 | 3 | 99.08 | 218.22 |
| 37 | 3 | 80.34 | 230.64 |
| 40 | 3 | 83.04 | 243.90 |

| agent_id | R0_mature | age | birth_t |
| --- | --- | --- | --- |
| 13 | 3 | 101.64 | 47.32 |
| 16 | 3 | 101.80 | 83.20 |
| 21 | 3 | 94.84 | 154.12 |
| 23 | 3 | 97.44 | 167.98 |
| 24 | 3 | 91.56 | 177.48 |
| 28 | 3 | 94.46 | 191.86 |
| 31 | 3 | 100.56 | 207.70 |
| 33 | 3 | 99.08 | 218.22 |
| 37 | 3 | 80.34 | 230.64 |
| 40 | 3 | 83.04 | 243.90 |


## Birth cohorts (drift + cohort fitness)

- bin_width: 200.0


### t ∈ [0, 200)  (n_births=29)

| metric | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| age | 72.77 | 30.38 | 76.88 | 104.22 |
| R0 | 1.14 | 0.00 | 1.00 | 3.00 |
| R0_mature | 1.14 | 0.00 | 1.00 | 3.00 |

- matured_share: 0.966


### t ∈ [200, 400)  (n_births=106)

| metric | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| age | 64.31 | 34.30 | 64.75 | 90.37 |
| R0 | 1.24 | 0.00 | 1.00 | 3.00 |
| R0_mature | 1.24 | 0.00 | 1.00 | 3.00 |

- matured_share: 1.000


### t ∈ [400, 600)  (n_births=210)

| metric | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| age | 56.22 | 29.09 | 55.86 | 83.41 |
| R0 | 1.03 | 0.00 | 1.00 | 2.00 |
| R0_mature | 1.03 | 0.00 | 1.00 | 2.00 |

- matured_share: 0.993


## Sample drift (cross-sectional longitudinal)

| metric | value |
| --- | --- |
| n_rows | 381 |
| n_unique_agents | 151 |
| t_min | 0.00 |
| t_max | 598.02 |


### Top |slope| vs time

| phenotype | slope_per_time |
| --- | --- |
| A_mature | -0.00755366 |
| frailty_gain | +0.000575448 |
| risk_aversion | +0.000351718 |
| sociability | +0.000219273 |
| sense_strength | +0.000166319 |
| E_repro_min | -0.000161318 |
| mobility | -0.000110238 |
| metabolism_scale | -9.70987e-05 |
| susceptibility | -8.74748e-05 |
| repro_rate | +5.51929e-05 |
