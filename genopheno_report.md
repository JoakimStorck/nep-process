# Genopheno standard report

- life: `life.jsonl`

- pop: `pop.jsonl`

- world: `world.jsonl`


## Counts

| metric | value |
| --- | --- |
| agents_seen | 12 |
| births_life | 12 |
| deaths_life | 10 |
| steps_life | 0 |


## Lifespan

| stat | value |
| --- | --- |
| mean | 238.648 |
| p10 | 78.326 |
| p50 | 132.080 |
| p90 | 400.340 |


## Offspring

| stat | value |
| --- | --- |
| mean | 0.000 |
| p10 | 0.000 |
| p50 | 0.000 |
| p90 | 0.000 |
| share_zero | 1.000 |


## Maturity gating

| metric | value |
| --- | --- |
| matured_share | 1.000 |
| matured_n | 12 |

| offspring_after_mature stat | value |
| --- | --- |
| mean | 0.000 |
| p10 | 0.000 |
| p50 | 0.000 |
| p90 | 0.000 |
| share_zero | 1.000 |


## Phenotype → fitness (correlations)

| phenotype | corr_vs_R0m_or_R0 |
| --- | --- |


## Phenotype selection differential (top group vs all)

| phenotype | Δ mean(top) - mean(all) |
| --- | --- |
| A_mature | +0.0000 |
| E_rep_min | +0.0000 |
| E_repro_min | +0.0000 |
| M_repro_min | +0.0000 |
| child_E_fast | +0.0000 |
| child_E_slow | +0.0000 |
| child_Fg | +0.0000 |
| child_M | +0.0000 |
| cold_aversion | +0.0000 |
| frailty_gain | +0.0000 |


## Lineage dominance

| agent_id | R0 | age | birth_t |
| --- | --- | --- | --- |

| agent_id | R0_mature | age | birth_t |
| --- | --- | --- | --- |


## Birth cohorts (drift + cohort fitness)

- bin_width: 500.0


### t ∈ [0, 500)  (n_births=12)

| metric | mean | p10 | p50 | p90 |
| --- | --- | --- | --- | --- |
| age | 238.65 | 78.33 | 132.08 | 400.34 |
| R0 | 0.00 | 0.00 | 0.00 | 0.00 |
| R0_mature | 0.00 | 0.00 | 0.00 | 0.00 |

- matured_share: 1.000


## Sample drift (cross-sectional longitudinal)

| metric | value |
| --- | --- |
| n_rows | 837 |
| n_unique_agents | 12 |
| t_min | 0.00 |
| t_max | 1510.02 |


### Top |slope| vs time

| phenotype | slope_per_time |
| --- | --- |
| A_mature | +0.00169513 |
| mobility | -0.00014894 |
| sociability | +0.000116663 |
| frailty_gain | -0.000114328 |
| cold_aversion | -9.75535e-05 |
| risk_aversion | -9.69655e-05 |
| sense_strength | -8.23692e-05 |
| child_E_fast | -6.34916e-05 |
| metabolism_scale | +2.99564e-05 |
| E_rep_min | -2.93555e-05 |
