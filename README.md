<!-- README.md -->
# NEP — Neuro-Evolutionary Process agent (minimal)

This repo implements a minimal agent-in-a-field simulation where adaptation is:
- **within an episode:** only through **internal neural dynamics** (state, traces)
- **across episodes:** through **inheritance + mutation + selection** (neuroevolution)

No patching, no rule-based policy layer, no time-based “undo”.

## Design manifesto (non-negotiable)

1) **Single time flow**
   - The world and agent evolve in the same discrete time step `dt`.
   - No scheduled callbacks, no event queues, no “undo”.

2) **Local causality**
   - The agent’s inputs are only local measurements + internal state.
   - The agent does not observe global time `t` and does not receive external reward/control signals.

3) **Negative control is allowed**
   - The network may output inhibition signals (e.g., `move_inh`, `eat_inh`).
   - These are not “goals”; they are internal gating dynamics.

4) **No mid-episode parameter patching**
   - No patch surfaces / whitelists / TTL patches.
   - Environment parameters are fixed per episode.

5) **Evolution is the only structural adaptation**
   - Network weights and architecture may mutate between episodes.
   - Inherited traits define the next generation.

6) **Fitness is offline only**
   - Fitness is computed by the experiment runner for selection.
   - Fitness is never fed back into the agent as an observation or control input.

## Code layout

- `world.py` — 2D toroidal fields: nutrient `N` and hazard `F` (diffusion + stochastic forcing)
- `agent.py` — body dynamics + sensors + NN-driven control (including inhibition outputs)
- `mlp.py` — mutable MLP genome (weights + architecture)
- `run.py` — run a single episode and compute fitness (offline measurement)
- `evolve.py` — population evolution (selection + mutation)
- `observer.py` — JSONL logging (read-only; can be disabled by passing `log_fp=None`)

## Running

Single episode:
```bash
python run.py --T 1200 --seed 3 --size 64 --log steps.jsonl