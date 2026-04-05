from __future__ import annotations

import numpy as np


def compute_plant_growth_window(WP, T: np.ndarray) -> np.ndarray:
    """
    Temperaturberoende tillväxtfönster G(T) i [0,1].
    Ren funktion: använder bara parametrar och indatafält.
    """
    Tmin, Topt, Tmax = float(WP.T_grow_min), float(WP.T_grow_opt), float(WP.T_grow_max)

    G = np.zeros_like(T, dtype=np.float32)

    if Topt > Tmin + 1e-9:
        G = np.where((T >= Tmin) & (T < Topt), (T - Tmin) / (Topt - Tmin), G)
    if Tmax > Topt + 1e-9:
        G = np.where((T >= Topt) & (T <= Tmax), (Tmax - T) / (Tmax - Topt), G)

    return np.clip(G, 0.0, 1.0).astype(np.float32, copy=False)


def compute_plant_wither_rate(WP, T: np.ndarray) -> np.ndarray:
    """
    Temperaturberoende wither/dieoff-rate m(T) >= 0.
    Ren funktion: använder bara parametrar och indatafält.
    """
    m = np.full_like(T, float(WP.B_wither_base), dtype=np.float32)

    if float(WP.B_wither_cold) > 0.0 and float(WP.cold_width) > 1e-9:
        Sc = np.clip((float(WP.T_cold) - T) / float(WP.cold_width), 0.0, 1.0).astype(np.float32, copy=False)
        m += float(WP.B_wither_cold) * Sc

    if float(WP.B_wither_hot) > 0.0 and float(WP.hot_width) > 1e-9:
        Sh = np.clip((T - float(WP.T_hot)) / float(WP.hot_width), 0.0, 1.0).astype(np.float32, copy=False)
        m += float(WP.B_wither_hot) * Sh

    return m


def step_plant_field(
    B: np.ndarray,
    *,
    WP,
    G: np.ndarray,
    m: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float, float]:
    """
    Ren uppdatering av växtfältet:
      - growth - wither + diffusion
      - returnerar (B_new, dM_growth, dM_wither)
    """
    lapB = (
        np.roll(B, 1, axis=0) + np.roll(B, -1, axis=0)
        + np.roll(B, 1, axis=1) + np.roll(B, -1, axis=1)
        - 4.0 * B
    )

    growth = (float(WP.B_regen) * G) * (1.0 - B / float(WP.B_K)) * B
    wither = m * B

    dB = growth - wither + float(WP.B_diff) * lapB
    B_new = np.clip(B + np.float32(dt) * dB, 0.0, float(WP.B_K)).astype(np.float32, copy=False)

    dM_growth = float(np.sum(np.float64(dt) * np.float64(growth)))
    dM_wither = float(np.sum(np.float64(dt) * np.float64(wither)))

    return B_new, dM_growth, dM_wither


def step_carcass_field(
    C: np.ndarray,
    *,
    WP,
    dt: float,
) -> tuple[np.ndarray, float]:
    """
    Ren uppdatering av kadaverfältet:
      - decay + diffusion
      - returnerar (C_new, dM_decay)
    """
    lapC = (
        np.roll(C, 1, axis=0) + np.roll(C, -1, axis=0)
        + np.roll(C, 1, axis=1) + np.roll(C, -1, axis=1)
        - 4.0 * C
    )

    decay_rate = float(WP.C_decay) * C
    dC = (-decay_rate) + float(WP.C_diff) * lapC

    C_new = (C + np.float32(dt) * np.float32(dC)).astype(np.float32, copy=False)
    C_new = np.maximum(C_new, 0.0).astype(np.float32, copy=False)

    dM_decay = float(np.sum(np.float64(dt) * np.float64(np.maximum(decay_rate, 0.0))))
    return C_new, dM_decay