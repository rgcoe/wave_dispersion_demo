# Copyright (c) 2026 Ryan G. Coe
# Licensed under the GNU General Public License v3.0 (see LICENSE).

import numpy as np

G = 9.81


def wavenumber(omega, h):
    """Solve omega^2 = g k tanh(k h) for k (Newton, deep-water start)."""
    omega = np.asarray(omega, dtype=float)
    h = np.asarray(h, dtype=float)
    k = omega**2 / G
    k = np.where(k * h < 1, omega / np.sqrt(G * h), k)
    for _ in range(60):
        t = np.tanh(k * h)
        f = G * k * t - omega**2
        df = G * t + G * k * h * (1 - t**2)
        k = np.maximum(k - f / df, 1e-12)
    return k


def omega_of_k(k, h):
    return np.sqrt(G * k * np.tanh(k * h))


def speeds(k, h):
    """Return phase speed c and group speed cg."""
    kh = k * h
    c = np.sqrt(G / k * np.tanh(kh))
    x = 2 * kh
    # x / sinh(x) written without exp overflow for large x
    n = 0.5 * (1 + 2 * x * np.exp(-x) / -np.expm1(-2 * x))
    return c, n * c


def deep_wavenumber(omega):
    """Deep-water limit (kh >> 1): omega^2 = g k."""
    return np.asarray(omega, dtype=float) ** 2 / G


def shallow_wavenumber(omega, h):
    """Shallow-water limit (kh << 1): omega = k sqrt(g h)."""
    return np.asarray(omega, dtype=float) / np.sqrt(G * np.asarray(h, dtype=float))


def deep_speeds(k):
    """Deep-water phase and group speed: c = sqrt(g/k), cg = c/2."""
    c = np.sqrt(G / np.asarray(k, dtype=float))
    return c, c / 2


def shallow_speeds(h):
    """Shallow-water phase and group speed: c = cg = sqrt(g h)."""
    c = np.sqrt(G * np.asarray(h, dtype=float))
    return c, c


def regime(h, L):
    r = h / L
    if r > 0.5:
        return "Deep", "h/L > 1/2"
    if r < 0.05:
        return "Shallow", "h/L < 1/20"
    return "Intermediate", "1/20 ≤ h/L ≤ 1/2"
