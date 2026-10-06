import numpy as np
import pytest

from physics import (G, deep_speeds, deep_wavenumber, omega_of_k, regime, shallow_speeds,
                     shallow_wavenumber, speeds, wavenumber)


@pytest.mark.parametrize("T", [2.0, 6.0, 10.0, 25.0])
@pytest.mark.parametrize("h", [0.5, 1.0, 10.0, 50.0, 500.0, 4000.0])
def test_wavenumber_satisfies_dispersion_relation(T, h):
    w = 2 * np.pi / T
    k = wavenumber(w, h)
    assert G * k * np.tanh(k * h) == pytest.approx(w**2, rel=1e-9)
    assert omega_of_k(k, h) == pytest.approx(w, rel=1e-9)


def test_wavenumber_vectorized_over_depth():
    h = np.logspace(0, np.log10(4000), 50)
    k = wavenumber(2 * np.pi / 10, h)
    assert k.shape == h.shape
    assert np.all(np.diff(k) <= 1e-12)  # k decreases as depth increases


def test_deep_water_limits():
    T = 8.0
    h = 4000.0
    k = wavenumber(2 * np.pi / T, h)
    c, cg = speeds(k, h)
    assert k == pytest.approx((2 * np.pi / T) ** 2 / G, rel=1e-9)
    assert c == pytest.approx(G * T / (2 * np.pi), rel=1e-6)
    assert cg == pytest.approx(c / 2, rel=1e-6)


def test_shallow_water_limits():
    T, h = 20.0, 0.1
    k = wavenumber(2 * np.pi / T, h)
    c, cg = speeds(k, h)
    assert c == pytest.approx(np.sqrt(G * h), rel=1e-3)
    assert cg == pytest.approx(c, rel=1e-3)


@pytest.mark.parametrize("kh", [0.05, 0.3, 1.0, 3.0, 8.0])
def test_group_speed_is_derivative_of_omega(kh):
    h = 20.0
    k = kh / h
    dk = k * 1e-6
    numeric = (omega_of_k(k + dk, h) - omega_of_k(k - dk, h)) / (2 * dk)
    _, cg = speeds(k, h)
    assert cg == pytest.approx(numeric, rel=1e-6)


def test_group_speed_between_half_phase_and_phase():
    kh = np.logspace(-2, 1, 100)
    k = kh / 10.0
    c, cg = speeds(k, 10.0)
    assert np.all(cg <= c * (1 + 1e-12))
    assert np.all(cg >= 0.5 * c * (1 - 1e-12))


def test_phase_speed_never_exceeds_shallow_limit():
    h = 15.0
    k = np.logspace(-3, 0, 100)
    c, _ = speeds(k, h)
    assert np.all(c <= np.sqrt(G * h) * (1 + 1e-12))


def test_known_value_10s_50m():
    # Reference from the standard tanh dispersion relation: L ~ 151.3 m
    k = wavenumber(2 * np.pi / 10, 50.0)
    assert 2 * np.pi / k == pytest.approx(151.3, abs=0.1)


@pytest.mark.parametrize(
    "h,L,expected",
    [(100, 100, "Deep"), (5, 100, "Intermediate"), (1, 100, "Shallow"),
     (50.1, 100, "Deep"), (5.0, 100, "Intermediate"), (4.9, 100, "Shallow")],
)
def test_regime_classification(h, L, expected):
    assert regime(h, L)[0] == expected


def test_deep_solution_matches_exact_in_deep_water():
    w, h = 2 * np.pi / 8, 4000.0
    k = deep_wavenumber(w)
    assert k == pytest.approx(wavenumber(w, h), rel=1e-9)
    c, cg = deep_speeds(k)
    assert c == pytest.approx(G * 8 / (2 * np.pi), rel=1e-9)
    assert cg == pytest.approx(c / 2)
    assert (c, cg) == pytest.approx(tuple(speeds(k, h)), rel=1e-6)


def test_shallow_solution_matches_exact_in_shallow_water():
    w, h = 2 * np.pi / 20, 0.1
    k = shallow_wavenumber(w, h)
    assert k == pytest.approx(wavenumber(w, h), rel=1e-3)
    c, cg = shallow_speeds(h)
    assert c == cg == pytest.approx(np.sqrt(G * h))
    assert (c, cg) == pytest.approx(tuple(speeds(k, h)), rel=1e-3)
