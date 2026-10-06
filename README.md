# Ocean Wave Dispersion Demo

An interactive [Streamlit](https://streamlit.io) app for exploring linear (Airy) ocean waves in deep,
intermediate, and shallow water. It shows how wavelength, phase velocity, and group velocity change
with wave period and water depth.

## Physics

Waves obey the dispersion relation

    ω² = g k tanh(k h)

with ω = 2π/T and wavenumber k = 2π/L. The app solves for k (Newton iteration), then computes

- phase speed `c = ω/k`
- group speed `cg = n c`, where `n = ½ (1 + 2kh / sinh 2kh)`

| Regime       | h/L          | Phase speed          | Group speed |
|--------------|--------------|----------------------|-------------|
| Deep         | > 1/2        | `c = gT/2π` (dispersive) | `cg = c/2`  |
| Intermediate | 1/20 – 1/2   | depends on T and h   | `c/2 < cg < c` |
| Shallow      | < 1/20       | `c = √(gh)`          | `cg = c` (non-dispersive) |

## App contents

Sidebar sliders set the wave period (s) and water depth (m, log scale).

- **Summary** – regime classification, wavelength, wavenumber, speeds, and a comparison with the
  deep- and shallow-water approximations.
- **Dispersion** – phase and group speed versus h/L (with Cg/C below), and normalized c/√(gh) with the
  deep/shallow limits. A vertical line marks the current wave.
- **Wave group** – animated superposition of two close wavenumbers showing crests moving at the phase
  speed and the envelope moving at the group speed.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```

## Tests

```
pytest
```

The tests in `test_physics.py` check the solver against the dispersion relation, the deep/shallow
limits, `cg = dω/dk`, speed bounds, and regime classification.

## Layout

- `app.py` – Streamlit UI and plots
- `physics.py` – wave physics (exact, deep-water, and shallow-water solutions)
- `test_physics.py` – pytest suite
