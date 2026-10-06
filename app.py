import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from physics import (G, deep_speeds, deep_wavenumber, omega_of_k, regime, shallow_speeds,
                     speeds, wavenumber)

st.set_page_config(page_title="Ocean Wave Dispersion", layout="wide")

COLORS = {"Deep": "#1f77b4", "Intermediate": "#2ca02c", "Shallow": "#d62728"}

st.title("🌊 Ocean Wave Dispersion")
st.caption("Linear (Airy) waves: ω² = g k tanh(k h). Phase speed c = ω/k, group speed cg = dω/dk.")

with st.sidebar:
    st.header("Wave parameters")
    T = st.slider("Wave period T (s)", 2.0, 25.0, 10.0, 0.5)
    log_h = st.slider("Water depth h (log scale)", 0.0, float(np.log10(4000)), float(np.log10(50)), 0.01,
                      format="10^%.2f m")
    h = float(10**log_h)
    st.markdown(f"**h = {h:.1f} m**")
    st.markdown("---")
    st.markdown("**Regimes (h/L)**\n- Deep: > 1/2\n- Intermediate: 1/20 to 1/2\n- Shallow: < 1/20")

omega = 2 * np.pi / T
k = float(wavenumber(omega, h))
L = 2 * np.pi / k
c, cg = (float(v) for v in speeds(k, h))
name, rule = regime(h, L)
c_deep, cg_deep = (float(v) for v in deep_speeds(deep_wavenumber(omega)))
c_shal, cg_shal = (float(v) for v in shallow_speeds(h))

tab1, tab2, tab4 = st.tabs(["Summary", "Dispersion", "Wave group"])

with tab1:
    st.subheader(f"{name}-water wave ({rule}, h/L = {h / L:.3f})")
    cols = st.columns(5)
    cols[0].metric("Wavelength L", f"{L:.1f} m")
    cols[1].metric("Wavenumber k", f"{k:.4f} rad/m")
    cols[2].metric("Phase speed c", f"{c:.2f} m/s")
    cols[3].metric("Group speed cg", f"{cg:.2f} m/s")
    cols[4].metric("cg / c", f"{cg / c:.2f}")
    st.markdown(
        f"""
| | Exact | Deep approx. (gT/2π) | Shallow approx. (√gh) |
|---|---|---|---|
| Phase speed c (m/s) | **{c:.2f}** | {c_deep:.2f} | {c_shal:.2f} |
| Group speed cg (m/s) | **{cg:.2f}** | {cg_deep:.2f} | {cg_shal:.2f} |
"""
    )
    st.markdown(
        "- **Deep:** c = gT/2π depends on period (dispersive), cg = c/2.\n"
        "- **Shallow:** c = cg = √(gh), independent of period (non-dispersive).\n"
        "- **Intermediate:** both depend on T and h, with c/2 < cg < c."
    )

with tab2:
    st.caption(f"Fixed wavelength L = {L:.1f} m (from the sidebar period and depth); depth is varied.")
    if not 0.01 <= h / L <= 1:
        st.info(f"Current wave h/L = {h / L:.3g} is outside the plotted range (0.01–1).")
    hL = np.logspace(-2, 0, 300)
    hv = hL * L
    Cv, Cgv = speeds(k, hv)
    nv = Cgv / Cv
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4])
    fig.add_scatter(x=hL, y=Cv, name="Phase speed, C", row=1, col=1)
    fig.add_scatter(x=hL, y=Cgv, name="Group velocity, C<sub>g</sub>", row=1, col=1)
    fig.add_scatter(x=hL, y=nv, name="n = C<sub>g</sub>/C", showlegend=False, row=2, col=1)
    for r in (1, 2):
        fig.add_vline(x=h / L, line_color="green", line_width=2, row=r, col=1,
                      annotation_text=f"current wave (h/L = {h / L:.3g})" if r == 1 else None,
                      annotation_position="top right")
    XTICKS = dict(type="log", range=[-2, 0], showgrid=True,
                  tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                  ticktext=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
    fig.update_xaxes(**XTICKS, row=1, col=1)
    fig.update_xaxes(**XTICKS, title_text="Non-dimensional depth, h/L [-]", row=2, col=1)
    AXIS = dict(mirror=True, showline=True, linecolor="black")
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS, showgrid=True)
    fig.update_yaxes(title_text="Speed [m/s]", row=1, col=1)
    fig.update_yaxes(title_text="C<sub>g</sub>/C [-]", row=2, col=1)
    fig.update_layout(height=650, template="simple_white", legend=dict(x=0.025, y=0.95, bordercolor="black", borderwidth=1))
    st.plotly_chart(fig, width="stretch")

    x = np.logspace(-2, 0, 400)  # h/L
    k_unit = 2 * np.pi  # L = 1, so depth h = x
    c_ex, cg_ex = speeds(k_unit, x)
    c_sh = shallow_speeds(x)[0]
    A, B = c_ex / c_sh, cg_ex / c_sh
    deep_c, deep_cg = (v / c_sh for v in deep_speeds(k_unit))
    f2 = go.Figure()
    f2.add_scatter(x=x, y=A, name="A: c/√(gh)", line=dict(color="blue", width=3))
    f2.add_scatter(x=x, y=B, name="B: c<sub>g</sub>/√(gh)", line=dict(color="red", width=3))
    f2.add_scatter(x=x, y=deep_c, name="Deep c", line=dict(color="blue", dash="dash"))
    f2.add_scatter(x=x, y=deep_cg, name="Deep c<sub>g</sub>", line=dict(color="red", dash="dash"))
    f2.add_scatter(x=x, y=shallow_speeds(x)[0] / c_sh, name="C: shallow limit",
                   line=dict(color="black", dash="dash"))
    for xv in (0.05, 0.5):
        f2.add_vline(x=xv, line_color="gray", line_width=1)
    f2.add_vline(x=h / L, line_color="green", line_width=2,
                 annotation_text=f"current wave (h/L = {h / L:.3g})",
                 annotation_position="top right")
    for xt, lab in ((0.022, "shallow"), (0.16, "intermediate"), (0.72, "deep")):
        f2.add_annotation(x=np.log10(xt), y=0.07, text=lab, showarrow=False)
    f2.update_xaxes(**XTICKS, title_text="Non-dimensional depth, h/L [-]", **AXIS)
    f2.update_yaxes(range=[0, 1.2], title_text="c / √(gh)", showgrid=True, **AXIS)
    f2.update_layout(height=450, template="simple_white", legend=dict(x=0.025, y=0.05, bordercolor="black", borderwidth=1))
    st.plotly_chart(f2, width="stretch")

with tab4:
    st.markdown(
        "Two waves with slightly different wavenumbers superpose. Ripples move at the "
        "**phase speed** (red dot tracks a crest); the envelope moves at the **group speed** "
        "(black dot tracks the envelope peak). Press ▶ to animate."
    )
    dk_frac = st.slider("Wavenumber spread Δk / k", 0.05, 0.4, 0.15, 0.01)
    k1, k2 = k * (1 - dk_frac / 2), k * (1 + dk_frac / 2)
    w1, w2 = omega_of_k(k1, h), omega_of_k(k2, h)
    dk, dw = k2 - k1, w2 - w1
    x = np.linspace(0, 4 * 2 * np.pi / dk, 1200)
    tmax = (x[-1] / 2) / c
    ts = np.linspace(0, tmax, 60)

    def eta(t):
        return np.cos(k1 * x - w1 * t) + np.cos(k2 * x - w2 * t)

    def env(t):
        return 2 * np.abs(np.cos((dk * x - dw * t) / 2))

    x_start = x[-1] / 4
    def frame_data(t):
        xg = x_start + cg * t
        xp = x_start + c * t
        return [
            go.Scatter(x=x, y=eta(t)),
            go.Scatter(x=x, y=env(t)),
            go.Scatter(x=x, y=-env(t)),
            go.Scatter(x=[xp], y=[2], mode="markers"),
            go.Scatter(x=[xg], y=[2], mode="markers"),
        ]

    d0 = frame_data(0)
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=d0[0].y, name="η(x,t)", line=dict(color="#1f77b4")),
            go.Scatter(x=x, y=d0[1].y, name="Envelope", line=dict(color="gray", dash="dash")),
            go.Scatter(x=x, y=d0[2].y, showlegend=False, line=dict(color="gray", dash="dash")),
            go.Scatter(x=d0[3].x, y=[2.3], mode="markers", name=f"phase c = {c:.1f} m/s",
                       marker=dict(color="red", size=12, symbol="triangle-down")),
            go.Scatter(x=d0[4].x, y=[2.3], mode="markers", name=f"group cg = {cg:.1f} m/s",
                       marker=dict(color="black", size=12, symbol="triangle-down")),
        ],
        frames=[
            go.Frame(
                name=f"{t:.2f}",
                data=[
                    go.Scatter(y=eta(t)),
                    go.Scatter(y=env(t)),
                    go.Scatter(y=-env(t)),
                    go.Scatter(x=[x_start + c * t], y=[2.3]),
                    go.Scatter(x=[x_start + cg * t], y=[2.3]),
                ],
            )
            for t in ts
        ],
    )
    fig.update_layout(
        xaxis_title="x (m)", yaxis_title="η (arb. units)", yaxis_range=[-2.6, 2.6],
        xaxis_range=[0, x[-1]],
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.15,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ])],
        sliders=[dict(
            steps=[dict(label=f"{t:.0f}s", method="animate",
                        args=[[f"{t:.2f}"], dict(mode="immediate",
                                                 frame=dict(duration=0, redraw=True))])
                   for t in ts],
            currentvalue=dict(prefix="t = "))],
    )
    st.plotly_chart(fig, width="stretch")
    if abs(cg / c - 0.5) < 0.03:
        st.info("Deep water: the envelope advances at about half the crest speed — crests appear to travel through the packet.")
    elif abs(cg / c - 1) < 0.03:
        st.info("Shallow water: crests and envelope travel together — the packet keeps its shape (non-dispersive).")
