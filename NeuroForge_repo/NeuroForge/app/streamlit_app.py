"""CarbonFL dashboard — NeuroForge · AIORI PS-003.

    streamlit run app/streamlit_app.py

Tabs
1. Results       - 3-seed tables and curves from results/summary.json (no recomputation)
2. Cost planner  - analytic bytes: FedAvg vs raw streaming for any K, model size, codec, horizon
3. Byte budget   - pick the best logged checkpoint that fits a byte budget
4. Live FedAvg   - run a short federated training in the browser session and watch bytes grow
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="Federated Traffic Forecasting · NeuroForge", page_icon="🚦", layout="wide")

COL = {"central": "#1f2937", "fedavg_iid_E1_fp32": "#2563eb", "fedavg_noniid_E1_fp32": "#dc2626",
       "fedavg_noniid_E1_int8": "#059669", "fedavg_noniid_E1_fp16": "#0d9488", "fedavg_iid_E5_fp32": "#7c3aed",
       "fedavg_noniid_E5_fp32": "#db2777", "fedavg_noniid_E1_fp32_drop25": "#d97706", "fedavg_speed_E1_fp32": "#7f1d1d",
       "local_iid_E1_fp32": "#93c5fd", "local_noniid_E1_fp32": "#fca5a5"}


@st.cache_data
def load_summary():
    p = os.path.join(ROOT, "results", "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_curves():
    import glob
    fs = glob.glob(os.path.join(ROOT, "results", "runs", "*.csv"))
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["group"] = df["run_id"].str.rsplit("_s", n=1).str[0]
    return df


S = load_summary()
C = load_curves()

st.title("🚦 Federated Traffic-Flow Prediction with Communication-Cost Evaluation")
st.caption("Team NeuroForge · AIORI Hackathon 3 · PS-003 — METR-LA (207 sensors) · GRU(32) · FedAvg K = 4 · seeds 11/22/33")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📡 Cost planner", "💰 Byte budget", "▶️ Live FedAvg"])

with tab1:
    if S is None:
        st.warning("Run `python scripts/run_experiments.py --suite all` and `python scripts/report.py` first.")
    else:
        T = {r["group"]: r for r in S["table"]}
        k1, k2, k3, k4 = st.columns(4)
        if "central" in T:
            k1.metric("Centralised test MAE", f"{T['central']['mae'][0]:.3f} mph", help="mean over 3 seeds")
        if "fedavg_iid_E1_fp32" in T:
            g = T["fedavg_iid_E1_fp32"]["mae"][0] - T["central"]["mae"][0]
            k2.metric("FedAvg IID test MAE", f"{T['fedavg_iid_E1_fp32']['mae'][0]:.3f} mph", f"{g:+.3f} vs central", delta_color="inverse")
        if "fedavg_noniid_E1_fp32" in T:
            g = T["fedavg_noniid_E1_fp32"]["mae"][0] - T["central"]["mae"][0]
            k3.metric("FedAvg non-IID test MAE", f"{T['fedavg_noniid_E1_fp32']['mae'][0]:.3f} mph", f"{g:+.3f} vs central", delta_color="inverse")
        k4.metric("Bytes per FedAvg round", f"{S['analytic_round_bytes']:,} B", "logged = analytic ✓")

        rows = [{"Method": r["label"], "Test MAE": f"{r['mae'][0]:.3f} ± {r['mae'][1]:.3f}",
                 "Test RMSE": f"{r['rmse'][0]:.3f} ± {r['rmse'][1]:.3f}", "MAE +15 min": f"{r['mae_15'][0]:.3f}",
                 "Total bytes (MB)": round(r["B_total"][0] / 1e6, 3), "Per-seed MAE": r["mae_per_seed"]} for r in S["table"]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        pick = st.multiselect("Compare runs", list(S["curves"].keys()),
                              default=[g for g in ("central", "fedavg_iid_E1_fp32", "fedavg_noniid_E1_fp32", "fedavg_noniid_E1_int8") if g in S["curves"]],
                              format_func=lambda g: T[g]["label"] if g in T else g)
        a, b = st.columns(2)
        fig = go.Figure()
        for g in pick:
            c = S["curves"][g]
            m, s = np.array(c["val_mae"]), np.array(c["val_std"])
            fig.add_trace(go.Scatter(x=c["round"], y=m, name=T[g]["label"], line=dict(color=COL.get(g))))
            fig.add_trace(go.Scatter(x=c["round"] + c["round"][::-1], y=list(m + s) + list((m - s)[::-1]), fill="toself",
                                     line=dict(width=0), fillcolor=COL.get(g, "#888"), opacity=0.15, showlegend=False, hoverinfo="skip"))
        fig.update_layout(title="Validation MAE per round (mean ± std)", xaxis_title="round / epoch", yaxis_title="MAE (mph)", height=420)
        a.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        for g in pick:
            c = S["curves"][g]
            if g == "central":
                fig.add_trace(go.Scatter(x=[S["raw_upload_bytes"] / 1e6], y=[min(c["val_mae"])], mode="markers", marker=dict(size=14, symbol="star", color=COL[g]), name="Centralised (raw upload)"))
            else:
                fig.add_trace(go.Scatter(x=np.array(c["B_cum"]) / 1e6, y=c["val_mae"], name=T[g]["label"], line=dict(color=COL.get(g))))
        fig.update_layout(title="Accuracy per byte", xaxis_type="log", xaxis_title="cumulative MB (log)", yaxis_title="validation MAE (mph)", height=420)
        b.plotly_chart(fig, width="stretch")
        img = os.path.join(ROOT, "results", "figures", "partitions.png")
        if os.path.exists(img):
            st.image(img, caption="How the 207 sensors are split into 4 simulated intersections")

with tab2:
    st.subheader("FedAvg vs raw streaming — analytic payload calculator")
    c1, c2, c3 = st.columns(3)
    K = c1.slider("Clients (intersections) K", 2, 50, 4)
    sensors = c1.slider("Sensors in the city", 10, 5000, 207)
    hidden = c2.select_slider("GRU hidden size", [8, 16, 32, 64, 128], 32)
    codec = c2.selectbox("Upload codec", ["fp32", "fp16", "int8"])
    rounds = c3.slider("Rounds per retraining", 1, 200, 50)
    retrain_days = c3.slider("Retrain every N days", 1, 30, 7)
    params = 3 * (4 * hidden + hidden * hidden + 2 * hidden) + hidden * 3 + 3
    bpv = {"fp32": 4, "fp16": 2, "int8": 1}[codec]
    per_round = K * params * 4 + K * (params * bpv + 8 + (8 if codec == "int8" else 0))
    fl_day = per_round * rounds / retrain_days
    raw_day = sensors * 288 * 4
    m1, m2, m3 = st.columns(3)
    m1.metric("Model parameters", f"{params:,}")
    m2.metric("FedAvg bytes / day (amortised)", f"{fl_day/1e3:,.1f} kB")
    m3.metric("Raw streaming bytes / day", f"{raw_day/1e3:,.1f} kB", f"{(fl_day-raw_day)/1e3:+,.1f} kB FedAvg − raw", delta_color="inverse")
    days = np.arange(1, 91)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=raw_day * days / 1e6, name="Raw streaming (centralised)", line=dict(color="#1f2937")))
    fl = np.array([per_round * rounds * (1 + (d - 1) // retrain_days) for d in days])
    fig.add_trace(go.Scatter(x=days, y=fl / 1e6, name=f"FedAvg ({codec}, retrain every {retrain_days} d)", line=dict(color="#059669", shape="hv")))
    fig.update_layout(xaxis_title="days of operation", yaxis_title="cumulative MB", height=400)
    st.plotly_chart(fig, width="stretch")
    st.info("FedAvg cost depends on model size × clients × rounds, **not** on how much data is collected; raw streaming grows with sensors × time. "
            "Small models and infrequent retraining favour FedAvg; very large models or very few sensors favour streaming.")

with tab3:
    st.subheader("Which checkpoint should I deploy for a given byte budget?")
    if C is None:
        st.warning("No run logs found.")
    else:
        groups = sorted(g for g in C.group.unique() if g.startswith("fedavg"))
        g = st.selectbox("Method", groups, index=groups.index("fedavg_noniid_E1_fp32") if "fedavg_noniid_E1_fp32" in groups else 0)
        seed = st.selectbox("Seed", sorted(C.seed.unique()))
        bmax = st.slider("Byte budget (MB)", 0.1, 8.0, 1.0, 0.1)
        d = C[(C.group == g) & (C.seed == seed)].sort_values("round")
        ok = d[d.B_cum <= bmax * 1e6]
        if ok.empty:
            st.error("Budget too small for even one round.")
        else:
            best = ok.loc[ok.val_mae.idxmin()]
            st.success(f"Deploy round **{int(best['round'])}** — validation MAE **{best.val_mae:.3f} mph**, uses {best.B_cum/1e6:.2f} MB of {bmax:.1f} MB.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d.B_cum / 1e6, y=d.val_mae, name="validation MAE", line=dict(color="#2563eb")))
            fig.add_vline(x=bmax, line_dash="dash", annotation_text="budget")
            fig.add_trace(go.Scatter(x=[best.B_cum / 1e6], y=[best.val_mae], mode="markers", marker=dict(size=14, color="#059669"), name="selected"))
            fig.update_layout(xaxis_title="cumulative MB", yaxis_title="MAE (mph)", height=380)
            st.plotly_chart(fig, width="stretch")

with tab4:
    st.subheader("Run FedAvg live (small, CPU-friendly)")
    st.caption("Downloads METR-LA on first use (~57 MB). Uses a coarse window stride so a round takes a few seconds.")
    c1, c2, c3, c4 = st.columns(4)
    part = c1.selectbox("Partition", ["iid", "noniid", "speed"])
    rounds = c2.slider("Rounds", 1, 10, 5)
    codec = c3.selectbox("Codec", ["fp32", "fp16", "int8"], key="livecodec")
    drop = c4.slider("Client dropout", 0.0, 0.5, 0.0, 0.05)
    if st.button("Start federated training", type="primary"):
        import torch
        from fedtraffic.data import download, prepare
        from fedtraffic.simulate import RunConfig, run_federated
        torch.set_num_threads(2)
        h5, loc = download(os.path.join(ROOT, "data"))
        prep = prepare(h5, loc)
        cfg = RunConfig(partition=part, rounds=rounds, codec=codec, dropout=drop, train_stride=48, val_stride=24, batch=1024, lr=1e-2)
        box = st.empty(); prog = st.progress(0.0); chart = st.empty()
        hist = []

        def log(msg):
            hist.append(msg)
            box.code("\n".join(hist[-8:]))
            prog.progress(min(len(hist) / rounds, 1.0))

        t0 = time.time()
        rows, summ = run_federated(prep, cfg, log=log)
        df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["round"], y=df.B_up + df.B_down, name="bytes this round", yaxis="y2", marker_color="#cbd5e1"))
        fig.add_trace(go.Scatter(x=df["round"], y=df.val_mae, name="validation MAE", line=dict(color="#2563eb", width=3)))
        fig.update_layout(yaxis=dict(title="MAE (mph)"), yaxis2=dict(title="bytes", overlaying="y", side="right"), height=380)
        chart.plotly_chart(fig, width="stretch")
        st.success(f"Test MAE {summ['test']['mae']:.3f} mph · RMSE {summ['test']['rmse']:.3f} · {summ['B_total']:,} bytes · {time.time()-t0:.0f}s")
        st.json({"client_stats": summ["client_stats"], "per_client_test": summ["per_client_test"]}, expanded=False)
