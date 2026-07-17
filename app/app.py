"""
Predictive Maintenance Digital Twin Dashboard
================================================
Loads the model and data your notebook already produced.
Does NOT retrain, does NOT generate synthetic data, does NOT
regenerate a random digital-twin snapshot on every rerun.

Navigation is a sidebar radio, not st.tabs() — 10 tabs overflow the
top bar and get clipped, sidebar nav has no such limit.

Expects these files (produced by Digital_Twin_cleaned.ipynb):
    xgb_rul_model.pkl
    digital_twin_snapshot.csv
    train_processed.csv
    test_predictions.csv      (optional — see note at bottom of file)

Run with:  streamlit run app/app.py   (from the project root)
"""

import os
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import io
from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artifacts")
RUL_CLIP = 125

# Must match FEATURE_COLS in the notebook exactly.
FEATURE_COLS = [
    "cycle", "op1", "op2",
    "sensor2", "sensor3", "sensor4", "sensor6", "sensor7", "sensor8",
    "sensor9", "sensor11", "sensor12", "sensor13", "sensor14",
    "sensor15", "sensor17", "sensor20", "sensor21",
]

# Reported test-set benchmark numbers from the notebook (update these two
# lines whenever you re-run the notebook's official test-set scoring cell —
# do NOT let them silently drift out of sync with what the notebook prints).
REPORTED_TEST_MAE = 13.30
REPORTED_TEST_R2 = 0.80
REPORTED_CV_R2_MEAN = 0.84
REPORTED_CV_R2_STD = 0.02


# ------------------------------------------------------------------
# LOAD REAL ARTIFACTS — fail loudly if missing, don't silently fake data
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = []
    for fname in ["xgb_rul_model.pkl", "digital_twin_snapshot.csv", "train_processed.csv"]:
        if not os.path.exists(os.path.join(ARTIFACT_DIR, fname)):
            missing.append(fname)
    if missing:
        return None, None, None, None, missing

    model = joblib.load(os.path.join(ARTIFACT_DIR, "xgb_rul_model.pkl"))
    snapshot_df = pd.read_csv(os.path.join(ARTIFACT_DIR, "digital_twin_snapshot.csv"))
    train_df = pd.read_csv(os.path.join(ARTIFACT_DIR, "train_processed.csv"))

    test_pred_path = os.path.join(ARTIFACT_DIR, "test_predictions.csv")
    test_pred_df = pd.read_csv(test_pred_path) if os.path.exists(test_pred_path) else None

    return model, snapshot_df, train_df, test_pred_df, []


model, snapshot_df, train_df, test_pred_df, missing_files = load_artifacts()

if missing_files:
    st.error(
        "Missing required file(s): " + ", ".join(missing_files) + ". "
        "Run the export cell (section 12) in Digital_Twin_cleaned.ipynb and "
        "confirm they land in the artifacts/ folder, then reload."
    )
    st.stop()

# Digital-twin risk fields already computed in the notebook — don't recompute
# with different logic here, or the two will drift apart.
if "Health_Index" not in snapshot_df.columns:
    st.error(
        "digital_twin_snapshot.csv is missing Health_Index / Failure_Risk_% / "
        "Recommendation columns — re-export from the notebook (section 11)."
    )
    st.stop()

RISK_COLOR = {"Immediate Maintenance": "#e34948", "Schedule Maintenance": "#eda100", "Healthy": "#008300"}
snapshot_df["risk"] = snapshot_df["Recommendation"]

# ------------------------------------------------------------------
# SIDEBAR — navigation + global controls
# ------------------------------------------------------------------
PAGES = [
    " Home", "Data Engineering", "EDA", " Model Performance",
    " Explainable AI", "Digital Twin", "Fleet Management",
    " Business Intelligence", "What-If Simulator", "Export",
]

st.sidebar.title("🏭 Digital Twin Controls")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")

selected_engine = st.sidebar.selectbox(
    "Select engine", sorted(snapshot_df["engine_id"].unique().tolist()), index=0
)
sensor_options = [c for c in FEATURE_COLS if c.startswith("sensor")]
selected_sensor = st.sidebar.selectbox("Sensor focus", sensor_options, index=0)
st.sidebar.markdown("---")

st.sidebar.caption(f"Test-set MAE {REPORTED_TEST_MAE} · R² {REPORTED_TEST_R2}  |  "
                    f"CV R² {REPORTED_CV_R2_MEAN}±{REPORTED_CV_R2_STD}")

st.title(" Predictive Maintenance Digital Twin")

# ==================================================================
# PAGE — HOME
# ==================================================================
if page == PAGES[0]:
    st.subheader("Executive Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total engines", snapshot_df["engine_id"].nunique())
    c2.metric("Avg predicted RUL", f"{snapshot_df['Predicted_RUL'].mean():.0f} cycles")
    c3.metric("High-risk (Immediate Maintenance)",
              int((snapshot_df["risk"] == "Immediate Maintenance").sum()))
    c4.metric("Test-set MAE", f"{REPORTED_TEST_MAE:.2f}")
    c5.metric("Test-set R²", f"{REPORTED_TEST_R2:.2f}")

    st.markdown("### Fleet health overview")
    fig = px.histogram(
        snapshot_df, x="Health_Index", nbins=20, color="risk",
        color_discrete_map=RISK_COLOR,
        labels={"Health_Index": "Health index (%)"},
    )
    fig.update_layout(bargap=0.05, height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Project summary")
    st.info(
        "RUL predicted from NASA C-MAPSS FD001 sensor telemetry using XGBoost. "
        "RUL is clipped at 125 cycles (standard C-MAPSS treatment — early-life "
        "cycles carry no visible degradation signal). Validated with "
        f"GroupKFold CV (R² {REPORTED_CV_R2_MEAN}±{REPORTED_CV_R2_STD}, no engine "
        f"overlap between folds) and scored against the official FD001 test-set "
        f"ground truth (MAE {REPORTED_TEST_MAE}, R² {REPORTED_TEST_R2})."
    )

# ==================================================================
# PAGE — DATA ENGINEERING
# ==================================================================
elif page == PAGES[1]:
    st.subheader("Data Engineering Pipeline")

    st.markdown("**Processed training data preview**")
    st.dataframe(train_df.head(8), use_container_width=True)
    st.caption(f"Shape: {train_df.shape}")

    st.markdown("### Constant / zero-variance features removed")
    st.write("`op3`, `sensor1`, `sensor5`, `sensor10`, `sensor16`, `sensor18`, `sensor19` "
             "— zero variance across all rows in FD001, dropped during cleaning.")

    st.markdown("### Feature engineering pipeline (what the notebook actually does)")
    st.code(
        "1. Load train_FD001 / test_FD001 / RUL_FD001\n"
        "2. Compute RUL = max_cycle(engine) - cycle\n"
        "3. Clip RUL at 125 cycles (piecewise RUL — standard C-MAPSS treatment)\n"
        "4. Drop constant / zero-variance columns\n"
        "5. GroupShuffleSplit + GroupKFold by engine_id (no engine in both\n"
        "   train and test — verified with an overlap assertion)\n"
        "6. Train XGBoost on remaining sensor + operational-setting columns",
        language="text",
    )
    st.caption("No rolling-window statistics or per-sensor normalization are computed — "
               "the model trains on raw sensor values plus cycle. This matches the code "
               "exactly; earlier drafts of this panel claimed steps that weren't implemented.")

    st.markdown("### RUL formula")
    st.latex(r"RUL_{t} = \min\big(C_{max}(engine) - C_{t},\ 125\big)")

# ==================================================================
# PAGE — EDA
# ==================================================================
elif page == PAGES[2]:
    st.subheader("Exploratory Data Analysis")

    st.markdown("### Dataset statistics")
    st.dataframe(train_df[FEATURE_COLS + ["RUL"]].describe().T, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### RUL distribution (clipped)")
        fig = px.histogram(train_df, x="RUL", nbins=40)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Engine life distribution")
        life_dist = train_df.groupby("engine_id")["cycle"].max()
        fig = px.histogram(life_dist, nbins=25, labels={"value": "Total cycles to failure"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation heatmap")
    corr = train_df[FEATURE_COLS + ["RUL"]].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Sensor trend — engine {selected_engine}")
    eng_df = train_df[train_df["engine_id"] == selected_engine]
    fig = px.line(eng_df, x="cycle", y=selected_sensor)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE — MODEL PERFORMANCE
# ==================================================================
elif page == PAGES[3]:
    st.subheader("Model Performance")
    st.caption(
        "Headline numbers below are the notebook's official test-set evaluation "
        "against RUL_FD001.txt ground truth, not a number recomputed here."
    )
    c1, c2 = st.columns(2)
    c1.metric("Official test-set MAE", f"{REPORTED_TEST_MAE:.2f}")
    c2.metric("Official test-set R²", f"{REPORTED_TEST_R2:.2f}")
    st.caption(f"Internal GroupKFold CV: R² {REPORTED_CV_R2_MEAN} ± {REPORTED_CV_R2_STD} "
               "(5 folds, no engine overlap between train/test in any fold).")

    st.markdown("### Feature importance")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig = px.bar(importance, orientation="h")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if test_pred_df is not None:
        st.markdown("### Actual vs predicted RUL (official test set)")
        fig = px.scatter(test_pred_df, x="Actual_RUL", y="Predicted_RUL", opacity=0.5)
        lo, hi = 0, RUL_CLIP
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="red", dash="dash"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        residuals = test_pred_df["Actual_RUL"] - test_pred_df["Predicted_RUL"]
        fig = px.scatter(x=test_pred_df["Predicted_RUL"], y=residuals,
                          labels={"x": "Predicted RUL", "y": "Residual"}, opacity=0.5)
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Actual-vs-predicted and residual plots need `test_predictions.csv` "
            "(engine_id, Actual_RUL, Predicted_RUL) exported from the notebook's "
            "test-set scoring cell — see the note at the bottom of app.py."
        )

# ==================================================================
# PAGE — EXPLAINABLE AI (SHAP)
# ==================================================================
elif page == PAGES[4]:
    st.subheader("Explainable AI — SHAP")
    st.caption("SHAP explains *why* the model predicted a given RUL for a specific engine.")

    explainer = shap.TreeExplainer(model)
    engine_row = snapshot_df[snapshot_df["engine_id"] == selected_engine][FEATURE_COLS]

    if len(engine_row):
        shap_values_engine = explainer(engine_row)
        pred_val = snapshot_df.loc[snapshot_df["engine_id"] == selected_engine, "Predicted_RUL"].values[0]
        st.markdown(f"### Why predicted RUL = {pred_val:.0f} for engine {selected_engine}?")

        fig_wf, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values_engine[0], max_display=10, show=False)
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

    st.markdown("### Global feature impact — SHAP summary plot")
    sample = train_df[FEATURE_COLS].sample(min(300, len(train_df)), random_state=42)
    shap_values_all = explainer(sample)
    fig_summary, ax2 = plt.subplots(figsize=(8, 5))
    shap.plots.beeswarm(shap_values_all, max_display=10, show=False)
    st.pyplot(fig_summary, use_container_width=True)
    plt.close(fig_summary)

# ==================================================================
# PAGE — DIGITAL TWIN
# ==================================================================
elif page == PAGES[5]:
    st.subheader(f"Digital Twin — Engine {selected_engine}")
    twin = snapshot_df[snapshot_df["engine_id"] == selected_engine].iloc[0]

    g1, g2, g3 = st.columns(3)
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=twin["Predicted_RUL"],
            title={"text": "Predicted RUL (cycles)"},
            gauge={"axis": {"range": [0, RUL_CLIP]}, "bar": {"color": "#2a78d6"}}))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=twin["Health_Index"],
            title={"text": "Health index (%)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLOR[twin["risk"]]}}))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)
    with g3:
        st.markdown("#### Failure risk")
        st.markdown(f"<h2 style='color:{RISK_COLOR[twin['risk']]}'>{twin['risk']}</h2>",
                    unsafe_allow_html=True)
        st.markdown("#### Recommendation")
        if twin["risk"] == "Immediate Maintenance":
            st.error("Schedule immediate inspection.")
        elif twin["risk"] == "Schedule Maintenance":
            st.warning("Plan maintenance in the near term.")
        else:
            st.success("No action needed. Continue routine monitoring.")

    st.markdown("### Snapshot sensor panel (this engine's sampled in-service cycle)")
    st.dataframe(twin[FEATURE_COLS].to_frame().T, use_container_width=True)
    st.caption(f"Snapshot cycle {int(twin['cycle'])} — sampled from the 60-90% point of this "
               "engine's life, seeded for reproducibility (see notebook section 10).")

# ==================================================================
# PAGE — FLEET MANAGEMENT
# ==================================================================
elif page == PAGES[6]:
    st.subheader("Fleet Management")

    st.markdown("### Fleet risk heatmap")
    n_engines = snapshot_df["engine_id"].nunique()
    grid_size = int(np.ceil(np.sqrt(n_engines)))
    heat = snapshot_df.sort_values("engine_id").reset_index(drop=True)
    risk_map = {"Healthy": 0, "Schedule Maintenance": 1, "Immediate Maintenance": 2}
    risk_num = heat["risk"].map(risk_map).values.astype(float)
    pad = grid_size * grid_size - len(risk_num)
    risk_num = np.concatenate([risk_num, [np.nan] * pad])
    grid = risk_num.reshape(grid_size, grid_size)
    fig = px.imshow(grid, color_continuous_scale=[[0, "#008300"], [0.5, "#eda100"], [1, "#e34948"]],
                     labels={"color": "Risk"})
    fig.update_layout(height=450, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Healthy · Schedule Maintenance · Immediate Maintenance — one cell per engine")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Maintenance priority queue")
        queue = snapshot_df.sort_values("Predicted_RUL").head(10)[["engine_id", "Predicted_RUL", "risk"]]
        icon_map = {"Immediate Maintenance": "", "Schedule Maintenance": "", "Healthy": ""}
        for _, r in queue.iterrows():
            st.write(f"{icon_map[r['risk']]} Engine {r['engine_id']} — RUL {r['Predicted_RUL']:.0f} cycles")
    with c2:
        st.markdown("### Fleet statistics")
        st.metric("Immediate Maintenance", int((snapshot_df["risk"] == "Immediate Maintenance").sum()))
        st.metric("Schedule Maintenance", int((snapshot_df["risk"] == "Schedule Maintenance").sum()))
        st.metric("Healthy", int((snapshot_df["risk"] == "Healthy").sum()))

# ==================================================================
# PAGE — BUSINESS INTELLIGENCE
# ==================================================================
elif page == PAGES[7]:
    st.subheader("Business Intelligence")
    st.caption("Illustrative cost model — replace with your organization's real maintenance costs.")

    c1, c2, c3 = st.columns(3)
    failure_cost = c1.number_input("Unplanned failure cost ($)", value=50000, step=1000)
    sched_cost = c2.number_input("Scheduled maintenance cost ($)", value=6000, step=500)
    n_high_risk = int((snapshot_df["risk"] == "Immediate Maintenance").sum())
    c3.metric("Immediate-maintenance engines", n_high_risk)

    total_savings = (failure_cost - sched_cost) * n_high_risk
    m1, m2, m3 = st.columns(3)
    m1.metric("Failure cost / engine", f"${failure_cost:,.0f}")
    m2.metric("Scheduled maintenance / engine", f"${sched_cost:,.0f}")
    m3.metric("Estimated total savings", f"${total_savings:,.0f}")

    fig = go.Figure(data=[
        go.Bar(name="Unplanned failure", x=["Cost per engine"], y=[failure_cost], marker_color="#e34948"),
        go.Bar(name="Scheduled maintenance", x=["Cost per engine"], y=[sched_cost], marker_color="#008300"),
    ])
    fig.update_layout(barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE — WHAT-IF SIMULATOR
# ==================================================================
elif page == PAGES[8]:
    st.subheader(f"What-If Simulator — Engine {selected_engine}")
    st.caption("Move the sliders to simulate sensor changes and see the predicted RUL update instantly.")

    base_row = snapshot_df[snapshot_df["engine_id"] == selected_engine][FEATURE_COLS].iloc[0].copy()
    sim_row = base_row.copy()

    cols = st.columns(3)
    for i, sensor in enumerate(sensor_options):
        with cols[i % 3]:
            lo, hi = float(train_df[sensor].min()), float(train_df[sensor].max())
            sim_row[sensor] = st.slider(sensor, lo, hi, float(base_row[sensor]), key=f"sim_{sensor}")

    new_pred = model.predict(pd.DataFrame([sim_row]))[0]
    baseline_pred = model.predict(pd.DataFrame([base_row]))[0]
    delta = new_pred - baseline_pred

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline predicted RUL", f"{baseline_pred:.0f}")
    c2.metric("Simulated predicted RUL", f"{new_pred:.0f}", delta=f"{delta:+.0f}")
    new_health = np.clip(new_pred / RUL_CLIP * 100, 0, 100)
    c3.metric("Simulated health index", f"{new_health:.0f}%")

# ==================================================================
# PAGE — EXPORT & REPORTS
# ==================================================================
elif page == PAGES[9]:
    st.subheader("Export & Reports")

    st.markdown("### Download fleet summary (CSV)")
    csv_buf = snapshot_df[["engine_id", "Predicted_RUL", "Health_Index", "risk"]].to_csv(index=False).encode()
    st.download_button("Download CSV report", csv_buf, "fleet_summary.csv", "text/csv")

    st.markdown("### Engine summary")
    st.dataframe(
        snapshot_df[["engine_id", "Predicted_RUL", "Health_Index", "risk"]].sort_values("Predicted_RUL"),
        use_container_width=True,
    )

    st.markdown("### Download PDF report")
    if st.button("Generate PDF report"):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as pdf_canvas

        buf = io.BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 800, "Predictive Maintenance — Fleet Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, 780, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.drawString(50, 760,
                     f"Engines: {snapshot_df['engine_id'].nunique()}   "
                     f"Test MAE: {REPORTED_TEST_MAE}   Test R2: {REPORTED_TEST_R2}")
        y = 730
        c.drawString(50, y, "Top 10 highest-risk engines:")
        y -= 20
        for _, r in snapshot_df.sort_values("Predicted_RUL").head(10).iterrows():
            c.drawString(60, y, f"Engine {r['engine_id']} — RUL {r['Predicted_RUL']:.0f} — {r['risk']}")
            y -= 16
        c.save()
        buf.seek(0)
        st.download_button("Download PDF", buf, "maintenance_report.pdf", "application/pdf")

# ------------------------------------------------------------------
# NOTE ON test_predictions.csv (optional but recommended)
# ------------------------------------------------------------------
# Add this to the end of your notebook's official-test-set-scoring cell
# (the one that produces test_last with Actual_RUL / Predicted_RUL) so
# Model Performance tab can show real actual-vs-predicted / residual plots
# instead of the "info" placeholder:
#
#   test_last[["engine_id", "Actual_RUL", "Predicted_RUL"]].to_csv(
#       "test_predictions.csv", index=False
#   )