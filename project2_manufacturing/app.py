"""
Manufacturing Failure Risk Monitor -- FMN AI Engineer Internship, Project 2.

Run locally:
    cd project2_manufacturing
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

import streamlit as st
import pandas as pd

from data_prep import load_and_clean
from features import build_features
from model import train_and_evaluate, build_machine_snapshot, score_history
import llm_client

st.set_page_config(page_title="Manufacturing Failure Risk Monitor", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "project2_manufacturing_sensors.csv")

FLAG_COLORS = {"HIGH_RISK": "🔴", "WATCH": "🟠", "OK": "🟢"}


@st.cache_data
def get_data():
    return load_and_clean(DATA_PATH)


@st.cache_data
def get_features(_df):
    return build_features(_df)


@st.cache_resource
def get_model(_feats):
    return train_and_evaluate(_feats)


def main():
    st.title("⚙️ Manufacturing Failure Risk Monitor")
    st.caption(
        "\"Machines go down without warning and it costs us hours of production. "
        "I want to know ahead of time which machines are at risk, and why.\" — this tool flags risk before it happens."
    )

    df = get_data()
    feats = get_features(df)
    result = get_model(feats)
    model = result["model"]
    snapshot = build_machine_snapshot(df, feats, model)

    counts = snapshot["flag"].value_counts()
    cols = st.columns(3)
    for col, flag in zip(cols, ["HIGH_RISK", "WATCH", "OK"]):
        col.metric(f"{FLAG_COLORS[flag]} {flag}", int(counts.get(flag, 0)))

    st.divider()

    st.dataframe(
        snapshot[["machine_id", "line", "flag", "failure_probability_24h",
                  "temperature_c", "vibration_mm_s", "vib_z_vs_own_baseline",
                  "run_hours_since_maintenance", "low_history"]],
        use_container_width=True, hide_index=True,
        column_config={"low_history": st.column_config.CheckboxColumn("Limited history")},
    )

    st.divider()

    st.subheader("Drill into a machine")
    machine_ids = snapshot["machine_id"].tolist()
    selected = st.selectbox("Choose a machine", options=machine_ids)

    if selected:
        row = snapshot[snapshot.machine_id == selected].iloc[0]
        c1, c2 = st.columns([2, 1])

        with c1:
            if not row["low_history"]:
                m_feats = feats[feats.machine_id == selected].sort_values("timestamp").tail(24 * 14)
                hist = score_history(m_feats, model)
                st.caption("Predicted failure risk over the last 14 days (spikes mark real failure events)")
                chart_df = hist.set_index("timestamp")[["failure_probability_24h"]]
                st.line_chart(chart_df)
                failures = hist[hist.actual_failure == 1]
                if not failures.empty:
                    st.caption(f"⚠️ Actual failure(s) in this window: {', '.join(str(t) for t in failures.timestamp)}")
            else:
                st.info(row["basis"])
                recent = df[df.machine_id == selected].sort_values("timestamp")
                st.line_chart(recent.set_index("timestamp")[["temperature_c", "vibration_mm_s"]])

        with c2:
            st.markdown(f"**{FLAG_COLORS[row.flag]} {row.flag}**")
            if row["failure_probability_24h"] is not None:
                st.write(f"Failure probability (next 24h): **{row.failure_probability_24h:.1%}**")
            st.write(f"Temperature: **{row.temperature_c} °C**")
            st.write(f"Vibration: **{row.vibration_mm_s} mm/s**")
            st.write(f"Vibration vs own baseline: **{row.vib_z_vs_own_baseline} std**")
            if row["low_history"]:
                st.warning(row["basis"])

        if st.button("🤖 Explain this flag", key=f"explain_{selected}"):
            with st.spinner("Generating explanation..."):
                try:
                    explanation = llm_client.generate_explanation(
                        entity_id=selected, entity_data=row.to_dict(), domain="manufacturing",
                    )
                    st.info(explanation)
                except Exception as e:
                    st.error(f"Couldn't generate explanation: {e}")

    st.divider()

    st.subheader("💬 Ask about machine risk")
    st.caption('e.g. "Which machines need attention this week?" or "Why is MCH-206 flagged historically?"')
    question = st.text_input("Your question", key="qa_input")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            try:
                answer = llm_client.answer_question(
                    question=question, full_table=snapshot.to_dict(orient="records"), domain="manufacturing",
                )
                st.success(answer)
            except Exception as e:
                st.error(f"Couldn't answer: {e}")

    with st.expander("📊 Model validation (time-based holdout)"):
        st.caption("Trained on data before Apr 1 2026, evaluated on April -- a true future holdout, not a random split.")
        m = result["metrics"]
        mcol, bcol = st.columns(2)
        with mcol:
            st.markdown("**Our model**")
            st.write(f"PR-AUC: {m['model_pr_auc']}")
            st.write(f"Precision: {m['model_precision']}")
            st.write(f"Recall: {m['model_recall']}")
            st.write(f"F1: {m['model_f1']}")
        with bcol:
            st.markdown("**Baseline rule (vibration z-score > 2)**")
            st.write(f"Precision: {m['baseline_rule_precision']}")
            st.write(f"Recall: {m['baseline_rule_recall']}")
        st.caption(
            f"⚠️ Based on only {m['n_test_positive_hours']} labelled at-risk hours in the test period "
            "(17 raw failure events fleet-wide over 120 days). Treat these numbers as directional evidence "
            "the approach works, not a tight production estimate -- see README Limitations."
        )


if __name__ == "__main__":
    main()
