"""
Supply Chain Risk Monitor -- FMN AI Engineer Internship, Project 1.

Run locally:
    cd project1_supply_chain
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
from model import build_sku_snapshot
from validate import backtest, summarize
import llm_client

st.set_page_config(page_title="Supply Chain Risk Monitor", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "project1_supply_chain_demand.csv")

FLAG_COLORS = {
    "CRITICAL": "🔴",
    "AT_RISK": "🟠",
    "OVERSTOCK": "🔵",
    "OK": "🟢",
}
FLAG_LABELS = {
    "CRITICAL": "Critical -- will stock out before replenishment arrives",
    "AT_RISK": "At risk -- below reorder point, order soon",
    "OVERSTOCK": "Overstocked -- capital tied up",
    "OK": "Healthy",
}


@st.cache_data
def get_data():
    return load_and_clean(DATA_PATH)


@st.cache_data
def get_snapshot(_df):
    return build_sku_snapshot(_df)


def main():
    st.title("📦 Supply Chain Risk Monitor")
    st.caption(
        "\"We keep getting caught off guard — some SKUs run out and delay production, "
        "others sit overstocked and tie up working capital.\" — this tool flags both, ahead of time, with why."
    )

    df = get_data()
    snapshot = get_snapshot(df)

    # ---- Summary row ----
    counts = snapshot["flag"].value_counts()
    cols = st.columns(4)
    for col, flag in zip(cols, ["CRITICAL", "AT_RISK", "OVERSTOCK", "OK"]):
        col.metric(f"{FLAG_COLORS[flag]} {flag}", int(counts.get(flag, 0)))

    st.divider()

    # ---- Filters ----
    left, right = st.columns([1, 3])
    with left:
        flag_filter = st.multiselect(
            "Filter by status", options=list(FLAG_LABELS.keys()),
            default=["CRITICAL", "AT_RISK", "OVERSTOCK"],
            format_func=lambda f: f"{FLAG_COLORS[f]} {f}",
        )
        category_filter = st.multiselect(
            "Filter by category", options=sorted(snapshot["category"].unique())
        )

    view = snapshot[snapshot["flag"].isin(flag_filter)] if flag_filter else snapshot
    if category_filter:
        view = view[view["category"].isin(category_filter)]

    with right:
        st.dataframe(
            view[["sku_id", "category", "flag", "current_stock", "forecast_daily_demand",
                  "days_of_cover", "lead_time_days", "reorder_point", "low_history"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "flag": st.column_config.TextColumn("Status"),
                "low_history": st.column_config.CheckboxColumn("Limited history"),
            },
        )

    st.divider()

    # ---- Drill-down + AI explanation ----
    st.subheader("Drill into a SKU")
    sku_options = view["sku_id"].tolist() or snapshot["sku_id"].tolist()
    selected_sku = st.selectbox("Choose a SKU", options=sku_options)

    if selected_sku:
        row = snapshot[snapshot.sku_id == selected_sku].iloc[0]
        c1, c2 = st.columns([2, 1])

        with c1:
            hist = df[df.sku_id == selected_sku].sort_values("date").tail(60)
            chart_df = hist.set_index("date")[["units_sold", "closing_stock"]]
            st.line_chart(chart_df)

        with c2:
            st.markdown(f"**{FLAG_COLORS[row.flag]} {row.flag}**")
            st.write(f"Current stock: **{row.current_stock:.0f}**")
            st.write(f"Forecast daily demand: **{row.forecast_daily_demand:.1f}**")
            st.write(f"Reorder point: **{row.reorder_point:.1f}**")
            st.write(f"Days of cover: **{row.days_of_cover}**")
            st.write(f"Lead time: **{row.lead_time_days} days**")
            if row.low_history:
                st.warning(f"Limited history: {row.forecast_basis}")

        if st.button("🤖 Explain this flag", key=f"explain_{selected_sku}"):
            with st.spinner("Generating explanation..."):
                try:
                    explanation = llm_client.generate_explanation(
                        entity_id=selected_sku,
                        entity_data=row.to_dict(),
                        domain="supply_chain",
                    )
                    st.info(explanation)
                except Exception as e:
                    st.error(f"Couldn't generate explanation: {e}")

    st.divider()

    # ---- Grounded Q&A ----
    st.subheader("💬 Ask about supply chain risk")
    st.caption('e.g. "Why is SKU-1000 flagged?" or "Which SKUs need attention this week?"')
    question = st.text_input("Your question", key="qa_input")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            try:
                answer = llm_client.answer_question(
                    question=question,
                    full_table=snapshot.to_dict(orient="records"),
                    domain="supply_chain",
                )
                st.success(answer)
            except Exception as e:
                st.error(f"Couldn't answer: {e}")

    with st.expander("📊 Model validation (backtest results)"):
        st.caption(
            "EWMA forecast backtested on the 25 established SKUs' last 30 days each, "
            "compared against naive baselines."
        )
        results = backtest(df)
        st.dataframe(summarize(results), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
