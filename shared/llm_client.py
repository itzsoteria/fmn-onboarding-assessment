"""
Shared Anthropic API wrapper for both apps.

DESIGN DECISION -- grounding strategy:
  Both datasets are small (28 SKUs, 17 machines). Rather than building a
  RAG pipeline (embeddings + vector store) to retrieve "relevant" rows,
  we pass the full current risk/status table straight into the prompt as
  structured JSON -- at this scale that's simpler, cheaper, faster, and
  strictly more accurate than an approximate retrieval step, because the
  model sees everything rather than a top-k guess.
  This does NOT scale to thousands of SKUs/machines -- see README
  "Limitations" for what would change (retrieval, SQL tool-use, or a
  pre-aggregation layer) at real Dangote/FMN scale.

Requires the ANTHROPIC_API_KEY environment variable to be set. In
Streamlit deployment, set it as a secret (see README "How to run").
"""
import json
import os
import streamlit as st
import anthropic

MODEL = "claude-sonnet-5"  # good balance of quality/cost for structured business explanations


def _get_api_key() -> str:
    # Support both a local .env-style environment variable and Streamlit's
    # secrets manager (used automatically on Streamlit Community Cloud).
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Set it as an environment variable "
            "locally, or add it to .streamlit/secrets.toml when deployed."
        )
    return key


_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=_get_api_key())
    return _client


def generate_explanation(entity_id: str, entity_data: dict, domain: str) -> str:
    """
    Generate a plain-English explanation for why one SKU or machine was
    flagged, grounded in its actual numbers (never a template string).

    entity_data: dict of the row's fields (forecast, thresholds, sensor
                 stats, feature importances, etc.) -- whatever the caller's
                 model module already computed.
    domain: "supply_chain" or "manufacturing" -- just shapes the framing.
    """
    role = {
        "supply_chain": "an inventory planner explaining a stock risk flag to a Supply Chain manager",
        "manufacturing": "a reliability engineer explaining a machine risk flag to a plant manager",
    }[domain]

    prompt = f"""You are {role}. Below is the underlying data for {entity_id}, as JSON.

Data:
{json.dumps(entity_data, indent=2, default=str)}

Write a short explanation (2-4 sentences, plain English, no jargon) of:
1. Why this item was flagged the way it was
2. What specific numbers in the data drove that flag
3. What action the business user should consider

Use only the numbers given above -- do not invent figures. If a value indicates
low confidence (e.g. limited history), say so plainly rather than hiding it."""

    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def answer_question(question: str, full_table: list, domain: str) -> str:
    """
    Grounded free-text Q&A. `full_table` is the list-of-dicts snapshot
    (every SKU or every machine's current numbers) -- the model answers
    strictly from this, not from general knowledge.
    """
    role = {
        "supply_chain": "a Supply Chain analyst assistant",
        "manufacturing": "a plant reliability assistant",
    }[domain]

    prompt = f"""You are {role}. Answer the user's question using ONLY the data table below.
If the answer isn't in the data, say so -- do not guess or use outside knowledge.
Cite specific IDs and numbers from the table in your answer.

Data table ({len(full_table)} rows), as JSON:
{json.dumps(full_table, indent=2, default=str)}

User question: {question}

Answer in 2-5 sentences, plain English, referencing the specific rows and numbers that support your answer."""

    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
