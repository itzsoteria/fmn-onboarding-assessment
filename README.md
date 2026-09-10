# FMN AI Engineer Internship -- Technical Assessment

Two tools, both following the same shape: turn raw operational data into a
plain-English, numbers-grounded flag a business user can act on, plus a
free-text way to ask about the underlying data.

- **Project 1:** [Supply Chain Risk Monitor](project1_supply_chain/)
- **Project 2:** [Manufacturing Failure Risk Monitor](project2_manufacturing/)

---

## Problem understanding

**Supply chain.** The sponsor's pain isn't "we don't have data" -- it's that
nobody is turning the data into a forward-looking, per-SKU answer before it's
too late. That means the tool has to do two things at once: forecast what
demand is *about to* look like (not just report what stock is *right now*),
and translate "forecast vs stock vs lead time" into one of a small number of
plain states a planner can scan in seconds. It also has to be honest when it
doesn't have enough history to trust its own forecast (the 3 new SKUs).

**Manufacturing.** The sponsor explicitly said "not just a red light with no
explanation" -- so the bar here isn't "predict failures", it's "predict
failures *and* say why, in terms a plant manager can act on immediately, and
show whether it's getting worse." With only 17 recorded failures across 120
days, I treated this as a genuine small-data problem rather than pretending
otherwise -- the model, its evaluation, and the README below all say so
plainly.

## Approach

### Project 1 -- Supply Chain

1. **Clean:** normalize inconsistent category casing; recover missing
   `units_sold`/`closing_stock` values using the exact inventory identity
   `closing_stock[t] = closing_stock[t-1] - units_sold[t] + units_received[t]`
   rather than a blind forward-fill (see `src/data_prep.py`). Recovered all
   136 missing values this way, out of 4,551 rows across 28 SKUs.
2. **Forecast:** per-SKU EWMA of daily demand. Backtested spans from 3 to 150
   days on the 25 established SKUs' last 30 days each -- accuracy kept
   improving out to ~span-60 and then plateaued, meaning demand here is close
   to statistically stationary (no strong trend/seasonality in this window).
   Landed on **span=60** as a longer-memory-but-still-adaptive choice.
3. **Flag logic:** reorder point = forecast demand over lead time + a safety
   buffer sized to each SKU's own demand volatility (95% service level).
   Two severity tiers rather than one: **CRITICAL** (will run dry before a
   fresh order could even arrive) vs **AT_RISK** (below the reorder trigger
   but there's still time) vs **OVERSTOCK** (holding >2x lead-time's worth of
   demand in stock) vs **OK**.
4. **New SKUs (12 days of history):** forecast falls back to the category's
   established-SKU average demand; always shown with a "limited history"
   flag rather than a confident-looking number built on 12 days of noise.

### Project 2 -- Manufacturing

1. **Clean:** dropped 20 exact-duplicate rows (10 machine/timestamp pairs
   appended out of order near the end of the file); filled scattered sensor
   dropouts with per-machine interpolation (never a global fill -- average
   vibration ranges from ~0.30 to ~1.26 mm/s machine to machine, so a global
   mean would erase each machine's own normal range).
2. **Features:** 24h/72h rolling mean & std of temperature and vibration,
   each machine's deviation from its *own* baseline (z-score -- this is what
   lets the same model reason sensibly about machines with very different
   normal operating ranges), a 24h vibration trend, and run-hours since
   maintenance.
3. **Label:** "does a failure happen anywhere in the next 24 hours" rather
   than "does it happen in this exact hour" -- turns 17 point-in-time events
   into ~408 learnable positive hours, which is both more learnable and
   closer to what the business actually wants to know.
4. **Model:** RandomForestClassifier, class-weight balanced. Chosen over
   gradient boosting because it needed no tuning to be stable at this data
   size and exposes feature importances natively, which ground the LLM
   explanations. Evaluated on a **true future holdout** (trained on data
   before Apr 1 2026, tested on April) -- not a random split, which would
   leak information across a time series.
5. **New machines (72h of history):** never scored by the trained model.
   Instead, their early-life sensor averages are compared -- as a z-score --
   against what the *established* machines' own first 72 hours looked like
   (not their steady-state), so it's an apples-to-apples comparison. Only
   elevated vibration counts as a risk signal (matches the failure pattern
   found in EDA); unusually low vibration is not flagged.
6. **Risk trend:** every machine's risk score is also computed retrospectively
   over its recent history, satisfying the brief's "track how risk is
   trending" -- and it's a strong validation story in its own right: see
   below.

## Model validation

**Supply chain (WAPE, backtested on 25 established SKUs' last 30 days):**

| Method | WAPE |
|---|---|
| EWMA span=60 (our model) | **0.204** |
| Naive: trailing 7-day average | 0.213 |
| Naive: yesterday's value | 0.254 |

**Manufacturing (time-based holdout, trained pre-April, tested on April 2026):**

| Metric | Our model | Baseline rule (vibration z > 2) |
|---|---|---|
| Precision | **0.521** | 0.284 |
| Recall | 0.656 | 0.911 |
| F1 | **0.581** | 0.433 |
| PR-AUC | 0.584 | -- |

The baseline rule catches almost everything (91% recall) at the cost of
swamping a plant manager with false alarms (72% of its flags are wrong). Our
model trades a bit of recall for roughly double the precision -- fewer, more
trustworthy alerts, which is what a "just tell me what needs attention"
tool actually needs.

Retrospectively scoring MCH-206's full history (which had 2 real failures)
shows the model's predicted probability climbing from a ~2% baseline to
90%+ roughly **24-36 hours before both actual failures**, then dropping
immediately after maintenance resets the clock. That lead time is the
sponsor's exact ask ("I want to know ahead of time").

**Honest caveat:** 17 raw failure events is a small base for a supervised
classifier. These numbers are directional evidence the approach works, not a
tight production estimate -- see Limitations below.

## AI-generated explanations & grounded Q&A

Both apps call the Anthropic API at runtime (`shared/llm_client.py`) --
nothing is templated. Each "Explain this flag" click passes that specific
SKU's or machine's actual computed numbers (forecast, thresholds, z-scores,
top model features) into the prompt, so the explanation is grounded in real
figures rather than a generic sentence.

The Q&A box passes the *entire* current snapshot table (28 SKU rows / 17
machine rows) into the prompt as JSON, rather than building a RAG/retrieval
pipeline. At this scale that's simpler, cheaper, and strictly more accurate
than approximate retrieval, since the model sees every row rather than a
top-k guess. **This does not scale** to a real Dangote/FMN-wide deployment
with thousands of SKUs or machines -- see Limitations.

## How to run

Each app is self-contained but imports `shared/llm_client.py`, so keep the
repo structure intact (don't move an app folder out on its own).

```bash
cd project1_supply_chain   # or project2_manufacturing
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # required for explanations & Q&A
streamlit run app.py
```

Both apps run the full pipeline (clean -> features -> model -> snapshot) on
startup and cache the results, so the first load takes a few seconds.

**Deployed links:** _add your Streamlit Community Cloud URLs here after
deploying -- see "Deployment" below._

### Deployment (Streamlit Community Cloud)

1. Push this whole repo to GitHub (keep `shared/` at the top level).
2. On share.streamlit.io, create an app pointing at
   `project1_supply_chain/app.py` (and a second app for
   `project2_manufacturing/app.py`).
3. In each app's Settings -> Secrets, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

## Limitations & next steps

- **Manufacturing model is trained on very few failures.** With more
  history (or more machines), a survival-analysis approach (e.g.
  time-to-failure modeling) would use the data more efficiently than a
  binary classifier and give a probability curve rather than a threshold.
- **No visibility into open purchase orders.** The supply chain flag
  assumes no replenishment is already in transit -- if FMN's systems track
  open POs, feeding that in directly would remove a real source of false
  positives.
- **Q&A doesn't scale past this dataset's size.** Passing the full table
  into the prompt works for 28 SKUs / 17 machines; a fleet-wide rollout
  would need real retrieval (SQL tool-use or a vector store over
  pre-aggregated summaries) instead.
- **Category/fleet-average fallback for new items is a starting point, not
  a forecast.** It gets better the moment those SKUs/machines accumulate a
  few weeks of their own history -- worth revisiting the cutover point
  (currently 21 days / 72 hours) with the business.
- **No alerting/notification layer.** Today both tools are pull (open the
  app to see status); a push mechanism (Slack/email digest for CRITICAL/
  HIGH_RISK items) would close the loop the sponsors actually described
  ("caught off guard").
- **Thresholds (reorder safety factor, overstock multiple, risk
  probability cutoffs) are defensible starting points, not tuned against
  a real cost model.** With sponsor input on the actual cost of a stockout
  vs. excess inventory (or a false alarm vs. a missed failure), these
  should move from "reasonable defaults" to "optimized against the
  business's actual costs."
