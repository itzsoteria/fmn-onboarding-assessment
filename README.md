# FMN AI Engineer Internship — Technical Assessment

## What this project does

Two tools that watch everyday business data and flag problems before they
get expensive:

1. **Supply Chain Risk Monitor** — warns which products are about to run
   out of stock, or are sitting in excess, before it becomes a problem.
2. **Machine Failure Risk Monitor** — warns which machines are likely to
   break down soon, and explains why.

Both tools use Claude (an AI assistant) to turn each warning into a plain-
English explanation, and let you type a question and get an answer straight
from the current data.

## The problem we're solving

**Supply Chain.** The business kept getting caught off guard — some
products ran out and delayed production, others piled up and tied up money
that could be used elsewhere. This tool looks ahead for every product and
flags both problems early, instead of finding out after the fact.

**Manufacturing.** Machines were breaking down with no warning, and when
they did, nobody could say why. This tool predicts a likely breakdown
roughly a day in advance and points to exactly which reading — temperature
or vibration — is driving that risk.

## How it works, in plain terms

**Supply Chain Monitor**
- Looks at each product's recent sales to estimate how much will sell over
  the next few days.
- Compares that estimate to how much stock is on hand and how long a new
  order takes to arrive.
- Flags each product as **Critical** (will run out before a new order
  could arrive), **At Risk** (running low — order soon), **Overstocked**
  (too much tied up), or **Healthy**.
- For 3 brand-new products with only two weeks of sales history, it plays
  it safe and uses the average for similar products instead of guessing
  off too little data.

**Machine Failure Monitor**
- Watches each machine's temperature and vibration over time.
- Learns what "normal" looks like for *that specific machine* — a reading
  that's normal for one machine can be a real warning sign on another.
- Estimates the chance a machine fails in the next 24 hours, and names
  which reading is driving that estimate.
- For 2 newly-installed machines with only a few days of history, it
  compares them to how other machines behaved in their own first few
  days, rather than guessing from almost no data.

## Does it actually work?

- The supply chain forecast is roughly **20% more accurate** than simply
  assuming "tomorrow looks like yesterday."
- Tested against two real historical machine breakdowns the model had
  never seen during training, its risk score reliably climbed a day or
  more *before* both events — then dropped back to normal right after
  each machine was serviced.
- Being upfront about a limit: only 17 breakdowns happened across the
  whole 4-month dataset. That's enough to show the approach clearly
  works, but not enough to call the accuracy numbers production-grade —
  they'd firm up with more data over time.

## The AI explanations

Every explanation is generated live from that product's or machine's
actual numbers — nothing is a pre-written template. The question box
works the same way: ask something in plain English, and it answers using
the real, current data behind the tool, not a guess.

## How to run it

```bash
cd project1_supply_chain          # or project2_manufacturing
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

**Live links:** *add your deployed Streamlit links here once deployed.*

## What's not finished yet, and what we'd do next

- We don't currently know if a product already has a new order on the
  way — with that information, some "Critical" warnings would likely
  turn out to already be handled.
- The machine model is based on very few real breakdowns, so its accuracy
  should be expected to improve as more data comes in.
- Right now, both tools require someone to open the app and check — a
  real rollout would send an alert automatically instead.
- The cutoff points (e.g. what counts as "too much stock" or "high risk")
  are reasonable starting estimates, not numbers agreed with the business
  yet — worth refining together.
- The question-answering feature works well at this size; a company-wide
  rollout with thousands of products or machines would need a more
  scalable search approach behind it.