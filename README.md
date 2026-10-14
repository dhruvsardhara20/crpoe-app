# CR-POE — Customer Review Mining and Product Opportunity Engine

**ICT802 Capstone Project · Group M36 · Victorian Institute of Technology**
Sponsor: Crunch Collective · Supervisor: Swathiga Guruswamy

Crunch Collective is a snack and wellness brand selling direct to customers online.
Its five-person marketing team cannot read the volume of reviews and survey
responses it now receives, so product decisions are driven by anecdote rather than
evidence. CR-POE reads that feedback automatically and returns a ranked list of
product opportunities, each backed by real customer quotes.

## What it does

1. **Ingest** — load customer reviews from CSV.
2. **Clean** — remove duplicates and markup, redact emails and phone numbers.
3. **Sentiment** — score every review from −1 to +1 with VADER.
4. **Topics** — group reviews into themes with LDA, labelled against seven snack
   attributes: taste, texture, packaging, price, sweetness, health, delivery.
5. **Rank** — score each topic by how often it comes up and how negative it is.
6. **Present** — a Streamlit dashboard in three tabs:
   - **Opportunities** — ranked bar chart, prioritisation matrix with recommended
     business actions and CSV export, sentiment trend.
   - **Review explorer** — filter by topic and sentiment, browse review cards with
     star rating, VADER score, topic and action badges, and a PII redaction toggle.
   - **Live simulator** — paste any review and run it through the real pipeline
     live, for demonstrations.

`opportunity = topic frequency × mean severity × 100`, so a complaint that is both
common and strongly negative ranks highest.

## Data

**Amazon Fine Food Reviews** (Stanford SNAP, public, free on Kaggle) — 568,454
reviews of food and snack products, 1999–2012. Used as an approved stand-in while
sponsor data is unavailable; the pipeline accepts any CSV with a text column.

> J. McAuley and J. Leskovec, "From amateurs to connoisseurs: modeling the
> evolution of user expertise through online reviews," *Proc. WWW*, 2013.

A 600-row fixture (`data/Reviews.sample.csv`) ships with the repo so the pipeline
runs before the full dataset is downloaded.

## Setup

On Windows from scratch (no Python installed), follow [SETUP_WINDOWS.md](SETUP_WINDOWS.md) instead.

```bash
uv venv && uv pip install -r requirements.txt
```

<details>
<summary>Without uv</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
</details>

Then fetch the VADER lexicon, and put `Reviews.csv` from Kaggle into `data/`:

```bash
.venv/bin/python -c "import nltk; nltk.download('vader_lexicon')"
```

## Run

```bash
.venv/bin/python pipeline.py --test                                  # self-check, no data needed
.venv/bin/python pipeline.py --raw data/Reviews.sample.csv --rows 0  # fixture run
.venv/bin/python pipeline.py                                         # 50k sample
.venv/bin/python pipeline.py --rows 0                                # full 568k dataset
.venv/bin/python -m streamlit run app.py                             # dashboard
```

`pipeline.py` writes `out/reviews_scored.parquet`, `out/opportunities.parquet` and
`out/model.joblib` (the fitted vectoriser and LDA, reused by the live simulator),
and prints a classification report comparing VADER against the star ratings.

VADER's lexicon is extended with snack-domain words (`stale`, `soggy`, `crushed`,
`crunchy`, `overpriced`, …) because the stock lexicon has no opinion on them. On the
sample fixture this lifts accuracy from 0.80 to 0.90 and negative recall from 0.66
to 1.00.

## Structure

| Path | Purpose |
|---|---|
| `pipeline.py` | Load, clean, redact PII, VADER sentiment, LDA topics, opportunity ranking |
| `app.py` | Streamlit dashboard: ranked opportunities, verbatim evidence, trend |
| `data/` | Input CSVs — only the sample fixture is committed |
| `out/` | Generated parquet files, not committed |
| `SETUP_WINDOWS.md` | Step-by-step Windows install for a machine with nothing on it |
| `PLAN.md` | Sprint plan, task owners, supervisor agenda |
| `docs/` | Assignment briefs, submissions, templates, meeting material |

## Stack

Python 3, pandas, NLTK (VADER), scikit-learn (LDA), Streamlit, Plotly. All free and
open source, matching the zero-budget constraint in the proposal.

## Team

| Member | ID | Role |
|---|---|---|
| Dhruv J. Sardhara | dh66379 | Team Leader — planning, risk, quality, supervisor liaison |
| Deep Padsala | de65795 | Data Engineering — ingestion, cleaning, PII redaction |
| Dhruv P. Patel | dh66262 | NLP & ML — sentiment, topic modelling, ranking |
| Axit A. Dhameliya | ax66506 | Dashboard & QA — UI, visualisation, testing |

## Status

Proposal (Part A) and Project Plan (Part B) submitted. Sprint 1 in progress:
baseline pipeline running, real dataset and accuracy evidence outstanding.
Next milestone: Specification & Design, Session 8. See `PLAN.md`.
