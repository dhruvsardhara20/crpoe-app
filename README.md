# CR-POE — Customer Review Mining and Product Opportunity Engine

Reads customer reviews and returns a ranked list of product opportunities, each
backed by real customer quotes.

ICT802 Capstone · Group M36 · Sponsor: Crunch Collective

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"

python pipeline.py --test                              # self-check, no data needed
python fetch_data.py                                   # download the dataset
python pipeline.py --rows 0 --topics 12                # build the model
streamlit run app.py                                   # dashboard
```

Windows from scratch, with no Python installed: see [SETUP_WINDOWS.md](SETUP_WINDOWS.md).

## Hosting on Streamlit Community Cloud

Streamlit Cloud runs `app.py` only — it never runs `pipeline.py` or `fetch_data.py`.
So the model and scored reviews in `out/` are committed to this repo, and the app
reads them directly. Point Streamlit Cloud at `app.py` and it works with no
further configuration.

To refresh what is hosted, rebuild locally and commit `out/`:

```bash
python pipeline.py --rows 0 --topics 12 --sample 15000
git add out && git commit -m "Refresh hosted model"
```

`--sample` keeps a stratified subset of the scored reviews so the committed file
stays small and fits the free tier's memory limit. Topic modelling and all
statistics still use the full 65,348 reviews; only the stored evidence is trimmed.

## What it does

1. **Ingest** — load reviews from CSV.
2. **Clean** — remove duplicates and markup, redact emails, phones, order IDs, handles.
3. **Sentiment** — score each review from −1 to +1 with VADER, extended with snack-domain words.
4. **Topics** — LDA fitted on complaint reviews, named by each topic's distinctive keywords.
5. **Rank** — `opportunity = topic frequency × negativity lift`, where lift compares a
   topic against the corpus baseline.
6. **Present** — Streamlit dashboard:
   - **Opportunities** — ranked chart, prioritisation matrix with recommended actions, CSV export
   - **Review explorer** — filter by topic and sentiment, verbatim evidence, PII redaction toggle
   - **Live simulator** — paste any review and run it through the real model

## Data

**Amazon Fine Food Reviews** — Stanford SNAP, public and free, no account needed.
`fetch_data.py` downloads from the primary source and converts it to CSV.

> J. McAuley and J. Leskovec, "From amateurs to connoisseurs: modeling the evolution
> of user expertise through online reviews," *Proc. WWW*, 2013.

A 600-row fixture (`data/Reviews.sample.csv`) ships with the repo so the pipeline runs
before anything is downloaded:

```bash
python pipeline.py --raw data/Reviews.sample.csv --rows 0 --topics 4
```

Corpus funnel on the full dataset:

| Stage | Reviews |
|---|---|
| Ingested | 568,454 |
| After removing duplicates | 393,892 |
| Snack reviews analysed | 65,348 |

## Results

Sentiment accuracy against star ratings, binary (3-star excluded): **weighted F1 0.884**.

Known limitations, measured not assumed:

- Neutral class fails (F1 0.04) — VADER cannot detect mixed reviews
- Negative recall 0.42 — under half of complaints are caught
- Sarcasm scores positive
- Three of twelve topics are unlabelled; the snack keyword filter leaks off-domain products

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Load, clean, redact PII, sentiment, topics, ranking. `--test` runs a self-check |
| `app.py` | Streamlit dashboard |
| `fetch_data.py` | Download and convert the dataset |
| `data/` | Input CSVs — only the sample fixture is committed |
| `out/` | Model and scored reviews — **committed**, required for hosting |

## Stack

Python 3.12, pandas, NLTK (VADER), scikit-learn (LDA), Streamlit, Plotly.
All free and open source.
