# CR-POE: Customer Review Mining and Product Opportunity Engine

Mines customer reviews and ranks product opportunities, with the supporting
quotes attached to each one.

ICT802 Capstone, Group M36. Sponsor: Crunch Collective.

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

Streamlit Cloud only runs `app.py`, not the pipeline. The model and scored
reviews in `out/` are committed so the hosted app can read them directly.
Point Streamlit Cloud at `app.py` and it works with no extra config.

To refresh the hosted copy, rebuild locally and commit `out/`:

```bash
python pipeline.py --rows 0 --topics 12 --sample 15000
git add out && git commit -m "Refresh hosted model"
```

`--sample` keeps a stratified subset of scored reviews so the committed parquet
fits the free tier's memory. Topic modelling and all stats still run on the
full 65,348 reviews; only the stored evidence rows are trimmed.

## What it does

1. **Ingest.** Load reviews from CSV.
2. **Clean.** Drop duplicates and markup; redact emails, phones, order IDs, handles.
3. **Sentiment.** Score each review from −1 to +1 with VADER, extended with snack-domain words.
4. **Topics.** LDA fitted on complaint reviews, labelled by each topic's distinctive keywords.
5. **Rank.** `opportunity = topic frequency × negativity lift`, where lift is
   the topic's % negative over the corpus baseline.
6. **Present.** Streamlit dashboard with three tabs:
   - *Opportunities:* ranked chart, prioritisation matrix with recommended actions, CSV export.
   - *Review explorer:* filter by topic and sentiment, verbatim evidence, PII redaction toggle.
   - *Live simulator:* paste any review and run it through the fitted model.

## Data

Amazon Fine Food Reviews, from Stanford SNAP (public, no account needed).
`fetch_data.py` downloads the archive and converts it to the CSV layout the
pipeline reads.

> J. McAuley and J. Leskovec, "From amateurs to connoisseurs: modeling the evolution
> of user expertise through online reviews," *Proc. WWW*, 2013.

A 600-row fixture (`data/Reviews.sample.csv`) is in the repo so the pipeline
can be smoke-tested before downloading anything:

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

Sentiment accuracy vs. star ratings (binary, 3-star excluded): **weighted F1 0.884**.

Known limitations (from the evaluation output, not assumed):

- Neutral class fails (F1 0.04); VADER does not handle mixed reviews.
- Negative recall 0.42: just under half of complaints are caught.
- Sarcasm tends to score positive.
- Three of twelve topics stay unlabelled, and the snack keyword filter leaks a few off-domain products.

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
