# CR-POE pipeline. Reads data/Reviews.csv, writes scored reviews and ranked
# opportunity topics to out/.
#   python pipeline.py            # 50k sample
#   python pipeline.py --rows 0   # full dataset
import argparse
import os
import re
import sys

import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

RAW = "data/Reviews.csv"
OUT_REVIEWS = "out/reviews_scored.parquet"
OUT_TOPICS = "out/opportunities.parquet"
OUT_MODEL = "out/model.joblib"
OUT_STATS = "out/stats.json"

# ProductIds are opaque Amazon codes, so we filter snack reviews by keyword.
SNACK_WORDS = r"chip|crisp|cookie|biscuit|cracker|popcorn|pretzel|granola|snack|bar\b|nuts|trail mix"

# Attributes the sponsor cares about. Used to label LDA topics.
ATTRIBUTES = {
    "taste": r"taste|flavou?r|delicious|bland|yummy",
    "texture": r"crunch|crisp|soggy|stale|chewy|hard|soft",
    "packaging": r"packag|box|bag|seal|crush|broken|melt",
    "price": r"price|expensive|cheap|value|cost|money",
    "sweetness": r"sweet|sugar|sugary|splenda|stevia|artificial",
    "health": r"healthy|calorie|organic|gluten|protein|fat\b|natural",
    "delivery": r"ship|deliver|arriv|late|prime|order",
}

# Marketing action per attribute, keyed as (if negative lift, if positive).
ACTIONS = {
    "packaging": ("Escalate to QA and the packaging supplier", "Feature packaging in social proof"),
    "delivery": ("Audit 3PL courier SLAs", "Promote fast dispatch in campaigns"),
    "price": ("Introduce bundle and subscription pricing", "Hold pricing; value is landing"),
    "texture": ("Brief R&D on freshness and shelf life", "Lead with texture in product copy"),
    "taste": ("Reformulate the weakest-scoring flavours", "Feature in marketing social proof"),
    "sweetness": ("Trial a reduced-sugar variant", "Keep the current sweetness profile"),
    "health": ("Scope a nut-free / low-sugar SKU", "Push clean-ingredient messaging"),
    "other": ("Log to the feedback repository for review", "Log to the feedback repository"),
}

# Domain additions to VADER's lexicon (scale: -4..+4). Set by looking at
# misclassified reviews; "stale", "soggy" etc. are not in the base lexicon.
DOMAIN_LEXICON = {
    "stale": -2.6, "soggy": -2.2, "mouldy": -3.2, "moldy": -3.2, "expired": -2.6,
    "crushed": -2.2, "torn": -2.0, "broken": -2.0, "leaked": -2.0, "melted": -1.4,
    "chalky": -1.8, "bland": -1.8, "overpriced": -2.4, "pricey": -1.6, "refund": -1.6,
    "crunchy": 1.8, "crisp": 1.5, "crispy": 1.8, "fresh": 1.6, "moreish": 2.0,
}

# Extra stopwords on top of ENGLISH_STOP_WORDS. Two groups:
#   - generic praise ("like", "good", "really") that otherwise dominates topics
#   - product nouns ("chocolate", "cookies", "chips") so LDA clusters by issue
#     not category. Product is known from ProductId.
NOISE = """like just good great taste tastes tasted love really product products
amazon buy bought order ordered try tried get got make makes made price flavor
flavors flavour snack snacks food eat eating one two also would could box bag bags
ve don didn doesn wasn isn aren haven wouldn couldn ll did thought time use used
little better
chocolate cookies cookie bar bars chips crisps popcorn corn nuts nut peanut peanuts
butter cereal rice flour wheat biscuits crackers tea coffee candy vanilla cream jerky
green cheese salt salty sauce granola oat oats fruit dried mix dog dogs treats
pretzel pretzels almond almonds cashew raisin
""".split()

PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"#\d{4,8}\b"), "[ORDER_ID]"),
    (re.compile(r"(?<![\w.])@\w{3,}"), "[HANDLE]"),
    (re.compile(r"\b(?:\+?61|0)[2-478](?:[ -]?\d){8}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
]
# Stage counts for the ingest funnel, serialised to out/stats.json.
FUNNEL: dict[str, int] = {}

TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
SPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    # Case and "not" are kept because VADER needs them.
    text = TAG.sub(" ", str(text))
    text = URL.sub(" ", text)
    for pattern, token in PII:
        text = pattern.sub(token, text)
    return SPACE.sub(" ", text).strip()


def vader():
    # Downloads the lexicon on first use so Streamlit Cloud works without a setup step.
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(DOMAIN_LEXICON)
    return sia


def load(rows: int, raw: str = RAW) -> pd.DataFrame:
    df = pd.read_csv(raw, nrows=rows or None)
    FUNNEL["ingested"] = len(df)
    # Reviews are reposted across product variants; dedupe on author + time + text.
    df = df.drop_duplicates(subset=["UserId", "Time", "Text"])
    FUNNEL["deduplicated"] = len(df)
    df = df[df["Text"].notna() & (df["Text"].str.len() > 20)]
    df = df[df["Text"].str.contains(SNACK_WORDS, case=False, na=False)].copy()
    FUNNEL["snack_reviews"] = len(df)
    for k, v in FUNNEL.items():
        print(f"  {k:16} {v:>9,}  ({v / FUNNEL['ingested']:6.1%})")

    df["raw"] = df["Text"].astype(str)
    df["clean"] = df["Text"].map(clean)
    df["date"] = pd.to_datetime(df["Time"], unit="s")
    # star_label is the ground truth we evaluate VADER against.
    df["star_label"] = pd.cut(df["Score"], [0, 2, 3, 5], labels=["negative", "neutral", "positive"])
    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sia = vader()
    df["compound"] = [sia.polarity_scores(t)["compound"] for t in df["clean"]]
    df["sentiment"] = pd.cut(
        df["compound"], [-1.01, -0.05, 0.05, 1.0], labels=["negative", "neutral", "positive"]
    )
    return df


def add_topics(df: pd.DataFrame, n_topics: int = 8):
    # Fit LDA on complaints only. Fitting on the whole corpus produces
    # product-category topics (cookies, chocolate), not issue topics.
    # Use stars OR VADER so this still works on unrated sponsor data.
    complaints = df[(df["Score"] <= 3) | (df["compound"] < 0.05)]
    if len(complaints) < 50:
        complaints = df
    stop = list(ENGLISH_STOP_WORDS) + NOISE
    # min_df scales with corpus size so the fixture and the 568k row run both work.
    vec = CountVectorizer(
        max_df=0.25, min_df=min(15, max(2, len(complaints) // 50)), stop_words=stop,
        ngram_range=(1, 2), max_features=5000,
    )
    vec.fit(complaints["clean"])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method="online")
    lda.fit(vec.transform(complaints["clean"]))
    print(f"topics fitted on {len(complaints):,} complaint reviews")
    df["topic"] = lda.transform(vec.transform(df["clean"])).argmax(axis=1)

    # Name topics from the keywords unique to each one (the attribute alone is
    # too coarse, five topics can map to "health"). Attribute stays as its own column.
    vocab = vec.get_feature_names_out()
    tops = [[vocab[j] for j in w.argsort()[-12:][::-1]] for w in lda.components_]
    names, keywords, attributes = [], [], []
    for i, words in enumerate(tops):
        shared = {w for k, other in enumerate(tops) if k != i for w in other[:8]}
        unique = [w for w in words if w not in shared] or words
        matched = [a for a, pat in ATTRIBUTES.items() if re.search(pat, " ".join(words[:6]))]
        prefix = matched[0].title() if matched else "Unlabelled"
        names.append(f"{prefix} · {', '.join(unique[:2])}")
        keywords.append(", ".join(words[:8]))
        attributes.append("/".join(matched[:2]) or "unlabelled")
    return df, (names, keywords, attributes), (vec, lda)


def rank(df: pd.DataFrame, labels: tuple[list[str], list[str], list[str]]) -> pd.DataFrame:
    # Opportunity score = topic frequency * negativity lift vs corpus baseline.
    g = df.groupby("topic")
    out = pd.DataFrame(
        {
            "topic": g.size().index,
            "reviews": g.size().values,
            "mean_compound": g["compound"].mean().values,
            "pct_negative": g["sentiment"].apply(lambda s: (s == "negative").mean()).values,
        }
    )
    names, keywords, attributes = labels
    out["label"] = [names[t] for t in out["topic"]]
    out["keywords"] = [keywords[t] for t in out["topic"]]
    out["attribute"] = [attributes[t] for t in out["topic"]]
    out["frequency"] = out["reviews"] / out["reviews"].sum()
    out["severity"] = (1 - out["mean_compound"]) / 2  # compound [-1,1] -> [1,0]
    # This corpus is 91% positive, so an absolute negativity threshold flags
    # nothing. Lift is the topic's % negative over the corpus baseline.
    baseline = (df["sentiment"] == "negative").mean() or 1e-9
    out["lift"] = (out["pct_negative"] / baseline).round(2)
    out["severity_score"] = (out["frequency"] * out["severity"] * 100).round(2)
    out["opportunity"] = (out["frequency"] * out["lift"] * 100).round(2)
    out["tier"] = out["lift"].map(
        lambda x: "Urgent fix" if x >= 1.5 else "Strategic review" if x >= 1.15 else "Strength"
    )
    out["action"] = [
        ACTIONS.get(a.split("/")[0], ACTIONS["other"])[0 if x >= 1.15 else 1]
        for a, x in zip(out["attribute"], out["lift"])
    ]
    return out.sort_values("opportunity", ascending=False).reset_index(drop=True)


def accuracy(df: pd.DataFrame) -> None:
    from sklearn.metrics import classification_report

    b = df[df["Score"] != 3]
    print("\nBinary (3-star excluded):")
    print(classification_report(
        (b["Score"] >= 4).map({True: "positive", False: "negative"}),
        (b["compound"] >= 0.05).map({True: "positive", False: "negative"}),
        zero_division=0, digits=3,
    ))
    ok = df["star_label"].notna() & df["sentiment"].notna()
    print("Three-class (VADER struggles on mixed reviews):")
    print(classification_report(df.loc[ok, "star_label"], df.loc[ok, "sentiment"],
                                zero_division=0, digits=3))


def test() -> None:
    assert clean("<br />bad!! me@x.com") == "bad!! [EMAIL]"
    assert clean("order #9022 via @crunchco") == "order [ORDER_ID] via [HANDLE]"
    assert vader().polarity_scores("the bag was torn and the chips were stale")["compound"] < -0.4
    assert "not" in clean("not good"), "negation must survive cleaning"
    df = pd.DataFrame({"compound": [-0.8, 0.9], "sentiment": ["negative", "positive"], "topic": [0, 1]})
    df["sentiment"] = df["sentiment"].astype("category")
    r = rank(df, (["T0 · broken", "T1 · yum"], ["bag, broken", "yum, tasty"],
                  ["packaging", "taste"]))
    assert r.iloc[0]["topic"] == 0, "the above-baseline topic must rank first"
    assert r.iloc[0]["tier"] == "Urgent fix" and "supplier" in r.iloc[0]["action"]
    assert r.iloc[1]["tier"] == "Strength"
    print("self-check ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000, help="0 = all rows")
    ap.add_argument("--topics", type=int, default=8)
    ap.add_argument("--raw", default=RAW, help="input CSV (use data/Reviews.sample.csv to smoke-test)")
    ap.add_argument("--sample", type=int, default=0,
                    help="rows to keep in the saved review file (0 = all); use for hosting")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        return test()

    os.makedirs("out", exist_ok=True)
    df = load(args.rows, args.raw)
    df = add_sentiment(df)
    df, labels, model = add_topics(df, args.topics)
    opps = rank(df, labels)
    accuracy(df)

    joblib.dump({"vec": model[0], "lda": model[1], "names": labels[0],
                 "keywords": labels[1], "attributes": labels[2]}, OUT_MODEL)
    cols = ["ProductId", "date", "Score", "Summary", "raw", "clean", "compound", "sentiment", "topic"]
    saved = df[cols]
    if args.sample and args.sample < len(saved):
        # Stratify by topic so each topic keeps evidence rows after sampling.
        idx = saved.groupby("topic").sample(frac=args.sample / len(saved), random_state=0).index
        saved = saved.loc[idx]
        print(f"saving a {len(saved):,}-row sample for hosting (full run used {len(df):,})")
    saved.to_parquet(OUT_REVIEWS, index=False)
    opps.to_parquet(OUT_TOPICS, index=False)
    import json
    stats = dict(FUNNEL)
    stats["stored_rows"] = len(saved)
    for mood in ("positive", "negative", "neutral"):
        stats[f"pct_{mood}"] = float((df["sentiment"] == mood).mean())
    stats["mean_compound"] = float(df["compound"].mean())
    with open(OUT_STATS, "w") as fh:
        json.dump(stats, fh)
    print("\nTop opportunities:")
    print(opps[["label", "reviews", "pct_negative", "opportunity"]].head())
    print(f"\nwrote {OUT_REVIEWS}, {OUT_TOPICS} and {OUT_MODEL}")


if __name__ == "__main__":
    sys.exit(main())
