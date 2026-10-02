"""CR-POE dashboard. Run: streamlit run app.py (needs pipeline.py output first)."""
import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import OUT_MODEL, OUT_REVIEWS, OUT_STATS, OUT_TOPICS, clean, vader

st.set_page_config(page_title="CR-POE — Crunch Collective", layout="wide")

st.markdown("""<style>
.block-container{padding-top:2.5rem;max-width:1180px}
.card{background:var(--background-color);border:1px solid rgba(128,128,128,.28);
      border-radius:10px;padding:14px 18px;margin-bottom:12px}
.rhead{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
       font-size:12px;color:#7c8797;margin-bottom:8px}
.body{font-size:14.5px;line-height:1.55;margin-bottom:10px}
.badge{display:inline-block;font-size:11.5px;font-weight:600;padding:3px 9px;
       border-radius:5px;margin:2px 4px 2px 0;border:1px solid}
.pos{background:#f0fdf4;color:#15803d;border-color:#bbf7d0}
.neg{background:#fef2f2;color:#b91c1c;border-color:#fecaca}
.mix{background:#fffbeb;color:#b45309;border-color:#fde68a}
.top{background:#f1f5f9;color:#475569;border-color:#e2e8f0}
.act{background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe}
.red{background:#fde68a;color:#78350f;padding:1px 5px;border-radius:3px;font-weight:600}
</style>""", unsafe_allow_html=True)

TONE = {"positive": "pos", "negative": "neg", "neutral": "mix"}
PRESETS = {
    "Packaging defect with PII": "I ordered the protein balls via order #9022 last Thursday. The bag arrived torn open and smelt stale. Please refund me at david_clark@outlook.com.au immediately.",
    "High praise": "These cacao clusters are unreal!! 10/10 crunch, love having them with Greek yogurt every morning. Best guilt-free snack.",
    "Feature request": "Please bring out a dairy-free matcha flavoured bar! I love your products but need more vegan morning options.",
    "Price resistance": "Good taste and healthy ingredients, but honestly too pricey for only 120 grams. Won't reorder unless there's a discount code.",
}


@st.cache_resource
def load(_stamp):  # ponytail: mtime in the key, so a fresh pipeline run is picked up
    reviews = pd.read_parquet(OUT_REVIEWS)
    reviews["has_pii"] = reviews["clean"].str.contains(
        r"\[EMAIL\]|\[PHONE\]|\[ORDER_ID\]|\[HANDLE\]", regex=True)
    opps = pd.read_parquet(OUT_TOPICS)
    model = joblib.load(OUT_MODEL)
    try:
        with open(OUT_STATS) as fh:
            funnel = json.load(fh)
    except FileNotFoundError:
        funnel = {}
    return reviews, opps, model, funnel


try:
    reviews, opps, model, funnel = load(os.path.getmtime(OUT_REVIEWS))
except FileNotFoundError:
    st.error("No results yet. Run `python pipeline.py` first.")
    st.stop()

label_of = dict(zip(opps["topic"], opps["label"]))
action_of = dict(zip(opps["topic"], opps["action"]))

st.title("Crunch Collective — Review Mining Prototype")
st.caption("Automated NLP sentiment and product opportunity engine · Group M36 · ICT802 Capstone")
if funnel:
    st.caption(
        f"Corpus: **{funnel['ingested']:,}** reviews ingested → "
        f"**{funnel['deduplicated']:,}** after removing duplicates → "
        f"**{funnel['snack_reviews']:,}** snack reviews analysed"
        + (f" · {funnel['stored_rows']:,} stored here as evidence"
           if funnel.get("stored_rows", 0) < funnel["snack_reviews"] else "")
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
top = opps.iloc[0]
urgent = opps[opps["tier"] == "Urgent fix"]
k = st.columns(4)
analysed = funnel.get("snack_reviews", len(reviews))
stored = funnel.get("stored_rows", len(reviews))
k[0].metric("Snack reviews analysed", f"{analysed:,}",
            help=f"Filtered from {funnel.get('ingested', 0):,} ingested reviews. "
                 f"{stored:,} are stored here as evidence." if stored < analysed else
                 f"Filtered from {funnel.get('ingested', 0):,} ingested reviews.")
pct_pos = funnel.get("pct_positive", (reviews["sentiment"] == "positive").mean())
k[1].metric("Positive sentiment", f"{pct_pos:.0%}",
            help=f"Mean compound {funnel.get('mean_compound', reviews['compound'].mean()):+.2f}, "
                 f"across all {analysed:,} analysed reviews")
k[2].metric("Urgent complaint topics", len(urgent),
            help="Topics where most reviews are negative")
k[3].metric("Top opportunity", top["label"].split("·")[1].strip(), help=top["action"])

tab1, tab2, tab3 = st.tabs(["Opportunities", "Review explorer", "Live simulator"])

# ── Tab 1: ranked opportunities + matrix ──────────────────────────────────────
with tab1:
    st.plotly_chart(
        px.bar(opps.sort_values("opportunity"), x="opportunity", y="label", orientation="h",
               color="lift", color_continuous_scale="Reds", hover_data=["keywords", "reviews"],
               labels={"opportunity": "Opportunity score", "label": "", "lift": "Negativity lift"},
               height=90 + 42 * len(opps)),
        width='stretch',
    )
    st.subheader("Product opportunity prioritisation matrix")
    st.caption("Opportunity = topic frequency × negativity lift. Lift compares the topic against the corpus baseline, so tiers still work on a corpus that is 91% positive.")
    table = opps.assign(
        Evidence=[reviews[reviews["topic"] == t].nsmallest(1, "compound")["clean"].str[:90].squeeze()
                  if (reviews["topic"] == t).any() else "" for t in opps["topic"]]
    )[["tier", "label", "attribute", "keywords", "Evidence", "reviews", "pct_negative",
       "lift", "opportunity", "action"]]
    st.dataframe(
        table, width='stretch', hide_index=True,
        column_config={
            "tier": "Priority tier", "label": "Topic",
            "attribute": "Attribute",
            "keywords": st.column_config.TextColumn("Top keywords", width="medium"),
            "Evidence": "Customer evidence (most negative)",
            "reviews": st.column_config.NumberColumn("Reviews", format="%d"),
            "pct_negative": st.column_config.ProgressColumn("% negative", min_value=0, max_value=1),
            "lift": st.column_config.NumberColumn("Negativity lift", format="%.2fx",
                                                  help="1.00 = corpus average"),
            "opportunity": st.column_config.NumberColumn("Score", format="%.2f"),
            "action": "Recommended business action",
        },
    )
    st.download_button("Export matrix (CSV)", table.to_csv(index=False).encode(),
                       "opportunities.csv", "text/csv")

    st.subheader("Sentiment over time")
    trend = reviews.set_index("date").groupby(pd.Grouper(freq="YE"))["compound"].mean().reset_index()
    st.plotly_chart(px.line(trend, x="date", y="compound", markers=True,
                            labels={"date": "", "compound": "Mean compound"}),
                    width='stretch')

# ── Tab 2: filterable review cards ────────────────────────────────────────────
with tab2:
    f = st.columns([3, 2, 2])
    topics = f[0].multiselect("Topic", opps["label"], default=list(opps["label"]))
    moods = f[1].multiselect("Sentiment", ["negative", "neutral", "positive"], default=["negative"])
    redact = f[2].toggle("PII regex redaction", value=True,
                         help="Off shows the raw review text as ingested")
    pii_only = f[2].checkbox("Only reviews with detected PII",
                             help="Amazon strips contact details, so these are rare")

    ids = [t for t, lab in label_of.items() if lab in topics]
    hits = reviews[reviews["topic"].isin(ids) & reviews["sentiment"].isin(moods)]
    if pii_only:
        hits = hits[hits["has_pii"]]
    st.caption(f"{len(hits):,} of {len(reviews):,} reviews match. Showing the 15 strongest. "
               f"{reviews['has_pii'].sum():,} reviews in the corpus contain redactable PII "
               f"({reviews['has_pii'].mean():.2%}) — the redaction toggle only changes those.")

    for _, r in hits.reindex(hits["compound"].abs().sort_values(ascending=False).index).head(15).iterrows():
        text = r["clean"] if redact else r["raw"]
        for token in ("[EMAIL]", "[PHONE]", "[ORDER_ID]", "[HANDLE]"):
            text = text.replace(token, f"<span class='red'>{token}</span>")
        tone = TONE[str(r["sentiment"])]
        st.markdown(f"""<div class="card">
          <div class="rhead"><span>{r['ProductId']} · {r['date']:%b %Y}</span>
          <span>{'★' * int(r['Score'])}{'☆' * (5 - int(r['Score']))} {r['Score']}.0</span></div>
          <div class="body">{text[:600]}</div>
          <span class="badge {tone}">VADER {r['compound']:+.2f} · {r['sentiment']}</span>
          <span class="badge top">LDA topic: {label_of[r['topic']]}</span>
          <span class="badge act">{action_of[r['topic']]}</span>
        </div>""", unsafe_allow_html=True)

# ── Tab 3: live pipeline test ─────────────────────────────────────────────────
with tab3:
    st.subheader("Live review simulator")
    st.caption("Paste any review to run it through the same redaction, VADER scoring and "
               "fitted LDA model. Built for live demonstration.")

    if "draft" not in st.session_state:
        st.session_state.draft = ""
    cols = st.columns(len(PRESETS))
    for col, (name, text) in zip(cols, PRESETS.items()):
        if col.button(name, width='stretch'):
            st.session_state.draft = text

    entry = st.text_area("Review text", key="draft", height=120,
                         placeholder="Type or paste a customer review…")

    if entry.strip():
        sanitised = clean(entry)
        score = vader().polarity_scores(sanitised)["compound"]
        mood = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        topic = int(model["lda"].transform(model["vec"].transform([sanitised])).argmax())

        shown = sanitised
        for token in ("[EMAIL]", "[PHONE]", "[ORDER_ID]", "[HANDLE]"):
            shown = shown.replace(token, f"<span class='red'>{token}</span>")
        st.markdown("**Sanitised text**", unsafe_allow_html=True)
        st.markdown(f"<div class='card'>{shown}</div>", unsafe_allow_html=True)

        # Only short values stay in st.metric — it truncates without wrapping.
        out = st.columns(2)
        out[0].metric("Predicted sentiment", mood.title())
        out[1].metric("VADER compound", f"{score:+.2f}")
        st.markdown(f"**LDA topic** — {model['names'][topic]} "
                    f"({model['attributes'][topic]}) &nbsp;·&nbsp; {model['keywords'][topic]}")
        st.success(f"**Recommended action** — {action_of.get(topic, 'Log to the feedback repository')}")
