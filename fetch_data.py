# Fetch Amazon Fine Food Reviews from Stanford SNAP (no Kaggle token needed)
# and convert to the CSV layout the pipeline expects. ~122 MB gz, ~300 MB CSV.
#   python fetch_data.py
import csv
import gzip
import os
import urllib.request

URL = "https://snap.stanford.edu/data/finefoods.txt.gz"
GZ = "data/finefoods.txt.gz"
OUT = "data/Reviews.csv"

# Map SNAP's record keys to the Kaggle CSV columns the pipeline reads.
FIELDS = {
    "product/productId": "ProductId",
    "review/userId": "UserId",
    "review/profileName": "ProfileName",
    "review/helpfulness": "Helpfulness",
    "review/score": "Score",
    "review/time": "Time",
    "review/summary": "Summary",
    "review/text": "Text",
}
COLUMNS = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator",
           "HelpfulnessDenominator", "Score", "Time", "Summary", "Text"]


def download() -> None:
    if os.path.exists(GZ):
        print(f"{GZ} already present, skipping download")
        return
    print(f"downloading {URL} (~122 MB)…")
    urllib.request.urlretrieve(URL, GZ + ".part")
    os.replace(GZ + ".part", GZ)


def convert() -> None:
    # SNAP format: one "key: value" line per field, blank line between records.
    n = 0
    with gzip.open(GZ, "rt", encoding="latin-1") as fh, open(OUT, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        record = {}
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if record:
                    writer.writerow(row(record, n))
                    n += 1
                    if n % 100_000 == 0:
                        print(f"  {n:,} reviews")
                record = {}
                continue
            key, _, value = line.partition(": ")
            if key in FIELDS:
                record[FIELDS[key]] = value
        if record:
            writer.writerow(row(record, n))
            n += 1
    print(f"wrote {OUT} — {n:,} reviews")


def row(record: dict, n: int) -> dict:
    helpful = record.pop("Helpfulness", "0/0")
    num, _, den = helpful.partition("/")
    return {
        "Id": n + 1,
        "HelpfulnessNumerator": num or 0,
        "HelpfulnessDenominator": den or 0,
        **{c: record.get(c, "") for c in COLUMNS if c in record or c in FIELDS.values()},
    }


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download()
    convert()
