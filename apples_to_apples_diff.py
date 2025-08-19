#!/usr/bin/env python3
# apples_to_apples_diff.py

import csv, hashlib, io, os, sys, datetime, time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import urllib3

print("SCRIPT_VERSION: 2025-08-19-1")  # <-- shows up in logs so we know this file is the one running

# --- CONFIG ---
TARGETS = [
    ("Enbridge/Dominion - Residential",
     "https://www.energychoice.ohio.gov/ApplesToApplesComparision.aspx?Category=NaturalGas&RateCode=1&TerritoryId=1"),
]

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)
today = datetime.date.today().isoformat()

# Silence warnings if we use verify=False as fallback
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36")
}

def http_get(url, max_retries=3, backoff=2.0):
    last_err = None
    for _ in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    # Final fallback: skip SSL verification for this public CSV/page
    r = requests.get(url, headers=HEADERS, timeout=60, verify=False)
    r.raise_for_status()
    return r

def find_csv_export_link(page_url):
    r = http_get(page_url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Primary: search by visible link text
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        if "export all offers to csv" in text:
            href = a.get("href")
            if href:
                return href if href.startswith(("http://", "https://")) else urljoin(page_url, href)

    # Fallback: any link that looks like CSV
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".csv" in href.lower():
            return href if href.startswith(("http://", "https://")) else urljoin(page_url, href)

    raise RuntimeError("CSV export link not found on page: " + page_url)

def normalized_df_from_csv(url):
    r = http_get(url)
    data = io.BytesIO(r.content)
    df = pd.read_csv(data, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    id_cols = [c for c in df.columns if ("offer" in c.lower() and "id" in c.lower())]
    if id_cols:
        keycol = id_cols[0]
    else:
        key_fields = []
        for c in [
            "Supplier", "Company", "Company Name", "Supplier Name",
            "Rate", "Price", "Term", "Term (Months)", "Offer", "Offer Details",
            "Terms of Service", "Early Termination Fee", "Monthly Fee",
            "Introductory", "Renewable Percentage", "Cancellation Fee",
        ]:
            if c in df.columns:
                key_fields.append(c)

        def make_key(row):
            blob = "||".join(str(row.get(c, "")) for c in key_fields)
            return hashlib.sha1(blob.encode("utf-8")).hexdigest()

        df["OfferKey"] = df.apply(make_key, axis=1)
        keycol = "OfferKey"

    df["_Key"] = df[keycol]
    return df

def diff_frames(prev, curr):
    key = "_Key"
    prev = prev.set_index(key, drop=False)
    curr = curr.set_index(key, drop=False)

    added_keys = curr.index.difference(prev.index)
    removed_keys = prev.index.difference(curr.index)
    common_keys = curr.index.intersection(prev.index)

    changed = []
    for k in common_keys:
        rp, rc = prev.loc[k], curr.loc[k]
        diffs = {}
        for col in sorted(set(curr.columns).intersection(prev.columns)):
            if col.startswith("_"):
                continue
            vp = "" if pd.isna(rp[col]) else str(rp[col])
            vc = "" if pd.isna(rc[col]) else str(rc[col])
            if vp != vc:
                diffs[col] = {"before": vp, "after": vc}
        if diffs:
            changed.append({
                "key": k,
                "supplier": rc.get("Supplier") or rc.get("Company") or "",
                "changes": diffs
            })

    return {
        "added": curr.loc[added_keys].to_dict(orient="records"),
        "removed": prev.loc[removed_keys].to_dict(orient="records"),
        "changed": changed
    }

def main():
    reports = []
    for name, page in TARGETS:
        csv_url = find_csv_export_link(page)
        curr_df = normalized_df_from_csv(csv_url)

        snap_path = os.path.join(OUTDIR, f"{name}_{today}.csv".replace(" ", "_"))
        curr_df.to_csv(snap_path, index=False)

        prefix = f"{name}_".replace(" ", "_")
        prev_files = sorted(
            f for f in os.listdir(OUTDIR)
            if f.startswith(prefix) and f.endswith(".csv") and today not in f
        )

        if prev_files:
            prev_df = pd.read_csv(os.path.join(OUTDIR, prev_files[-1]), dtype=str)
            report = diff_frames(prev_df, curr_df)
        else:
            report = {"added": curr_df.to_dict(orient="records"), "removed": [], "changed": []}

        reports.append((name, report))

    lines = []
    for name, r in reports:
        lines.append(f"=== {name} ===")
        lines.append(f"New: {len(r['added'])} | Removed: {len(r['removed'])} | Changed: {len(r['changed'])}")
        for ch in r["changed"]:
            who = ch.get("supplier") or ch["key"]
            lines.append(f"- {who}:")
            for col, vals in ch["changes"].items():
                lines.append(f"    {col}: '{vals['before']}' -> '{vals['after']}'")
        lines.append("")

    report_txt = "\n".join(lines)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(report_txt)

if __name__ == "__main__":
    sys.exit(main())
