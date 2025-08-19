import csv, hashlib, io, os, sys, datetime, requests
from bs4 import BeautifulSoup
import pandas as pd

# -------- CONFIG: add pages you care about --------
TARGETS = [
    ("Enbridge/Dominion - Residential",
     "https://www.energychoice.ohio.gov/ApplesToApplesComparision.aspx?Category=NaturalGas&RateCode=1&TerritoryId=1"),
    # Example extras (uncomment when ready):
    # ("Duke - Residential", "https://www.energychoice.ohio.gov/ApplesToApplesComparision.aspx?Category=NaturalGas&RateCode=1&TerritoryId=10"),
    # ("CenterPoint - Residential", "https://www.energychoice.ohio.gov/ApplesToApplesComparision.aspx?Category=NaturalGas&RateCode=1&TerritoryId=2"),
]
OUTDIR = "data"  # where snapshots are stored

os.makedirs(OUTDIR, exist_ok=True)
today = datetime.date.today().isoformat()

def find_csv_export_link(page_url):
    r = requests.get(page_url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        if "export all offers to csv" in text:
            href = a.get("href")
            if not href:
                continue
            if href.startswith(("http://", "https://")):
                return href
            from urllib.parse import urljoin
            return urljoin(page_url, href)
    raise RuntimeError("CSV export link not found on page: " + page_url)

def normalized_df_from_csv(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = io.BytesIO(r.content)
    df = pd.read_csv(data, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Prefer a real ID if present; otherwise build a stable hash
    id_cols = [c for c in df.columns if ("offer" in c.lower() and "id" in c.lower())]
    if id_cols:
        keycol = id_cols[0]
    else:
        key_fields = []
        for c in ["Supplier", "Company", "Company Name", "Supplier Name",
                  "Rate", "Price", "Term", "Term (Months)", "Offer", "Offer Details",
                  "Terms of Service", "Early Termination Fee", "Monthly Fee"]:
            if c in df.columns:
                key_fields.append(c)
        def make_key(row):
            blob = "||".join(str(row.get(c,"")) for c in key_fields)
            return hashlib.sha1(blob.encode("utf-8")).hexdigest()
        df["OfferKey"] = df.apply(make_key, axis=1)
        keycol = "OfferKey"

    df["_Key"] = df[keycol]
    return df

def diff_frames(prev, curr):
    key = "_Key"
    prev = prev.set_index(key, drop=False)
    curr = curr.set_index(key, drop=False)

    added = curr.index.difference(prev.index)
    removed = prev.index.difference(curr.index)
    common = curr.index.intersection(prev.index)

    changed = []
    for k in common:
        rp, rc = prev.loc[k], curr.loc[k]
        diffs = {}
        for col in sorted(set(curr.columns).intersection(prev.columns)):
            if col.startswith("_"): continue
            vp, vc = ("" if pd.isna(rp[col]) else str(rp[col])), ("" if pd.isna(rc[col]) else str(rc[col]))
            if vp != vc:
                diffs[col] = {"before": vp, "after": vc}
        if diffs:
            changed.append({
                "key": k,
                "supplier": rc.get("Supplier") or rc.get("Company") or "",
                "changes": diffs
            })

    return {
        "added": curr.loc[added].to_dict(orient="records"),
        "removed": prev.loc[removed].to_dict(orient="records"),
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
        prev_files = sorted([f for f in os.listdir(OUTDIR) if f.startswith(prefix) and f.endswith(".csv") and today not in f])
        if prev_files:
            prev_df = pd.read_csv(os.path.join(OUTDIR, prev_files[-1]), dtype=str)
            report = diff_frames(prev_df, curr_df)
        else:
            report = {"added": curr_df.to_dict(orient="records"), "removed": [], "changed": []}
        reports.append((name, report))

    # Human-friendly text report
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
