#!/usr/bin/env python3
# apples_to_apples_diff.py
# Snapshot Ohio PUCO Apples-to-Apples CSV(s), compare to prior snapshot, and report field-level changes.
# Handles ASP.NET WebForms postback CSV export.

import csv, hashlib, io, os, sys, datetime, time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import urllib3

print("SCRIPT_VERSION: 2025-08-19-POSTBACK-SAFE")

# --- CONFIG: add the pages you want to monitor ---
TARGETS = [
    ("Enbridge/Dominion - Residential",
     "https://www.energychoice.ohio.gov/ApplesToApplesComparision.aspx?Category=NaturalGas&RateCode=1&TerritoryId=1"),
    # Add more if needed
]

OUTDIR = "data"  # where daily CSV snapshots are saved
os.makedirs(OUTDIR, exist_ok=True)
today = datetime.date.today().isoformat()

# Quiet warnings if we fall back to verify=False (public CSV is acceptable risk here).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36")
}

def safe_name(name: str) -> str:
    """Make a string safe for filenames."""
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)

def http_get(session, url, max_retries=3, backoff=2.0, verify=True):
    last_err = None
    for _ in range(max_retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=60, verify=verify)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    if verify:
        return http_get(session, url, max_retries=1, backoff=0.0, verify=False)
    raise last_err

def http_post(session, url, data, max_retries=3, backoff=2.0, verify=True):
    last_err = None
    for _ in range(max_retries):
        try:
            r = session.post(url, headers=HEADERS, data=data, timeout=60, verify=verify)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    if verify:
        return http_post(session, url, data, max_retries=1, backoff=0.0, verify=False)
    raise last_err

def extract_hidden_fields(soup):
    fields = {}
    for name in ["__EVENTTARGET", "__EVENTARGUMENT", "__VIEWSTATE", "__VIEWSTATEGENERATOR",
                 "__EVENTVALIDATION", "__VIEWSTATEENCRYPTED"]:
        tag = soup.find("input", {"name": name})
        if tag and tag.has_attr("value"):
            fields[name] = tag["value"]
    for inp in soup.find_all("input", {"type": "hidden"}):
        name = inp.get("name")
        if name and name not in fields:
            fields[name] = inp.get("value", "")
    return fields

def find_export_target(soup):
    # Try visible text first
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        href = (a.get("href") or "").strip()
        if "export all offers to csv" in text:
            if href.lower().startswith("javascript:__dopostback("):
                inside = href[href.index("(")+1:href.rindex(")")]
                evt_target = inside.split(",")[0].strip().strip("'").strip('"')
                return ("postback", evt_target)
            return ("direct", href)
    # Fallbacks
    for a in soup.find_all("a", href=True):
        if ".csv" in a["href"].lower():
            return ("direct", a["href"])
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().lower()
        if href.startswith("javascript:__dopostback("):
            txt = (a.get_text() or "").strip().lower()
            if "export" in txt:
                inside = a["href"][a["href"].index("(")+1:a["href"].rindex(")")]
                evt_target = inside.split(",")[0].strip().strip("'").strip('"')
                return ("postback", evt_target)
    raise RuntimeError("Could not find export link or postback target on the page.")

def fetch_csv_bytes_from_page(page_url):
    with requests.Session() as s:
        r = http_get(s, page_url)
        soup = BeautifulSoup(r.text, "html.parser")
        kind, payload = find_export_target(soup)

        if kind == "direct":
            href = payload
            full = href if href.startswith(("http://", "https://")) else urljoin(page_url, href)
            r2 = http_get(s, full)
            return r2.content

        # postback
        event_target = payload
        fields = extract_hidden_fields(soup)
        fields["__EVENTTARGET"] = event_target
        fields["__EVENTARGUMENT"] = fields.get("__EVENTARGUMENT", "")
        r2 = http_post(s, page_url, data=fields)

        ct = r2.headers.get("Content-Type", "").lower()
        if "text/csv" in ct or "application/vnd.ms-excel" in ct:
            return r2.content

        soup2 = BeautifulSoup(r2.text, "html.parser")
        for a in soup2.find_all("a", href=True):
            if ".csv" in a["href"].lower():
                full = a["href"] if a["href"].startswith(("http://", "https://")) else urljoin(page_url, a["href"])
                r3 = http_get(s, full)
                return r3.content

        if r2.text and ("," in r2.text or "\t" in r2.text) and "\n" in r2.text:
            return r2.content

        raise RuntimeError("Postback completed but CSV content was not found")

def df_from_csv_bytes(data_bytes):
    data = io.BytesIO(data_bytes)
    df = pd.read_csv(data, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # applymap warning -> column-wise strip
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

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
        csv_bytes = fetch_csv_bytes_from_page(page)
        curr_df = df_from_csv_bytes(csv_bytes)

        # Safe filename bits
        base = safe_name(name)
        snap_path = os.path.join(OUTDIR, f"{base}_{today}.csv".replace(" ", "_"))
        curr_df.to_csv(snap_path, index=False)

        prefix = f"{base}_".replace(" ", "_")
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
