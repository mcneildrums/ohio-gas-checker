#!/usr/bin/env python3
# apples_to_apples_diff.py
# Goal: Email *only* price/rate and term changes + list new/removed suppliers.
# Also writes a full summary for reference.

import csv, hashlib, io, os, sys, datetime, time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import urllib3

print("SCRIPT_VERSION: 2025-08-20-FOCUSED-DIFF")

# --- PAGES TO MONITOR (add more tuples if needed) ---
TARGETS = [
    ("Enbridge/Dominion - Residential",
     "https://www.energychoice.ohio.gov/ApplesToApplesComparision.aspx?Category=NaturalGas&RateCode=1&TerritoryId=1"),
]

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)
today = datetime.date.today().isoformat()

# Quiet warnings if we use verify=False as fallback
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# --- Column name variants seen on Apples-to-Apples CSVs ---
SUPPLIER_NAME_COLS = [
    "Supplier", "Company", "Company Name", "Supplier Name",
    "SupplierCompanyName", "CompanyName"
]

RATE_FIELDS = [
    "Rate", "Price", "PricePerMcf", "Price per Mcf", "Price per MCF",
    "MCF Price", "Rate ($/MCF)"
]

TERM_FIELDS = [
    "TermLength", "Term (Months)", "Term", "Contract Term", "Term Months"
]

# ------------ HTTP helpers with retries/timeouts ------------

def setup_session():
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(HEADERS)
    return s

def http_get(session, url, verify=True, connect_timeout=20, read_timeout=180):
    try:
        r = session.get(url, timeout=(connect_timeout, read_timeout), verify=verify)
        r.raise_for_status()
        return r
    except Exception:
        if verify:
            r = session.get(url, timeout=(connect_timeout, read_timeout), verify=False)
            r.raise_for_status()
            return r
        raise

def http_post(session, url, data, verify=True, connect_timeout=20, read_timeout=240):
    headers = {
        **HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.energychoice.ohio.gov",
        "Referer": url,
    }
    try:
        r = session.post(url, data=data, timeout=(connect_timeout, read_timeout), verify=verify, headers=headers)
        r.raise_for_status()
        return r
    except Exception:
        if verify:
            r = session.post(url, data=data, timeout=(connect_timeout, read_timeout), verify=False, headers=headers)
            r.raise_for_status()
            return r
        raise

# ----------------- Scrape CSV (direct or postback) -----------------

def extract_hidden_fields(soup):
    fields = {}
    for name in ["__EVENTTARGET", "__EVENTARGUMENT", "__VIEWSTATE", "__VIEWSTATEGENERATOR",
                 "__EVENTVALIDATION", "__VIEWSTATEENCRYPTED"]:
        tag = soup.find("input", {"name": name})
        if tag and tag.has_attr("value"):
            fields[name] = tag["value"]
    for inp in soup.find_all("input", {"type": "hidden"}):
        n = inp.get("name")
        if n and n not in fields:
            fields[n] = inp.get("value", "")
    return fields

def find_export_target(soup):
    # visible text first
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        href = (a.get("href") or "").strip()
        if "export all offers to csv" in text:
            if href.lower().startswith("javascript:__dopostback("):
                inside = href[href.index("(")+1:href.rindex(")")]
                evt_target = inside.split(",")[0].strip().strip("'").strip('"')
                return ("postback", evt_target)
            return ("direct", href)
    # fallbacks
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
    s = setup_session()
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
    if "__ASYNCPOST" not in fields:
        fields["__ASYNCPOST"] = "false"
    time.sleep(2)
    r2 = http_post(s, page_url, data=fields)

    ct = (r2.headers.get("Content-Type") or "").lower()
    if "text/csv" in ct or "application/vnd.ms-excel" in ct:
        return r2.content

    # try to discover CSV link from returned html
    try:
        soup2 = BeautifulSoup(r2.text, "html.parser")
        for a in soup2.find_all("a", href=True):
            if ".csv" in a["href"].lower():
                full = a["href"] if a["href"].startswith(("http://", "https://")) else urljoin(page_url, a["href"])
                r3 = http_get(s, full)
                return r3.content
    except Exception:
        pass

    # last resort: looks like CSV text but mislabeled
    if r2.text and ("," in r2.text or "\t" in r2.text) and "\n" in r2.text:
        return r2.content

    raise RuntimeError("Postback completed but CSV content was not found")

# ----------------- CSV → DataFrame with unique headers & keys -----------------

def df_from_csv_bytes(data_bytes):
    data = io.BytesIO(data_bytes)
    df = pd.read_csv(data, dtype=str)

    # Strip header whitespace
    df.columns = [c.strip() for c in df.columns]

    # Ensure column names are unique (Rate, Rate.2, ...)
    seen, unique_cols = {}, []
    for c in df.columns:
        if c not in seen:
            seen[c] = 1
            unique_cols.append(c)
        else:
            seen[c] += 1
            unique_cols.append(f"{c}.{seen[c]}")
    df.columns = unique_cols

    # Strip cell whitespace
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Build a stable key: prefer *Offer*ID; else hash key fields
    id_cols = [c for c in df.columns if ("offer" in c.lower() and "id" in c.lower())]
    if id_cols:
        keycol = id_cols[0]
    else:
        key_fields = []
        for c in SUPPLIER_NAME_COLS + RATE_FIELDS + TERM_FIELDS + ["Offer", "Offer Details"]:
            if c in df.columns:
                key_fields.append(c)
        def make_key(row):
            blob = "||".join(str(row.get(c, "")) for c in key_fields)
            return hashlib.sha1(blob.encode("utf-8")).hexdigest()
        df["OfferKey"] = df.apply(make_key, axis=1)
        keycol = "OfferKey"

    df["_Key"] = df[keycol]
    return df

# ----------------- Diff with rate/term focus & context -----------------

def _cell_to_str(v):
    if isinstance(v, pd.Series):
        v = v.iloc[0] if not v.empty else ""
    if pd.isna(v):
        return ""
    return str(v)

def _first_present_series(row_like, cols):
    for c in cols:
        if c in row_like.index:
            v = row_like[c]
            if isinstance(v, pd.Series):
                v = v.iloc[0] if not v.empty else ""
            if pd.isna(v):
                v = ""
            v = str(v).strip()
            if v:
                return v
    return ""

def _label_term(val): return (val or "").strip() or "n/a"
def _label_rate(val): return (val or "").strip() or "n/a"

def diff_frames(prev, curr):
    key = "_Key"
    prev = prev.set_index(key, drop=False)
    curr = curr.set_index(key, drop=False)

    added_keys = curr.index.difference(prev.index)
    removed_keys = prev.index.difference(curr.index)
    common_keys = curr.index.intersection(prev.index)

    changed = []
    common_cols = sorted(set(curr.columns).intersection(prev.columns))
    common_cols = [c for c in common_cols if not c.startswith("_")]

    for k in common_keys:
        rp, rc = prev.loc[k], curr.loc[k]
        diffs = {}
        for col in common_cols:
            vp = _cell_to_str(rp[col])
            vc = _cell_to_str(rc[col])
            if vp != vc:
                diffs[col] = {"before": vp, "after": vc}

        if diffs:
            supplier_now = _first_present_series(rc, SUPPLIER_NAME_COLS) or _first_present_series(rp, SUPPLIER_NAME_COLS)
            term_before = _first_present_series(rp, TERM_FIELDS)
            term_after  = _first_present_series(rc, TERM_FIELDS)
            rate_before = _first_present_series(rp, RATE_FIELDS)
            rate_after  = _first_present_series(rc, RATE_FIELDS)

            changed.append({
                "key": k,
                "supplier": supplier_now,
                "changes": diffs,
                "term_before": term_before,
                "term_after": term_after,
                "rate_before": rate_before,
                "rate_after": rate_after,
            })

    return {
        "added": curr.loc[added_keys].to_dict(orient="records"),
        "removed": prev.loc[removed_keys].to_dict(orient="records"),
        "changed": changed
    }

# ----------------- Build reports (full + focused changes) -----------------

def build_reports(name, report_dict):
    """
    Full report + a focused changes-only section that:
      - Shows only price/rate and term changes (with context)
      - Lists NEW suppliers with their rate/term
      - Lists REMOVED suppliers with their rate/term
    """
    full, chg = [], []

    full.append(f"=== {name} ===")
    full.append(f"New: {len(report_dict['added'])} | Removed: {len(report_dict['removed'])} | Changed: {len(report_dict['changed'])}")

    chg.append(f"=== Changes for {name} ===")

    # PRICE/TERM CHANGES
    interesting = []
    for ch in report_dict["changed"]:
        touched_rates = [f for f in ch["changes"].keys() if f in RATE_FIELDS]
        touched_terms = [f for f in ch["changes"].keys() if f in TERM_FIELDS]
        if touched_rates or touched_terms:
            interesting.append(ch)

    if interesting:
        chg.append("PRICE/TERM CHANGES:")
        chg.append("Company | Field | Old -> New (context)")
        chg.append("----------------------------------------")
        for ch_item in interesting:
            supplier = ch_item.get("supplier") or ch_item["key"]
            # rate changes
            for f in [f for f in RATE_FIELDS if f in ch_item["changes"]]:
                before = ch_item["changes"][f]["before"]
                after  = ch_item["changes"][f]["after"]
                term_ctx = ch_item.get("term_after") or ch_item.get("term_before")
                chg.append(f"{supplier} | Rate | {_label_rate(before)} -> {_label_rate(after)} (term: {_label_term(term_ctx)})")
            # term changes
            for f in [f for f in TERM_FIELDS if f in ch_item["changes"]]:
                before = ch_item["changes"][f]["before"]
                after  = ch_item["changes"][f]["after"]
                rate_ctx = ch_item.get("rate_after") or ch_item.get("rate_before")
                chg.append(f"{supplier} | Term | {_label_term(before)} -> {_label_term(after)} (rate: {_label_rate(rate_ctx)})")
    else:
        chg.append("No rate/term changes today.")

    # NEW SUPPLIERS
    if report_dict["added"]:
        chg.append("")
        chg.append("NEW SUPPLIERS:")
        for row in report_dict["added"]:
            supplier, rate, term = "", "", ""
            for c in SUPPLIER_NAME_COLS:
                if c in row and str(row[c]).strip():
                    supplier = str(row[c]).strip(); break
            for c in RATE_FIELDS:
                if c in row and str(row[c]).strip():
                    rate = str(row[c]).strip(); break
            for c in TERM_FIELDS:
                if c in row and str(row[c]).strip():
                    term = str(row[c]).strip(); break
            chg.append(f"{supplier or '(unknown)'} (Rate={_label_rate(rate)}, Term={_label_term(term)})")

    # REMOVED SUPPLIERS
    if report_dict["removed"]:
        chg.append("")
        chg.append("REMOVED SUPPLIERS:")
        for row in report_dict["removed"]:
            supplier, rate, term = "", "", ""
            for c in SUPPLIER_NAME_COLS:
                if c in row and str(row[c]).strip():
                    supplier = str(row[c]).strip(); break
            for c in RATE_FIELDS:
                if c in row and str(row[c]).strip():
                    rate = str(row[c]).strip(); break
            for c in TERM_FIELDS:
                if c in row and str(row[c]).strip():
                    term = str(row[c]).strip(); break
            chg.append(f"{supplier or '(unknown)'} (Rate={_label_rate(rate)}, Term={_label_term(term)})")

    # Full report extras (optional)
    if report_dict["changed"]:
        full.append("Changed details:")
        for ch_item in report_dict["changed"]:
            supplier = ch_item.get("supplier") or ch_item["key"]
            for col, vals in ch_item["changes"].items():
                full.append(f"- {supplier} | {col}: '{vals['before']}' -> '{vals['after']}'")
    else:
        full.append("No field-level changes today.")

    if report_dict["added"]:
        full.append(f"Added offers: {len(report_dict['added'])}")
    if report_dict["removed"]:
        full.append(f"Removed offers: {len(report_dict['removed'])}")

    full.append("")
    chg.append("")
    return "\n".join(full), "\n".join(chg)

# ----------------- Main -----------------

def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)

def main():
    reports_full, reports_changes = [], []

    for name, page in TARGETS:
        csv_bytes = fetch_csv_bytes_from_page(page)
        curr_df = df_from_csv_bytes(csv_bytes)

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

        full_text, chg_text = build_reports(name, report)
        reports_full.append(full_text)
        reports_changes.append(chg_text)

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reports_full))
    with open("changes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reports_changes))

    print("\n".join(reports_changes))

if __name__ == "__main__":
    sys.exit(main())
