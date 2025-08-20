#!/usr/bin/env python3
# apples_to_apples_diff.py
# Purpose: Email only price/rate and term changes + list truly new/removed offers,
# and keep keys STABLE so row/order changes don't create noise.

import hashlib, io, os, sys, datetime, time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import urllib3

print("SCRIPT_VERSION: 2025-08-20-STABLE-KEYS")

# --- Pages to monitor ---
TARGETS = [
    ("Enbridge/Dominion - Residential",
     "https://www.energychoice.ohio.gov/ApplesToApplesComparision.aspx?Category=NaturalGas&RateCode=1&TerritoryId=1"),
]

OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)
today = datetime.date.today().isoformat()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Column variants seen in CSVs
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

# "Stable" (non-volatile) fields we may use to define an offer identity.
# IMPORTANT: EXCLUDES any rate/term/marketing text so keys don't change when price/term change.
OFFER_ID_HINT_FIELDS = [
    "OfferId", "OfferID", "GasOfferId", "GasOfferID", "Offer Code", "OfferCode"
]
STABLE_ID_FIELDS = [
    # Supplier identity
    "SupplierCompanyName", "CompanyName", "Supplier", "Company", "Supplier Name", "Company Name",
    # Plan meta that tends to be stable for a given offer
    "RateType", "EarlyTerminationFee", "Monthly Fee", "MonthlyFee", "Cancellation Fee", "CancellationFee",
    # URLs often uniquely identify the exact plan SKU
    "TermsOfServiceURL", "SignUpNowURL",
]

# ---------- HTTP helpers with retries/timeouts ----------

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

# ---------- Scrape CSV via link or WebForms postback ----------

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
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        href = (a.get("href") or "").strip()
        if "export all offers to csv" in text:
            if href.lower().startswith("javascript:__dopostback("):
                inside = href[href.index("(")+1:href.rindex(")")]
                evt_target = inside.split(",")[0].strip().strip("'").strip('"')
                return ("postback", evt_target)
            return ("direct", href)
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

    # try finding a CSV link in the returned HTML
    try:
        soup2 = BeautifulSoup(r2.text, "html.parser")
        for a in soup2.find_all("a", href=True):
            if ".csv" in a["href"].lower():
                full = a["href"] if a["href"].startswith(("http://", "https://")) else urljoin(page_url, a["href"])
                r3 = http_get(s, full)
                return r3.content
    except Exception:
        pass

    # fallback: mislabeled CSV text
    if r2.text and ("," in r2.text or "\t" in r2.text) and "\n" in r2.text:
        return r2.content

    raise RuntimeError("Postback completed but CSV content was not found")

# ---------- CSV → DataFrame with unique headers & STABLE key ----------

def _uniquify_columns(df):
    df.columns = [c.strip() for c in df.columns]
    seen, uniq = {}, []
    for c in df.columns:
        if c not in seen:
            seen[c] = 1
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}.{seen[c]}")
    df.columns = uniq
    return df

def _first_present(row_like, cols):
    import pandas as pd
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

def _stable_offer_key(row):
    """
    Prefer a real OfferID-ish column. If absent, hash a tuple of stable fields.
    Critically, DO NOT include rate, term, or marketing text in the key.
    """
    # 1) Real ID columns if present
    for c in row.index:
        if any(k.lower() in c.lower() for k in OFFER_ID_HINT_FIELDS):
            v = str(row[c]).strip()
            if v:
                return f"ID::{v}"

    # 2) Build from stable fields (supplier + URLs + plan meta)
    parts = []
    for c in STABLE_ID_FIELDS:
        if c in row.index:
            v = str(row[c]).strip()
            parts.append(f"{c}={v}")

    # If we couldn't collect anything (extremely unlikely), fallback to supplier only:
    if not parts:
        supplier = _first_present(row, SUPPLIER_NAME_COLS)
        parts = [f"Supplier={supplier}"]

    blob = "||".join(parts)
    return "HK::" + hashlib.sha1(blob.encode("utf-8")).hexdigest()

def df_from_csv_bytes(data_bytes):
    data = io.BytesIO(data_bytes)
    df = pd.read_csv(data, dtype=str)
    df = _uniquify_columns(df)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Create stable key per row
    df["_Key"] = df.apply(_stable_offer_key, axis=1)

    # Normalize a couple of convenience columns for later display/grouping
    df["_Supplier"] = df.apply(lambda r: _first_present(r, SUPPLIER_NAME_COLS), axis=1)
    df["_Rate"]     = df.apply(lambda r: _first_present(r, RATE_FIELDS), axis=1)
    df["_Term"]     = df.apply(lambda r: _first_present(r, TERM_FIELDS), axis=1)
    return df

# ---------- Diff (now based on STABLE offer key) ----------

def _cell_to_str(v):
    import pandas as pd
    if isinstance(v, pd.Series):
        v = v.iloc[0] if not v.empty else ""
    if pd.isna(v):
        return ""
    return str(v)

def diff_frames(prev, curr):
    key = "_Key"
    prev = prev.set_index(key, drop=False)
    curr = curr.set_index(key, drop=False)

    added_keys = curr.index.difference(prev.index)
    removed_keys = prev.index.difference(curr.index)
    common_keys = curr.index.intersection(prev.index)

    changed = []
    # Compare over the intersection of columns, skip internal (_*)
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
            changed.append({
                "key": k,
                "supplier": rc.get("_Supplier") or rp.get("_Supplier") or "",
                "rate_before": rp.get("_Rate", ""),
                "rate_after":  rc.get("_Rate", ""),
                "term_before": rp.get("_Term", ""),
                "term_after":  rc.get("_Term", ""),
                "changes": diffs,
            })

    return {
        "added": curr.loc[added_keys].to_dict(orient="records"),
        "removed": prev.loc[removed_keys].to_dict(orient="records"),
        "changed": changed
    }

# ---------- Build reports (focused + full) ----------

def _label(v, na="n/a"):
    v = (v or "").strip()
    return v if v else na

def build_reports(name, report_dict):
    full, chg = [], []
    full.append(f"=== {name} ===")
    full.append(f"New: {len(report_dict['added'])} | Removed: {len(report_dict['removed'])} | Changed: {len(report_dict['changed'])}")

    chg.append(f"=== Changes for {name} ===")

    # PRICE/TERM CHANGES (only)
    interesting = []
    for ch in report_dict["changed"]:
        touched = set(ch["changes"].keys())
        rate_touched = any(f in touched for f in RATE_FIELDS)
        term_touched = any(f in touched for f in TERM_FIELDS)
        if rate_touched or term_touched:
            interesting.append(ch)

    if interesting:
        chg.append("PRICE/TERM CHANGES:")
        chg.append("Company | Field | Old -> New (context)")
        chg.append("----------------------------------------")
        for ch_item in interesting:
            supplier = ch_item.get("supplier") or ch_item["key"]
            # Rate changes
            if ch_item["rate_before"] != ch_item["rate_after"]:
                chg.append(f"{supplier} | Rate | {_label(ch_item['rate_before'])} -> {_label(ch_item['rate_after'])} (term: {_label(ch_item['term_after'] or ch_item['term_before'])})")
            # Term changes
            if ch_item["term_before"] != ch_item["term_after"]:
                chg.append(f"{supplier} | Term | {_label(ch_item['term_before'])} -> {_label(ch_item['term_after'])} (rate: {_label(ch_item['rate_after'] or ch_item['rate_before'])})")
    else:
        chg.append("No rate/term changes today.")

    # NEW OFFERS (deduped by supplier|term|rate for readability)
    if report_dict["added"]:
        chg.append("")
        chg.append("NEW SUPPLIERS:")
        seen = set()
        for row in report_dict["added"]:
            supplier = _label(row.get("_Supplier"))
            rate = _label(row.get("_Rate"))
            term = _label(row.get("_Term"))
            key = (supplier, rate, term)
            if key in seen:
                continue
            seen.add(key)
            chg.append(f"{supplier} (Rate={rate}, Term={term})")

    # REMOVED OFFERS (deduped)
    if report_dict["removed"]:
        chg.append("")
        chg.append("REMOVED SUPPLIERS:")
        seen = set()
        for row in report_dict["removed"]:
            supplier = _label(row.get("_Supplier"))
            rate = _label(row.get("_Rate"))
            term = _label(row.get("_Term"))
            key = (supplier, rate, term)
            if key in seen:
                continue
            seen.add(key)
            chg.append(f"{supplier} (Rate={rate}, Term={term})")

    # Optional: brief full change dump (can help debug)
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

# ---------- Main ----------

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
