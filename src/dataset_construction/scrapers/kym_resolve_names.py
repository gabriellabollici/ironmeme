"""Resolve templates_to_edit names to KYM entries and write the updated benchmark.

For every benchmark row whose ``templates_to_edit`` cell is non-empty:

1. Replace ``template_name`` with that new name.
2. If the new name already appears in the dataset's ``template_name`` column,
   reuse that row's description.
3. Otherwise resolve the name against Know Your Meme:
   - exact (normalized) title match against the offline ``kym.json`` dump, else
   - a live KYM search for the exact name, then the entry's About section.
4. If KYM has no entry for the exact name, leave ``template_name`` and
   ``description`` blank.

Rows whose ``templates_to_edit`` is ``-remove this one-`` or ``Not a template``
are dropped from the dataset entirely.

This was a one-off curation step: it reads a hand-edited working sheet with a
``templates_to_edit`` column (``/tmp/ironmeme_working.json``, not part of the
repo), and its result is already in ``benchmark/sources/templates.csv`` (and
the two dropped memes in ``build_benchmark.EXCLUDED_IDS``). Kept as a helper
for resolving new names against KYM.
"""

import json
import random
import re
import sys
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

from dataset_construction.scrapers.common import HEADERS
from paths import DATASETS_DIR

KYM_SOURCE_PATH = DATASETS_DIR / "memes/template_sources/kym.json"

MIN_DELAY = 6
MAX_DELAY = 9

REMOVE_MARKERS = {"-remove this one-", "not a template"}


def normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_kym_entries():
    """Return normalized title -> dict(title, url, about) and slug -> dict."""
    with open(KYM_SOURCE_PATH, encoding="utf-8") as f:
        records = json.load(f)
    by_title, by_slug = {}, {}
    for rec in records:
        slug = rec.get("url", "").rstrip("/").rsplit("/", 1)[-1].lower()
        title = (rec.get("title") or "").strip()
        about = ((rec.get("content", {}) or {}).get("about", {}) or {}).get("text", [])
        entry = {
            "title": title,
            "url": rec.get("url", ""),
            "slug": slug,
            "about": about[0] if about else None,
        }
        if title:
            by_title.setdefault(normalize(title), entry)
        if slug:
            by_slug.setdefault(slug, entry)
    return by_title, by_slug


def clean_query(query):
    """Make a user-supplied name searchable on KYM: straighten apostrophes."""
    return (
        query.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def search_kym(session, query):
    response = session.get(
        "https://knowyourmeme.com/search",
        params={"q": clean_query(query)},
        timeout=25,
    )
    if response.status_code != 200:
        return []
    # KYM redirects straight to the entry when there is a single strong match.
    match = re.search(r"/memes/([a-z0-9-]+)$", response.url)
    if match:
        body = BeautifulSoup(response.text, "html.parser")
        title = body.select_one("h1") or body.select_one("title")
        return [("/memes/" + match.group(1), title.get_text(strip=True) if title else "")]
    body = BeautifulSoup(response.text, "html.parser")
    results = []
    seen = set()
    for link in body.select("article a[href]"):
        href = link.get("href", "")
        if not re.match(r"^/memes/[a-z0-9-]+$", href):
            continue
        title = link.get_text(" ", strip=True).replace("\u2605", "").strip()
        if title and href not in seen:
            seen.add(href)
            results.append((href, title))
    return results


def similarity(query, title):
    a, b = set(normalize(query).split()), set(normalize(title).split())
    jaccard = len(a & b) / max(1, len(a | b))
    edit = __import__("difflib").SequenceMatcher(
        None, normalize(query), normalize(title)
    ).ratio()
    return max(jaccard, edit)


def is_good_match(query, title):
    tokens = set(normalize(query).split())
    title_tokens = set(normalize(title).split())
    all_present = bool(tokens) and tokens <= title_tokens
    jaccard = len(tokens & title_tokens) / max(1, len(tokens | title_tokens))
    edit = __import__("difflib").SequenceMatcher(
        None, normalize(query), normalize(title)
    ).ratio()
    if edit >= 0.55:
        return True
    if jaccard >= 0.6:
        return True
    if all_present and jaccard >= 0.35:
        return True
    return False


def fetch_about(session, entry_url):
    """Fetch an entry page and return the first substantive About paragraph."""
    response = session.get(entry_url, timeout=25)
    if response.status_code != 200:
        return None
    body = BeautifulSoup(response.text, "html.parser")
    container = body.select_one(".bodycopy")
    if container is None:
        return None
    for paragraph in container.find_all("p"):
        text = paragraph.get_text(" ", strip=True)
        if len(text) > 60:
            return text
    return None


def resolve_live(session, name):
    """Resolve a name against live KYM search; return (about_text, entry_url)."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    results = search_kym(session, name)
    if not results:
        return None, None
    key = normalize(name)
    exact = [r for r in results if normalize(r[1]) == key]
    if exact:
        best = exact[0]
    else:
        best = max(results, key=lambda r: similarity(name, r[1]))
        if not is_good_match(name, best[1]):
            return None, None
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    about = fetch_about(session, "https://knowyourmeme.com" + best[0])
    if not about:
        return None, best[0]
    return about, best[0]


def resolve_name(session, name, by_title, log, cache, sources):
    """Return (template_name_or_None, description_or_None)."""
    key = normalize(name)
    if key in cache:
        return cache[key]
    entry = by_title.get(key)
    if entry and entry["about"]:
        cache[key] = (name, entry["about"])
        sources[key] = f"offline kym.json: {entry['title']}"
        return cache[key]
    if session is None:
        cache[key] = (None, None)
        return cache[key]
    about, url = resolve_live(session, name)
    if about is None:
        log.append(name)
        sources[key] = "not found"
        cache[key] = (None, None)
        return cache[key]
    cache[key] = (name, about)
    sources[key] = f"live KYM: {url}"
    return cache[key]


def main(input_path=None, output_path="/tmp/ironmeme_edited.csv"):
    df = pd.read_json(input_path or "/tmp/ironmeme_working.json")

    orig_lower = {}
    for name, desc in df.loc[
        df["template_name"].notna() & (df["template_name"] != ""),
        ["template_name", "description"],
    ].itertuples(index=False):
        key = normalize(str(name))
        if key not in orig_lower:
            orig_lower[key] = (str(name), str(desc) if pd.notna(desc) else "")

    df = df[~df["templates_to_edit"].str.strip().str.lower().isin(REMOVE_MARKERS)]

    by_title, _by_slug = load_kym_entries()
    session = requests.Session()
    session.headers.update(HEADERS)

    not_found = []
    cache = {}
    sources = {}
    matched_existing = 0
    matched_offline = 0
    matched_live = 0
    blanks = 0

    for idx, row in df.iterrows():
        raw_edit = row["templates_to_edit"]
        edit = "" if pd.isna(raw_edit) else str(raw_edit).strip()
        if not edit:
            continue
        key = normalize(edit)
        if key in orig_lower:
            _, desc = orig_lower[key]
            df.at[idx, "template_name"] = edit
            df.at[idx, "description"] = desc
            matched_existing += 1
            continue
        name, about = resolve_name(session, edit, by_title, not_found, cache, sources)
        if name is None:
            df.at[idx, "template_name"] = ""
            df.at[idx, "description"] = ""
            blanks += 1
        else:
            df.at[idx, "template_name"] = name
            df.at[idx, "description"] = about or ""
            if key in by_title and by_title[key]["about"]:
                matched_offline += 1
            else:
                matched_live += 1
        print(
            f"{edit!r} -> {df.at[idx, 'template_name']!r} "
            f"[{df.at[idx, 'description'][:60]!r}...]"
        )
        sys.stdout.flush()

    print()
    print(f"matched existing: {matched_existing}")
    print(f"matched offline: {matched_offline}")
    print(f"matched live: {matched_live}")
    print(f"not found (left blank): {blanks}")
    print("NOT FOUND NAMES:")
    for n in not_found:
        print("  ", repr(n))

    review = {k: v for k, v in sources.items()}
    with open("/tmp/kym_resolve_review.json", "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    df.to_csv(output_path, index=False)
    print(f"written: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
