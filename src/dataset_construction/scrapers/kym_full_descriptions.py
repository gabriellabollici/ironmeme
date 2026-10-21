"""Complete truncated template descriptions with the full KYM About text.

KYM's ``meta description`` is truncated at ~155-160 characters, which is what
``template_descriptions.json`` (and so a freshly matched template) carries.
The full summary lives in the entry's About section (``content.about.text`` in
the KYM dump, or the first ``.bodycopy`` paragraph on the live site).

For every template in ``benchmark/sources/templates.csv`` whose description
lacks a sentence-final punctuation mark (i.e. it stops mid-sentence), this
script replaces it with:

1. the full About paragraph from ``kym.json`` (offline), if complete, else
2. the first paragraph of the live entry's ``.bodycopy`` (fetched from KYM).

Only rows with ``curated`` other than ``yes`` (suggestions added by
match_benchmark_templates) are touched; hand-curated rows and
``PRESERVED_NAMES`` are never changed.
"""

import difflib
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from dataset_construction.scrapers.common import HEADERS
from paths import DATASETS_DIR, TEMPLATES_TABLE

TEMPLATES_PATH = DATASETS_DIR / "embedding_datasets/templates.json"
KYM_SOURCE_PATH = DATASETS_DIR / "memes/template_sources/kym.json"
OUTPUT_PATH = DATASETS_DIR / "memes/template_sources/template_descriptions.json"
LOG_FILE = DATASETS_DIR / "memes/template_sources/template_descriptions_log.txt"

MIN_DELAY = 6
MAX_DELAY = 9

# Deliberately set descriptions from our manual curation; never overwrite these.
# Minimum similarity for trusting a KYM search result (below this, the template
# is logged for manual review instead of being auto-assigned).
SEARCH_MATCH_THRESHOLD = 0.55

# Curated slug overrides for templates whose KYM entry search can't find
# (verified manually against knowyourmeme.com).
CURATED_SLUGS = {
    "Expanding Brain": "galaxy-brain",
    "Star Wars Yoda": "yoda",
    "Grus Plan": "grus-plan",
    "Hard To Swallow Pills": "hard-to-swallow-pills",
    "Mr Krabs Blur Meme": "mr-krabs",
    "Yall Got Any More Of That": "yall-got-anymore-of",
    "Blank Nut Button": "nut-button",
    "Jack Sparrow Being Chased": "jack-sparrow",
    "Success Kid Original": "success-kid-i-hate-sandcastles",
    "Chef Gordon Ramsay": "gordon-ramsay",
    "Angry Chef Gordon Ramsay": "gordon-ramsay",
    "UNO Draw 25 Cards": "uno-reverse-card",
    "Sad Pablo Escobar": "pablo-escobar-waiting",
    "Interupting Kanye": "kanye-interrupts",
}

PRESERVED_NAMES = {
    "My Parents Are Dead / Batman Slapping Robin",
    "We Need To Go Deeper",
    "Jack Sparrow Run",
    "Correction Guy",
    "Steven Crowder's \"Change My Mind\" Campus Sign",
    "Friend Zone Fiona",
    "Hard To Swallow Pills",
    "Me And The Boys",
}

TERMINAL = (".", "!", "?", "\u201d", "\u2019", "\u00bb", '"')



def normalize(text):
    """Lowercase, de-accent and collapse any non-alphanumeric runs to spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def similarity(query, title):
    """Max of token Jaccard and normalized edit similarity (0..1)."""
    a, b = set(normalize(query).split()), set(normalize(title).split())
    jaccard = len(a & b) / max(1, len(a | b))
    edit = difflib.SequenceMatcher(None, normalize(query), normalize(title)).ratio()
    return max(jaccard, edit)


def is_good_match(query, title):
    """Whether a search result title reliably corresponds to the query.

    Accepts exact/near-exact matches (high edit similarity or high token
    overlap), plus subset matches where every query token appears in the
    title (e.g. "success kid" -> "Success Kid / I Hate Sandcastles"). The
    all-tokens-present rule guards against unrelated results sharing a
    single keyword.
    """
    tokens = set(normalize(query).split())
    title_tokens = set(normalize(title).split())
    all_present = bool(tokens) and tokens <= title_tokens
    jaccard = len(tokens & title_tokens) / max(1, len(tokens | title_tokens))
    edit = difflib.SequenceMatcher(None, normalize(query), normalize(title)).ratio()
    if edit >= SEARCH_MATCH_THRESHOLD:
        return True
    if jaccard >= 0.6:
        return True
    if all_present and jaccard >= 0.35:
        return True
    return False


def search_kym(session, query):
    """Return (href, title) pairs from the KYM search results page."""
    response = session.get(
        "https://knowyourmeme.com/search", params={"q": query}, timeout=25
    )
    if response.status_code != 200:
        return []
    body = BeautifulSoup(response.text, "html.parser")
    h1 = next(
        (h for h in body.find_all("h1") if "Search Results" in h.get_text()), None
    )
    if h1 is None:
        return []
    node = h1.parent
    container = None
    for _ in range(6):
        node = node.find_next_sibling()
        if node is None:
            break
        if node.name in ("div", "section") and len(node.select('a[href*="/memes/"]')) > 3:
            container = node
            break
    if container is None:
        return []
    seen = set()
    results = []
    for link in container.select("a[href]"):
        href = link.get("href", "")
        if not re.match(r"^/memes/[a-z0-9-]+$", href):
            continue
        title = link.get_text(" ", strip=True).replace("★", "").strip()
        if title and href not in seen:
            seen.add(href)
            results.append((href, title))
    return results

def _log(message):
    print(message, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def load_kym_full_text():
    """Return slug -> full About paragraph and exact title -> About paragraph."""
    with open(KYM_SOURCE_PATH, encoding="utf-8") as f:
        records = json.load(f)
    by_slug, by_title, title_slug = {}, {}, {}
    for rec in records:
        slug = rec.get("url", "").rstrip("/").rsplit("/", 1)[-1].lower()
        about = (rec.get("content", {}).get("about", {}) or {}).get("text", [])
        title = (rec.get("title") or "").strip().lower()
        if title:
            title_slug.setdefault(title, slug)
            by_title.setdefault(title, about[0] if about else None)
        if about:
            by_slug.setdefault(slug, about[0])
    return by_slug, by_title, title_slug


def load_name_to_slug():
    """Parse template name -> KYM slug from the fetch log (empty if there is none)."""
    if not LOG_FILE.exists():
        return {}
    log = LOG_FILE.read_text(encoding="utf-8")
    mapping = {}
    for line in log.splitlines():
        m = re.match(r".*\[\d+/\d+\] '([^']*)' -> (.*)", line)
        if not m:
            continue
        sm = re.search(r"/memes/([a-z0-9-]+)", m.group(2))
        if sm:
            mapping[m.group(1)] = sm.group(1)
    return mapping


def fetch_bodycopy(session, entry_url):
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


def find_flagged_names():
    """Unique template names, among rows not yet curated, whose description stops mid-sentence."""
    df = pd.read_csv(TEMPLATES_TABLE, dtype=str, keep_default_na=False)
    uncurated = df[df["curated"] != "yes"]
    names = []
    for name, desc in uncurated.groupby("template_name")["template_description"].first().items():
        if pd.isna(desc) or not str(desc).strip():
            continue
        if name in PRESERVED_NAMES:
            continue
        if not str(desc).rstrip().endswith(TERMINAL):
            names.append(name)
    return df, names


def main():
    df, names = find_flagged_names()
    if not names:
        print(f"No uncurated template description in {TEMPLATES_TABLE} needs completing.")
        return
    by_slug, by_title, title_slug = load_kym_full_text()
    name_to_slug = load_name_to_slug()
    descriptions = {}
    if OUTPUT_PATH.exists():
        descriptions = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

    templates = pd.read_json(TEMPLATES_PATH)
    name_to_filename = {}
    for _, row in templates.iterrows():
        stem = Path(row["filenames"]).stem.replace("-", " ").replace("_", " ").strip()
        name_to_filename.setdefault(stem.lower(), row["filenames"])

    session = requests.Session()
    session.headers.update(HEADERS)

    def find_slug(name):
        slug = (
            name_to_slug.get(name)
            or CURATED_SLUGS.get(name)
            or title_slug.get(name.strip().lower())
        )
        if slug:
            return slug.removeprefix("/memes/")
        return None

    def live_bodycopy(name, slug):
        for _ in range(3):
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            body = fetch_bodycopy(session, f"https://knowyourmeme.com/memes/{slug}")
            if body:
                return body
        return None

    _log(f"Completing descriptions for {len(names)} template(s)")
    fixed, skipped = 0, []
    for i, name in enumerate(names, start=1):
        editable = (df["template_name"] == name) & (df["curated"] != "yes")
        current = str(df.loc[editable, "template_description"].iloc[0])
        slug = find_slug(name)
        full = by_slug.get(slug) if slug else None
        source = "kym.json about"
        if full is None:
            full = by_title.get(name.strip().lower())
        if full is None:
            if slug is None:
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                results = search_kym(session, name)
                if results:
                    best = max(results, key=lambda r: similarity(name, r[1]))
                    if is_good_match(name, best[1]):
                        slug = best[0].removeprefix("/memes/")
            if slug:
                full = live_bodycopy(name, slug)
                source = f"live bodycopy (/memes/{slug})"

        if (
            full
            and full != current
            and (len(full) > len(current) or full.rstrip().endswith(TERMINAL))
        ):
            filename = name_to_filename.get(name.strip().lower())
            if filename:
                descriptions[filename] = full
                OUTPUT_PATH.write_text(
                    json.dumps(descriptions, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            df.loc[editable, "template_description"] = full
            df.to_csv(TEMPLATES_TABLE, index=False)
            _log(f"[{i}/{len(names)}] {name!r} -> {source}")
            fixed += 1
        else:
            skipped.append(name)
            _log(f"[{i}/{len(names)}] {name!r} -> no full text found")
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    _log(f"Done. completed {fixed}, skipped {len(skipped)}")
    for s in skipped:
        _log(f"  skipped: {s}")


if __name__ == "__main__":
    main()
