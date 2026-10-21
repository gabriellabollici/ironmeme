"""Label inventories and the single invalid-label rule used by all scoring code.

Every predicted (and ground-truth) label for the three tasks goes through the
same rule before it is scored:

  1. surface normalisation -- lowercase; drop a trailing parenthetical copied
     from the prompt (e.g. "Victim (Inferior/Positive)" -> "victim"); remove
     whitespace, hyphens and underscores (so "in-group", "IN_GROUP" and
     "in group" all become "ingroup"); accept a simple plural/singular
     variant ("villains" -> "villain", "citizen" -> "Citizens");
  2. exact lookup in the task's fixed inventory (plus the surface-variant
     aliases below);
  3. anything that does not match is discarded (canon_* returns None).

No semantic mapping happens here (e.g. "communists" is NOT mapped to
"Left wing"). An optional alias file (labels/aliases.yaml) can add such
mappings for a separate *lenient* score -- see set_aliases(); it is never
active unless a script is explicitly run with --aliases.
"""
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

RESOURCES_DIR = Path(__file__).resolve().parent
GT_VIEWPOINT_FILLS_CSV = RESOURCES_DIR / "gt_viewpoint_fills.csv"

# ---------------------------------------------------------------------------
# Inventories
# ---------------------------------------------------------------------------
APPROVED_IDENTITIES: List[str] = [
    "Women", "Men", "White people", "Feminist", "Anti feminist",
    "LGTBQ+ rights supporter", "Anti LGTBQ+", "Misogynist/male supremacist",
    "Sex worker", "Racist", "Anti racist", "people of color",
    "National identity", "Nationalist/regionalist", "Progressive", "Conservative",
    "Left wing", "Right wing", "Liberal", "Authoritarian", "Republican",
    "Democrat", "Anticapitalist", "Corporativist/capitalist", "Voter",
    "Abstentionist", "Politicians/government", "World leaders/elites",
    "Media elites", "Celebrity", "Green party supporter", "Libertarian",
    "Politically disillusioned", "Conspiracy theorist", "Climate activist",
    "Climate change skeptic/denier", "Climate disengaged", "Plant based/vegan",
    "Clean energy advocate", "Nuclear energy advocate", "Citizens",
]
VIEWPOINTS: List[str] = ["ingroup", "outgroup"]
ROLES: List[str] = [
    "hero", "villain", "victim", "sage", "fool",
    "charmer", "monster", "winner", "loser",
]

# Surface-variant spellings only (the taxonomy's own "LGTBQ+" is a typo of
# the common "LGBTQ+"; a model writing the common spelling means the same label).
IDENTITY_ALIASES: Dict[str, str] = {
    "LGBTQ+ rights supporter": "LGTBQ+ rights supporter",
    "Anti LGBTQ+": "Anti LGTBQ+",
}

TASKS = ("identity", "viewpoint", "role")

_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")
_SEPARATORS_RE = re.compile(r"[\s\-_]+")


def surface_key(raw: Any) -> str:
    """Case/whitespace/hyphen/underscore-insensitive key for a label."""
    s = _TRAILING_PAREN_RE.sub("", str(raw).strip().lower())
    return _SEPARATORS_RE.sub("", s)


def _build_table(labels: List[str], aliases: Dict[str, str]) -> Dict[str, str]:
    table = {surface_key(lab): lab for lab in labels}
    for src, dst in aliases.items():
        table[surface_key(src)] = dst
    # Fail loudly if a plural/singular variant of one label could resolve to
    # another label (it would make the lookup order-dependent).
    for key, lab in table.items():
        for variant in (key + "s", key[:-1] if key.endswith("s") else None):
            if variant and variant in table and table[variant] != lab:
                raise ValueError(f"Ambiguous plural variant: {key!r} vs {variant!r}")
    return table


_TABLES: Dict[str, Dict[str, str]] = {
    "identity": _build_table(APPROVED_IDENTITIES, IDENTITY_ALIASES),
    "viewpoint": _build_table(VIEWPOINTS, {}),
    "role": _build_table(ROLES, {}),
}

# Lenient-score aliases (task -> {surface_key: canonical label}); empty unless
# set_aliases() is called.
_ALIASES: Dict[str, Dict[str, str]] = {t: {} for t in TASKS}


def _lookup(key: str, table: Dict[str, str]) -> Optional[str]:
    candidates = [key]
    if key.endswith("es"):
        candidates.append(key[:-2])
    if key.endswith("s"):
        candidates.append(key[:-1])
    candidates.append(key + "s")
    for k in candidates:
        if k and k in table:
            return table[k]
    return None


def canon(task: str, raw: Any) -> Optional[str]:
    """Map *raw* to its inventory label for *task*, or None if invalid."""
    if raw is None:
        return None
    key = surface_key(raw)
    if not key:
        return None
    if key in _ALIASES[task]:
        return _ALIASES[task][key]
    return _lookup(key, _TABLES[task])


def canon_identity(raw: Any) -> Optional[str]:
    return canon("identity", raw)


def canon_viewpoint(raw: Any) -> Optional[str]:
    return canon("viewpoint", raw)


def canon_role(raw: Any) -> Optional[str]:
    return canon("role", raw)


def inventory(task: str) -> List[str]:
    """Lowercased label inventory, as used for the metric classes."""
    return {
        "identity": [x.lower() for x in APPROVED_IDENTITIES],
        "viewpoint": list(VIEWPOINTS),
        "role": list(ROLES),
    }[task]


# ---------------------------------------------------------------------------
# Lenient aliases (labels/aliases.yaml)
# ---------------------------------------------------------------------------
def load_aliases(path: Path) -> Dict[str, Dict[str, str]]:
    """Read an aliases.yaml file: {identities|viewpoints|roles: {raw: label}}."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    section = {"identity": "identities", "viewpoint": "viewpoints", "role": "roles"}
    out: Dict[str, Dict[str, str]] = {t: {} for t in TASKS}
    for task, name in section.items():
        for src, dst in (data.get(name) or {}).items():
            target = _lookup(surface_key(dst), _TABLES[task])
            if target is None:
                raise ValueError(f"{path}: alias target {dst!r} is not a valid {task} label")
            out[task][surface_key(src)] = target
    return out


def set_aliases(aliases: Optional[Dict[str, Dict[str, str]]]) -> None:
    """Activate lenient aliases for this process (None/{} resets to strict)."""
    for task in TASKS:
        _ALIASES[task] = dict((aliases or {}).get(task, {}))


# ---------------------------------------------------------------------------
# Blank ground-truth viewpoints
# ---------------------------------------------------------------------------
_gt_fills: Optional[Dict[str, Dict[int, str]]] = None


def gt_viewpoint_fills() -> Dict[str, Dict[int, str]]:
    """{image_filename: {position: viewpoint}} for blank GT viewpoints that
    another annotator labeled (see resolve_blank_gt_viewpoints.py)."""
    global _gt_fills
    if _gt_fills is None:
        _gt_fills = {}
        if GT_VIEWPOINT_FILLS_CSV.is_file():
            with open(GT_VIEWPOINT_FILLS_CSV, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    vp = canon_viewpoint(row["filled_viewpoint"])
                    if vp:
                        _gt_fills.setdefault(row["image_filename"], {})[int(row["position"])] = vp
    return _gt_fills


def gt_viewpoint_at(image_filename: str, position: int, raw: Any) -> Optional[str]:
    """Canonical GT viewpoint at *position*, filling a blank from the
    annotator-derived fills when one exists; None if still invalid/blank."""
    vp = canon_viewpoint(raw)
    if vp is None:
        vp = gt_viewpoint_fills().get(image_filename, {}).get(position)
    return vp


def coerce_to_list(val: Any) -> Any:
    """Parse a stringified Python/JSON list into an actual list, if needed."""
    if isinstance(val, str):
        val = val.strip()
        if not val or val == "[]":
            return []
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
    return val
