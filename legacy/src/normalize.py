"""Text normalization v1. Language-agnostic (no per-country branches).

Pipeline: NFKD unicode -> strip accents/combining marks -> casefold ->
'&' -> 'and' -> non-alphanumeric to space -> collapse whitespace ->
token-level abbreviation expansion from *_ABBREV dicts below.

Dicts derived from TRAIN fit-part + full-pool token frequencies, evidence in
artifacts/metrics/normdict_evidence.json and artifacts/metrics/eda_struct.json.
Ambiguous tokens (pa=Pennsylvania, in=Indiana, co in addresses, lp/pc/pllc)
are deliberately NOT expanded. See reports/assumptions.md.

Bump NORMALIZE_VERSION and re-run dependents on any change.
"""

import re
import unicodedata

NORMALIZE_VERSION = 1

_COMMON = {
    "and": "and",  # '&' is pre-replaced with 'and'
}

# name-side only (legal forms, honorifics, trade words)
NAME_ABBREV = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "svc": "services",
    "assoc": "associates",
    "bros": "brothers",
    "shri": "sri",
    "shree": "sri",
}

# address-side only (street types, unit types)
ADDR_ABBREV = {
    "rd": "road",
    "st": "street",  # NOTE: 'saint' conflated; consistent both sides (see assumptions)
    "ave": "avenue",
    "avenu": "avenue",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "blvd": "boulevard",
    "ste": "suite",
    "apt": "apartment",
    "hwy": "highway",
    "hway": "highway",
    "pkwy": "parkway",  # 'parkway' full form not counted; standard mapping, see assumptions
}

# trailing legal-suffix tokens (analysis helper only, NOT applied in normalize)
SUFFIX_TOKENS = frozenset([
    "llc", "incorporated", "inc", "corporation", "corp", "limited", "ltd",
    "private", "pvt", "company", "co", "llp", "pllc", "lp", "pc", "pa",
])

_PUNCT_RE = re.compile(r"[^0-9a-z\s]")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _base(text):
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.casefold()
    t = t.replace("&", " and ")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def normalize(text, kind="name"):
    """kind in {'name','addr'}; returns normalized string."""
    t = _base(text)
    if not t:
        return ""
    table = NAME_ABBREV if kind == "name" else ADDR_ABBREV
    toks = t.split(" ")
    return " ".join(table.get(w, w) for w in toks)


def normalize_name(text):
    return normalize(text, "name")


def normalize_address(text):
    return normalize(text, "addr")


def strip_suffix_tokens(normed_name):
    """Remove trailing legal-suffix tokens (for legal-suffix-gap analysis)."""
    toks = normed_name.split(" ")
    while toks and toks[-1] in SUFFIX_TOKENS:
        toks.pop()
    return " ".join(toks)


def postal_tokens(normed_address):
    """Digit-only tokens of length 4-6 (US ZIP / India PIN / France code)."""
    return [t for t in normed_address.split(" ") if t.isdigit() and 4 <= len(t) <= 6]


def tokens(text):
    return _TOKEN_RE.findall(text.lower())
