"""Offline romanization of Indic (and other alphabetic) scripts using ONLY the Python
stdlib Unicode character database (unicodedata.name). No external tables, models or
data: e.g. 'DEVANAGARI LETTER KA' -> 'k' + inherent 'a'; 'DEVANAGARI VOWEL SIGN II'
replaces the inherent vowel with 'i'; VIRAMA drops it; word-final inherent 'a' is
dropped (schwa deletion). Output is coarse and meant for BLOCKING keys (tokens,
trigrams, phonetic codes), then compared against normalized Latin text.
Experimental (bucketing diagnostic); not yet used by the pipeline.
"""

import unicodedata

_INDIC = {"DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU",
          "KANNADA", "MALAYALAM", "SINHALA"}
_SIMPLE = [("aa", "a"), ("ii", "i"), ("uu", "u"), ("ee", "e"), ("oo", "o"),
           ("tt", "t"), ("dd", "d"), ("nn", "n"), ("ss", "sh"), ("ll", "l"), ("rr", "r"),
           ("ny", "n"), ("ng", "n"), ("zh", "l")]
_VOWELS = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e",
           "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au", "VOCALIC R": "ri",
           "VOCALIC RR": "ri", "VOCALIC L": "li", "CANDRA E": "e", "CANDRA O": "o",
           "SHORT E": "e", "SHORT O": "o"}
_cache = {}


def _info(c):
    r = _cache.get(c)
    if r is not None:
        return r
    nm = unicodedata.name(c, "")
    parts = nm.split(" ", 1)
    script = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if script not in _INDIC:
        r = ("other", "")
    elif rest.startswith("LETTER "):
        v = rest[7:]
        if v in _VOWELS:
            r = ("vowel", _VOWELS[v])
        else:
            base = v.lower()
            if base.endswith("a"):
                base = base[:-1]
            if base == "c":
                base = "ch"
            r = ("cons", base)
    elif rest.startswith("VOWEL SIGN "):
        r = ("sign", _VOWELS.get(rest[11:], ""))
    elif rest.startswith("SIGN VIRAMA") or rest == "SIGN VIRAMA":
        r = ("virama", "")
    elif rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU"):
        r = ("nasal", "n")
    elif rest == "SIGN VISARGA":
        r = ("nasal", "h")
    elif rest.startswith("DIGIT "):
        r = ("digit", str(unicodedata.digit(c, 0)))
    else:
        r = ("skip", "")
    _cache[c] = r
    return r


def romanize(text):
    """Return a lowercase ASCII approximation; Latin/ASCII passes through unchanged."""
    if not text or text.isascii():
        return text.lower() if text else ""
    out = []
    pending = False          # inherent 'a' owed by the last consonant
    for c in text:
        kind, val = _info(c)
        if kind == "other":
            if pending:
                out.append("a")
                pending = False
            if not c.isalnum():
                out.append(" ")
            else:
                out.append(c)
            continue
        if kind == "cons":
            if pending:
                out.append("a")
            out.append(val)
            pending = True
        elif kind == "sign":
            out.append(val)
            pending = False
        elif kind == "virama":
            pending = False
        elif kind == "vowel":
            if pending:
                out.append("a")
                pending = False
            out.append(val)
        elif kind == "nasal":
            if pending:
                out.append("a")
                pending = False
            out.append(val)
        elif kind == "digit":
            if pending:
                out.append("a")
                pending = False
            out.append(val)
    # final inherent vowel of the string is dropped (schwa deletion), so is any
    # inherent vowel right before a space (handled because spaces go through 'other'
    # branch after appending 'a'): remove word-final 'a' that came from inheritance
    s = "".join(out).lower()
    toks = []
    for t in s.split():
        for a, b in _SIMPLE:
            t = t.replace(a, b)
        toks.append(t)
    return " ".join(toks)


def romanize_words_schwa(text):
    """Variant with word-final schwa deletion (e.g. 'raama' -> 'ram')."""
    r = romanize(text)
    return " ".join(t[:-1] if len(t) > 3 and t.endswith("a") else t for t in r.split())


def has_indic(text):
    return any(_info(c)[0] not in ("other",) for c in text if ord(c) > 127)
