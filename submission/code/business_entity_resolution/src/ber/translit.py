"""Indic romanizer (plan §5.2.1): verbatim port of plan/reference/translit_ref.py. Stdlib only."""

import unicodedata

INDIC_LO, INDIC_HI = 0x0900, 0x0D7F

CONS = {"KA": "k", "KHA": "kh", "GA": "g", "GHA": "gh", "NGA": "n", "CA": "ch", "CHA": "ch",
        "JA": "j", "JHA": "jh", "NYA": "n", "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh",
        "NNA": "n", "TA": "t", "THA": "th", "DA": "d", "DHA": "dh", "NA": "n", "NNNA": "n",
        "PA": "p", "PHA": "ph", "BA": "b", "BHA": "bh", "MA": "m", "YA": "y", "RA": "r", "RRA": "r",
        "LA": "l", "LLA": "l", "LLLA": "l", "VA": "v", "SHA": "sh", "SSA": "sh", "SA": "s",
        "HA": "h", "QA": "k", "KHHA": "kh", "GHHA": "g", "ZA": "z", "DDDHA": "r", "RHA": "r",
        "FA": "f", "YYA": "y"}
VOW = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "VOCALIC R": "ri",
       "VOCALIC RR": "ri", "VOCALIC L": "li", "VOCALIC LL": "li", "E": "e", "EE": "e", "AI": "ai",
       "O": "o", "OO": "o", "AU": "au", "CANDRA E": "e", "SHORT E": "e", "CANDRA O": "o",
       "SHORT O": "o", "CANDRA A": "a"}

_cache = {}


def classify(c):
    """-> (class, romanization) of one character; non-Indic characters are ('other', c)."""
    r = _cache.get(c)
    if r is not None:
        return r
    o = ord(c)
    if not (INDIC_LO <= o <= INDIC_HI):
        r = ("other", c)
    else:
        nm = unicodedata.name(c, "")
        rest = nm.split(" ", 1)[1] if " " in nm else ""
        if rest in ("TIPPI", "SIGN BINDI"):     # Gurmukhi nasal signs (TIPPI has no "SIGN" in its name)
            r = ("anusvara", "")
        elif rest == "ADDAK":                  # Gurmukhi gemination mark: doubling collapses anyway
            r = ("ignore", "")
        elif rest == "OM":                     # DEVANAGARI OM ligature
            r = ("vowel", "om")
        elif rest.startswith("LETTER CHILLU "):
            x = rest[len("LETTER CHILLU "):]
            r = ("cons_dead", CONS.get(x + "A", x.lower()))
        elif rest == "LETTER KHANDA TA":
            r = ("cons_dead", "t")
        elif rest.startswith("LETTER "):
            x = rest[7:]
            if x in VOW:
                r = ("vowel", VOW[x])
            else:
                r = ("cons", CONS.get(x, x.lower()[:-1] if x.endswith("A") else x.lower()))
        elif rest.startswith("VOWEL SIGN "):
            r = ("vsign", VOW.get(rest[11:], ""))
        elif rest == "SIGN VIRAMA":
            r = ("virama", "")
        elif rest == "SIGN ANUSVARA":
            r = ("anusvara", "")
        elif rest == "SIGN CANDRABINDU":
            r = ("nasal", "n")
        elif rest == "SIGN VISARGA":
            r = ("nasal", "h")
        elif rest.startswith("DIGIT "):
            r = ("digit", str(unicodedata.digit(c)))
        elif rest.startswith("SIGN ") or "LENGTH MARK" in rest:
            r = ("ignore", "")
        else:
            r = ("space", " ")
    _cache[c] = r
    return r


def _is_rra(c):
    return INDIC_LO <= ord(c) <= INDIC_HI and unicodedata.name(c, "").endswith(" LETTER RRA")


def _is_na(c):
    n = unicodedata.name(c, "") if INDIC_LO <= ord(c) <= INDIC_HI else ""
    return n.endswith(" LETTER NA") or n.endswith(" LETTER NNNA")


def romanize(text):
    """-> (text with Indic parts romanized, had_indic)"""
    out = []
    owe = False
    had = False
    chars = list(text)
    n = len(chars)
    for i, c in enumerate(chars):
        k, v = classify(c)
        if k != "other":
            had = True
        if k == "cons":
            if owe:
                out.append("a")
            if _is_rra(c) and i >= 2 and classify(chars[i - 1])[0] == "virama":
                pk = chars[i - 2]
                if _is_rra(pk):          # RRA VIRAMA RRA -> "t" (drop the first "r")
                    if out and out[-1] == "r":
                        out.pop()
                    v = "t"
                elif _is_na(pk):         # (NA|NNNA) VIRAMA RRA -> "n" + "t"
                    v = "t"
            out.append(v)
            owe = True
        elif k == "cons_dead":
            if owe:
                out.append("a")
            out.append(v)
            owe = False
        elif k == "vsign":
            out.append(v)
            owe = False
        elif k == "virama":
            owe = False
        elif k == "vowel":
            if owe:
                out.append("a")
            out.append(v)
            owe = False
        elif k == "anusvara":
            if owe:
                out.append("a")
                owe = False
            nxt = ""
            for j in range(i + 1, n):
                kj, vj = classify(chars[j])
                if kj == "ignore":
                    continue
                if kj in ("cons", "cons_dead"):
                    nxt = vj
                break
            out.append("m" if nxt[:1] in ("p", "b", "m") else "n")
        elif k == "nasal":
            if owe:
                out.append("a")
                owe = False
            out.append(v)
        elif k == "ignore":
            continue
        else:  # other, digit, space -> word boundary for schwa deletion
            owe = False  # word-final inherent vowel dropped
            out.append(v)
    return "".join(out), had
