"""Reference implementation of plan §5.2.1 (romanizer) + §5.2 steps 1-12 for names/addresses
(without the learned dictionary) + §5.5.1 skeleton. Used to validate the spec and freeze goldens.
Stdlib only.
"""

import re
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
    r = _cache.get(c)
    if r is not None:
        return r
    o = ord(c)
    if not (INDIC_LO <= o <= INDIC_HI):
        r = ("other", c)
    else:
        nm = unicodedata.name(c, "")
        rest = nm.split(" ", 1)[1] if " " in nm else ""
        if rest.startswith("LETTER CHILLU "):
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


FOLD = {"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "ø": "o", "Ø": "o", "ł": "l",
        "Ł": "l", "đ": "d", "Đ": "d", "ð": "d", "þ": "th", "ı": "i", "ħ": "h"}
NAME_ABBREV = {"corp": "corporation", "inc": "incorporated", "ltd": "limited", "pvt": "private",
               "co": "company", "svc": "services", "assoc": "associates", "bros": "brothers",
               "shri": "sri", "shree": "sri"}
ADDR_ABBREV = {"rd": "road", "st": "street", "ave": "avenue", "avenu": "avenue", "dr": "drive",
               "ln": "lane", "ct": "court", "blvd": "boulevard", "ste": "suite", "apt": "apartment",
               "hwy": "highway", "hway": "highway", "pkwy": "parkway"}
_NULL = re.compile(r"<NULL>")
_PH = re.compile(r"<[A-Z_]+>")
_APOS = re.compile("['’‘ʼ`]")
_LD = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")


def normalize_field(text, kind="name", tdict=None):
    t = unicodedata.normalize("NFC", text)
    t = _NULL.sub(" ", t)
    t = _PH.sub(" cityname ", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Cf")
    t, had = romanize(t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = "".join(FOLD.get(c, c) for c in t)
    t = t.casefold()
    t = t.replace("&", " and ")
    t = _APOS.sub("", t)
    t = "".join(c if c.isalnum() else " " for c in t)
    if kind == "addr":
        t = _LD.sub(" ", t)
    table = NAME_ABBREV if kind == "name" else ADDR_ABBREV
    toks = [table.get(w, w) for w in t.split()]
    if had and tdict:
        toks = [tdict.get(w, w) for w in toks]
    return " ".join(toks), had


_SK_MAP = {**{c: "P" for c in "pbfvw"}, **{c: "T" for c in "td"}, **{c: "K" for c in "kgc"},
           **{c: "S" for c in "szj"}, "l": "L", "r": "R", "m": "M", "n": "N"}


def skeleton(tok):
    if not tok.isascii():
        return ""
    t = tok.replace("x", "ks").replace("ph", "f").replace("ch", "s").replace("sh", "s").replace("q", "k")
    out = []
    for c in t:
        if c.isdigit():
            s = c
        else:
            s = _SK_MAP.get(c, "")
        if s and (not out or out[-1] != s):
            out.append(s)
    return "".join(out)
