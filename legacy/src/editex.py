"""Own Editex-style phonetic edit distance (C2: implement yourself).

Groups follow Zobel & Dart: identical=0, same-group=1, otherwise=2;
insertions/deletions cost 1. Similarity = 1 - D/(len(a)+len(b)) in [0,1]
(max distance = delete-all + insert-all since sub cost <= 2 = del+ins).
"""

_GROUPS = ["aeiouy", "bp", "ckq", "dt", "lr", "mn", "gj", "fpv", "sxz", "csz"]
_GRP = {}
for _g in _GROUPS:
    for _ch in _g:
        _GRP.setdefault(_ch, []).append(_g)


def _sub(a, b):
    if a == b:
        return 0
    ga, gb = _GRP.get(a), _GRP.get(b)
    if ga and gb and set(ga) & set(gb):
        return 1
    return 2


def editex_dist(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i]
        for j in range(1, lb + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + _sub(ca, b[j - 1])))
        prev = cur
    return prev[lb]


def editex_sim(a, b):
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - editex_dist(a, b) / (len(a) + len(b))
