"""Own BK-tree over strings with rapidfuzz Levenshtein distance (C2: implement yourself)."""

from rapidfuzz.distance import Levenshtein


class BKTree:
    def __init__(self, distance=None):
        self.dist = distance or (lambda a, b: Levenshtein.distance(a, b))
        self.root = None  # (term, payload, {dist: child})

    def add(self, term, payload=None):
        node = [term, payload if payload is not None else term, {}]
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            d = self.dist(term, cur[0])
            if d == 0:
                # duplicate term: keep payload list
                if not isinstance(cur[1], list):
                    cur[1] = [cur[1]]
                cur[1].append(node[1])
                return
            nxt = cur[2].get(d)
            if nxt is None:
                cur[2][d] = node
                return
            cur = nxt

    def query(self, term, k):
        if self.root is None:
            return []
        out = []
        stack = [self.root]
        while stack:
            t, p, children = stack.pop()
            d = self.dist(term, t)
            if d <= k:
                out.append((d, p))
            lo, hi = d - k, d + k
            for cd, child in children.items():
                if lo <= cd <= hi:
                    stack.append(child)
        return out
