"""Read side of the M5 index (plan §5.6, used by §5.7 steps 1-2).

All arrays are opened with np.load(mmap_mode="r"), so construct an IndexStore inside each spawned worker.
"""

import numpy as np

from ber import hashing, io_utils

_EMPTY = np.zeros(0, dtype=np.uint32)


class _AtDf:
    """Full AT df of the split pool: df for indexed tokens, n_pool for tokens over the build limit
    (dropped; above any cap), `default` (0) for absent tokens. Same `.get` interface as a dict."""

    def __init__(self, store):
        self._store = store

    def get(self, token, default=0):
        d = int(self._store.df("AT", np.array([hashing.key64("AT", token)], dtype=np.uint64))[0])
        if d == self._store.sentinel:
            return self._store.n_pool
        return d if d else default


class IndexStore:
    """mmap view of index/{split}/ for the given families (AT also opened when XK is, for at_df_lookup)."""

    def __init__(self, split, families):
        self.dir = io_utils.paths()["work_root"] / "index" / split
        meta = io_utils.read_json(self.dir / "meta.json")
        self.n_pool = int(meta["n_pool"])
        self.G = frozenset(meta["G"])
        self.at_cap = int(meta["at_cap"])
        self.sentinel = self.n_pool + 1  # df reported for dropped keys: larger than any cap
        self._keys, self._indptr, self._post, self._dropped = {}, {}, {}, {}
        for F in dict.fromkeys(list(families) + (["AT"] if "XK" in families else [])):
            for d, x in ((self._keys, "keys"), (self._indptr, "indptr"), (self._post, "post"),
                         (self._dropped, "dropped")):
                d[F] = np.load(self.dir / f"{F}.{x}.npy", mmap_mode="r")
        self.at_df_lookup = _AtDf(self)

    @staticmethod
    def _member(sorted_arr, k):
        """(pos, found) of each k in a sorted uint64 array."""
        pos = np.searchsorted(sorted_arr, k)
        if sorted_arr.size == 0:
            return pos, np.zeros(k.shape, dtype=bool)
        found = sorted_arr[np.minimum(pos, sorted_arr.size - 1)] == k
        return pos, found

    def df(self, F, key64):
        """int64 df per key: 0 absent, df if indexed, self.sentinel if dropped (over the build limit)."""
        k = np.asarray(key64, dtype=np.uint64)
        pos, found = self._member(self._keys[F], k)
        ip = self._indptr[F]
        out = np.zeros(k.shape, dtype=np.int64)
        p = pos[found]
        out[found] = ip[p + 1] - ip[p]
        out[self._member(self._dropped[F], k)[1]] = self.sentinel
        return out

    def postings(self, F, key64):
        """Posting slice (uint32, ascending) per key, aligned with key64; empty if absent or dropped."""
        k = np.asarray(key64, dtype=np.uint64)
        pos, found = self._member(self._keys[F], k)
        ip, post = self._indptr[F], self._post[F]
        return [post[ip[p]:ip[p + 1]] if f else _EMPTY for p, f in zip(pos.tolist(), found.tolist())]
