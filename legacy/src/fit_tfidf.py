"""Fit char-ngram TF-IDF vectorizers on 500K pool sample (unsupervised) + token idf map.
Writes artifacts/tfidf_name.pkl, artifacts/tfidf_addr.pkl, artifacts/tok_idf.json.
Run: .venv/bin/python legacy/src/fit_tfidf.py
"""

import json
import os
import pickle
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX = os.path.join(ROOT, "artifacts/index")
ART = os.path.join(ROOT, "artifacts")


def main():
    rng = random.Random(99)
    nblob = np.memmap(os.path.join(IDX, "pool.names.bin"), mode="r", dtype=np.uint8)
    noff = np.load(os.path.join(IDX, "pool.names.off.npy"), mmap_mode="r")
    ablob = np.memmap(os.path.join(IDX, "pool.addrs.bin"), mode="r", dtype=np.uint8)
    aoff = np.load(os.path.join(IDX, "pool.addrs.off.npy"), mmap_mode="r")
    n = len(noff) - 1
    idx = rng.sample(range(n), 500000)
    names, addrs = [], []
    for i in idx:
        names.append(bytes(nblob[noff[i]:noff[i + 1]]).decode("utf-8", "replace"))
        addrs.append(bytes(ablob[aoff[i]:aoff[i + 1]]).decode("utf-8", "replace"))
    nn = [normalize_name(x) or " " for x in names]
    na = [normalize_address(x) or " " for x in addrs]
    vec_n = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100000)
    vec_n.fit(nn)
    print("name vocab:", len(vec_n.vocabulary_))
    vec_a = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100000)
    vec_a.fit(na)
    print("addr vocab:", len(vec_a.vocabulary_))
    with open(os.path.join(ART, "tfidf_name.pkl"), "wb") as f:
        pickle.dump(vec_n, f)
    with open(os.path.join(ART, "tfidf_addr.pkl"), "wb") as f:
        pickle.dump(vec_a, f)
    df = json.load(open(os.path.join(IDX, "pool.tokdf.json")))
    N = 10320219
    tok_idf = {t: float(np.log((N + 1) / (v + 1))) for t, v in df.items()}
    json.dump(tok_idf, open(os.path.join(ART, "tok_idf.json"), "w"))
    print("tok_idf size:", len(tok_idf))


if __name__ == "__main__":
    main()
