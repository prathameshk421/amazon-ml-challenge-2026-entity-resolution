"""M9 outputs (plan §5.10): candidate_pairs.tsv + matching_empty.tsv from candidates/TEST, then validate_stream.

candidate_pairs.tsv: header `source1_entity_id\\tcandidate_entity_ids`, one row per test S1 in file
order, IDs f"S{src}-{num}" in rank order joined by ",", empty allowed, newline="\\n", atomic write.
matching_empty.tsv is a format placeholder (every S1 with an empty list) and is never submitted.
Parts are streamed in s1 order, so only one part's ID strings are held at a time.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils
from ber.evaluate import size_stats
from ber.validate_stream import validate_candidates

CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
EMPTY_CHUNK = 100_000


def ordered_parts(parts_dir):
    """Finished part files sorted by s1 range; raises on a part without _SUCCESS or overlapping ranges."""
    parts = []
    for p in sorted(Path(parts_dir).glob("part-*.parquet")):
        if not p.with_suffix("._SUCCESS").exists():
            raise RuntimeError(f"{p} has no _SUCCESS marker (unfinished candidates run)")
        s1 = pq.read_table(p, columns=["s1_idx"]).column("s1_idx").to_numpy()
        if len(s1):
            parts.append((int(s1.min()), int(s1.max()), p))
    parts.sort()
    for (_, hi, a), (lo, _, b) in zip(parts, parts[1:]):
        if lo <= hi:
            raise RuntimeError(f"s1 ranges of {a.name} and {b.name} overlap")
    return [p for _, _, p in parts]


def _empty_rows(s1_ids, lo, hi):
    return "".join(f"{s1_ids[i]}\t\n" for i in range(lo, hi))


def write_candidate_file(parts_dir, s1_ids, src, num, out_path):
    """Write candidate_pairs.tsv atomically; return int32 list sizes per s1_idx."""
    n = len(s1_ids)
    sizes = np.zeros(n, dtype=np.int32)
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    nxt = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(CANDIDATE_HEADER)
        for part in ordered_parts(parts_dir):
            t = pq.read_table(part, columns=["s1_idx", "pool_idx", "rank"])
            s1 = t.column("s1_idx").to_numpy()
            if s1.min() < 0 or s1.max() >= n:
                raise RuntimeError(f"{part.name}: s1_idx outside [0, {n})")
            order = np.lexsort((t.column("rank").to_numpy(), s1))
            s1 = s1[order]
            pool = t.column("pool_idx").to_numpy()[order]
            ids = [f"S{s}-{m}" for s, m in zip(src[pool].tolist(), num[pool].tolist())]
            qs, starts, counts = np.unique(s1, return_index=True, return_counts=True)
            sizes[qs] = counts
            rows = []
            for q, st, c in zip(qs.tolist(), starts.tolist(), counts.tolist()):
                rows.append(_empty_rows(s1_ids, nxt, q))
                rows.append(f"{s1_ids[q]}\t{','.join(ids[st:st + c])}\n")
                nxt = q + 1
            f.write("".join(rows))
            del ids, rows
        for lo in range(nxt, n, EMPTY_CHUNK):
            f.write(_empty_rows(s1_ids, lo, min(lo + EMPTY_CHUNK, n)))
    os.replace(tmp, out_path)
    return sizes


def write_matching_empty(s1_ids, out_path):
    """Format placeholder: every S1 with an empty matched list (never submitted)."""
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(MATCHING_HEADER)
        for lo in range(0, len(s1_ids), EMPTY_CHUNK):
            f.write(_empty_rows(s1_ids, lo, min(lo + EMPTY_CHUNK, len(s1_ids))))
    os.replace(tmp, out_path)


def run_write_outputs(args):
    cfg = io_utils.load_config()
    log = io_utils.get_logger("write-outputs")
    t0 = time.time()
    s1 = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"),
                         columns=["s1_idx", "entity_id"])
    if not np.array_equal(s1["s1_idx"].to_numpy(), np.arange(len(s1))):
        raise RuntimeError("records/test_s1.parquet: s1_idx is not 0..n-1 in file order")
    s1_ids = s1["entity_id"].tolist()
    del s1
    work = io_utils.paths()["work_root"]
    store = work / "norm" / "test_pool_store"
    src = np.load(store / "src.npy", mmap_mode="r")
    num = np.load(store / "num.npy", mmap_mode="r")

    cand_path = io_utils.work_path("outputs", "candidate_pairs.tsv")
    sizes = write_candidate_file(work / "candidates" / "TEST", s1_ids, src, num, cand_path)
    log.info("wrote %s (%d rows, %d pairs)", cand_path, len(sizes), int(sizes.sum()))
    empty_path = io_utils.work_path("outputs", "matching_empty.tsv")
    write_matching_empty(s1_ids, empty_path)
    log.info("wrote %s", empty_path)

    test_dir = io_utils.paths()["data_root"] / "test"
    io_utils.log_test_usage("validate_stream on outputs/candidate_pairs.tsv",
                            "format + ID-existence validation of the test candidate file (§5.10)",
                            str(test_dir))
    issues = validate_candidates(str(cand_path), str(test_dir))
    log.info("validate_stream: %s", "PASS" if not issues else f"FAIL ({len(issues)} issues)")

    payload = {
        "rows": len(sizes),
        "pairs": int(sizes.sum()),
        "empty_lists": int((sizes == 0).sum()),
        "share_empty": float((sizes == 0).mean()) if len(sizes) else None,
        "sizes": size_stats(sizes, len(src)),
        "candidate_file_bytes": cand_path.stat().st_size,
        "matching_empty_file_bytes": empty_path.stat().st_size,
        "validation": {"validator": "validate_stream", "passed": not issues, "issues": issues},
        "wall_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(io_utils.peak_rss_mb(), 1),
    }
    io_utils.write_metrics("outputs_test", payload, n_queries=len(sizes), n_true_pairs=0, cfg=cfg)
    if issues:
        raise RuntimeError("candidate_pairs.tsv failed validate_stream:\n" + "\n".join(issues))
