"""Streaming, memory-light validator for `candidate_pairs.tsv` (plan §5.10, F15).

Mirrors every rule `student_resource/utils/validate_submission.py` enforces for the candidate file
(header compared lower-case, only "\\n" stripped, IDs split on ",", empty list allowed, one row per
test S1, no duplicate S1 rows, no duplicate IDs in a list, only S2-/S3- IDs, no S1 IDs, no unknown S1
rows) plus two stricter checks: ID existence is always on, and no "\\r" anywhere. Rows are read one at
a time; IDs are looked up in sorted np.uint64 arrays of (src << 32 | num) with np.searchsorted instead
of Python sets. Stdlib + numpy only, so it runs standalone:

    python validate_stream.py --candidate <file> --test-dir <dir>
"""

import argparse
import heapq
import os
import sys
from array import array

import numpy as np

DELIM = "\t"
MAX_EXAMPLES = 5  # how many offending IDs to show per issue
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
COL = "candidate_entity_ids"
FLUSH_KEYS = 1 << 20  # ID lookups are batched per this many keys
NUM_LIMIT = 1 << 32


def parse_id(eid, prefix):
    """Return the uint32 number of a canonical `<prefix><num>` ID (no leading zeros), else None."""
    if not eid.startswith(prefix):
        return None
    digits = eid[len(prefix):]
    if not (digits.isascii() and digits.isdigit()) or (len(digits) > 1 and digits[0] == "0"):
        return None
    num = int(digits)
    return num if num < NUM_LIMIT else None


def format_key(key):
    return f"S{key >> 32}-{key & (NUM_LIMIT - 1)}"


def examples(items, key=None):
    """Short sample of `items` for an error message (sorted, like the official validator)."""
    items = sorted(items, key=key)
    shown = ", ".join(str(i) for i in items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown


def load_keys(path, srcs, issues):
    """Sorted unique uint64 keys (src << 32 | num) of the first-column IDs of a source TSV.

    Header skipped, blank lines ignored, IDs stripped (as the official `read_ids`). Returns None and
    records an issue when the file is missing or holds an ID this encoding cannot represent.
    """
    if not os.path.isfile(path):
        issues.append(f"Test source file not found: {path} (check --test-dir). The ID-existence "
                      "check is mandatory in validate_stream.")
        return None
    keys, bad = array("Q"), []
    with open(path, "rb") as f:
        next(f, None)
        for raw in f:
            if not raw.strip():
                continue
            eid = raw.split(b"\t", 1)[0].decode("utf-8").strip()
            for src in srcs:
                num = parse_id(eid, f"S{src}-")
                if num is not None:
                    keys.append((src << 32) | num)
                    break
            else:
                bad.append(eid)
    if bad:
        issues.append(f"{os.path.basename(path)}: entity_id(s) not of the form S<src>-<number>: "
                      f"{examples(bad)}.")
        return None
    return np.unique(np.frombuffer(keys, dtype=np.uint64))


def _lookup(keys, valid):
    """Keys of the uint64 array `keys` that are absent from the sorted array `valid`."""
    pos = np.searchsorted(valid, keys)
    hit = pos < len(valid)
    hit[hit] = valid[pos[hit]] == keys[hit]
    return keys[~hit]


def _check_rows(path, s1_keys, pool_keys, issues):
    """Stream the candidate file; append issues; return (n_rows, n_empty) or None on a fatal problem."""
    name = os.path.basename(path)
    counts = np.zeros(len(s1_keys), dtype=np.uint8)  # rows seen per required S1 (saturates at 2)
    unknown_s1, dup_rows, intra_dupes = set(), set(), set()
    self_matches, wrong_prefix, unknown, cr_lines = set(), set(), set(), []
    pending = array("Q")
    n_rows = empties = 0

    with open(path, "rb") as f:
        header = f.readline().decode("utf-8")
        if not header:
            issues.append(f"{name} is empty.")
            return None
        if "\r" in header:
            cr_lines.append(1)
        if DELIM not in header and "," in header:
            issues.append(f"{name}: header has no TAB but contains commas — the file looks "
                          "COMMA-separated. Submissions must be TAB-separated (.tsv).")
            return None
        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != CANDIDATE_HEADER:
            issues.append(f"{name}: unexpected header {cols}. "
                          f"Expected exactly {CANDIDATE_HEADER} (tab-separated).")
            return None

        for line_num, raw in enumerate(f, start=2):
            line = raw.decode("utf-8")
            if "\r" in line:
                cr_lines.append(line_num)
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    issues.append(f"{name}: malformed row (no tab) at line {line_num}: "
                                  f"{line.rstrip()!r}")
                continue

            n_rows += 1
            num = parse_id(s1, "S1-")
            pos = int(np.searchsorted(s1_keys, num)) if num is not None else len(s1_keys)
            if pos < len(s1_keys) and s1_keys[pos] == num:
                if counts[pos]:
                    dup_rows.add(s1)
                counts[pos] = min(int(counts[pos]) + 1, 2)
            else:
                if s1 in unknown_s1:
                    dup_rows.add(s1)
                unknown_s1.add(s1)

            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                continue
            id_set = set(ids)
            if len(ids) != len(id_set):
                intra_dupes.add(s1)
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                else:
                    src = 2 if mid[1] == "2" else 3
                    num = parse_id(mid, f"S{src}-")
                    if num is None:
                        unknown.add(mid)
                    else:
                        pending.append((src << 32) | num)
            if len(pending) >= FLUSH_KEYS:
                unknown.update(map(format_key, _lookup(np.frombuffer(pending, np.uint64),
                                                       pool_keys).tolist()))
                pending = array("Q")
    if pending:
        unknown.update(map(format_key, _lookup(np.frombuffer(pending, np.uint64),
                                               pool_keys).tolist()))

    missing = np.flatnonzero(counts == 0)
    missing_ex = heapq.nsmallest(MAX_EXAMPLES + 1, (f"S1-{int(k)}" for k in s1_keys[missing]))
    missing_msg = (examples(missing_ex) if len(missing) <= MAX_EXAMPLES else
                   f"{len(missing)} total, e.g. {', '.join(missing_ex[:MAX_EXAMPLES])}, ...")
    findings = [
        (dup_rows, "{name}: duplicate source1_entity_id row(s): {ex}. "
                   "Each S1 entity may appear on only one row."),
        (intra_dupes, "{name}: repeated ID inside a {col} list for: {ex}. "
                      "No duplicate IDs are allowed within a list."),
        (self_matches, "{name}: {col} contains Source-1 IDs (self-matches): {ex}. "
                       "Only S2-/S3- IDs are allowed."),
        (wrong_prefix, "{name}: {col} contains IDs without an S2-/S3- prefix: {ex}."),
        (unknown, "{name}: {col} references IDs not in the test Source-2/3 files: {ex}."),
    ]
    for offenders, message in findings:
        if offenders:
            issues.append(message.format(name=name, ex=examples(offenders), col=COL))
    if len(missing):
        issues.append(f"{name}: required S1 entity(ies) missing: {missing_msg}. "
                      "Every entity in test_source1.tsv needs a row (empty = no match).")
    if unknown_s1:
        issues.append(f"{name}: row(s) using an S1 ID that is not in the test set: "
                      f"{examples(unknown_s1)}.")
    if cr_lines:
        issues.append(f"{name}: carriage return (\\r) on line(s): {examples(cr_lines, key=int)}. "
                      "Write with newline=\"\\n\".")
    return n_rows, empties


def _validate(candidate_path, test_dir):
    """Return (issues, (n_rows, n_empty) or None)."""
    issues = []
    try:
        if not os.path.isfile(candidate_path):
            return [f"File not found: {candidate_path}"], None
        source1 = os.path.join(test_dir, "test_source1.tsv")
        if not os.path.isfile(source1):
            return [f"Test source1 file not found: {source1} (check --test-dir)."], None
        s1_keys = load_keys(source1, (1,), issues)
        s2 = load_keys(os.path.join(test_dir, "test_source2.tsv"), (2,), issues)
        s3 = load_keys(os.path.join(test_dir, "test_source3.tsv"), (3,), issues)
        if s1_keys is None or s2 is None or s3 is None:
            return issues, None
        s1_keys &= np.uint64(NUM_LIMIT - 1)  # S1 keys: the number alone
        pool_keys = np.union1d(s2, s3)
        del s2, s3
        return issues, _check_rows(candidate_path, s1_keys, pool_keys, issues)
    except UnicodeDecodeError:
        issues.append(f"A file is not valid UTF-8 text (most likely {candidate_path}). Re-save it "
                      "as a plain UTF-8, tab-separated .tsv.")
    except OSError as exc:
        issues.append(f"Could not read a file: {exc}.")
    return issues, None


def validate_candidates(candidate_path, test_dir):
    """Validate a candidate_pairs.tsv against the test sources; return the list of issues (empty = PASS)."""
    return _validate(candidate_path, test_dir)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Streaming validator for candidate_pairs.tsv (bucketing output only).")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv",
                        help="Path to candidate_pairs.tsv (default: %(default)s)")
    parser.add_argument("--test-dir", "-t", default="dataset/test",
                        help="Folder with test_source1/2/3.tsv (default: %(default)s)")
    args = parser.parse_args()

    print("candidate_pairs.tsv streaming validator")
    print(f"  test dir: {args.test_dir}")
    issues, counts = _validate(args.candidate, args.test_dir)
    if counts is not None:
        n_rows, empties = counts
        print(f"  {os.path.basename(args.candidate)}: {n_rows} rows ({empties} empty, "
              f"{n_rows - empties} non-empty).")
    print()
    if issues:
        print(f"FAIL — {len(issues)} issue(s) to fix before submitting:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return 1
    print("PASS — no blocking issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
