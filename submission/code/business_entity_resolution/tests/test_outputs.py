"""§5.10 outputs: validate_stream rules, agreement with the official validator, write_candidate_file (plan §9)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ber.validate_stream import validate_candidates
from ber.write_outputs import write_candidate_file, write_matching_empty

REPO = Path(__file__).resolve().parents[4]
OFFICIAL = REPO / "student_resource" / "utils" / "validate_submission.py"
STREAM = Path(__file__).resolve().parents[1] / "src" / "ber" / "validate_stream.py"

S1 = ["S1-5", "S1-12", "S1-7", "S1-100", "S1-3"]
S2 = ["S2-7", "S2-8", "S2-40"]
S3 = ["S3-7", "S3-9"]
HEADER = "source1_entity_id\tcandidate_entity_ids\n"
GOOD = HEADER + ("S1-5\tS2-7,S3-7\n"
                 "S1-12\t\n"
                 "S1-7\tS3-9\n"
                 "S1-100\tS2-40,S2-8,S3-9\n"
                 "S1-3\t\n")


def write_source(path, ids):
    rows = "".join(f"{i}\tname {k}\taddr {k}\tUS\n" for k, i in enumerate(ids))
    Path(path).write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + rows,
                          encoding="utf-8", newline="\n")


class OutputsBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.test_dir = self.dir / "test"
        self.test_dir.mkdir()
        for n, ids in ((1, S1), (2, S2), (3, S3)):
            write_source(self.test_dir / f"test_source{n}.tsv", ids)

    def tearDown(self):
        self._tmp.cleanup()

    def cand(self, text, name="candidate_pairs.tsv"):
        p = self.dir / name
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return p

    def issues(self, text):
        return validate_candidates(str(self.cand(text)), str(self.test_dir))


class TestValidateStream(OutputsBase):
    def test_good_file_passes_both_validators(self):
        cand = self.cand(GOOD)
        self.assertEqual(validate_candidates(str(cand), str(self.test_dir)), [])
        self.assertNotIn(b"\r", cand.read_bytes())
        matching = self.dir / "matching_results.tsv"
        write_matching_empty(S1, matching)
        self.assertNotIn(b"\r", matching.read_bytes())
        out = subprocess.run([sys.executable, str(OFFICIAL), "--matching", str(matching),
                              "--candidate", str(cand), "--test-dir", str(self.test_dir),
                              "--check-ids"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("PASS", out.stdout)

    def test_cli(self):
        cand = self.cand(GOOD)
        ok = subprocess.run([sys.executable, str(STREAM), "--candidate", str(cand),
                             "--test-dir", str(self.test_dir)], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertIn("PASS", ok.stdout)
        bad = self.cand(GOOD.replace("S1-12\t\n", ""), "bad.tsv")
        fail = subprocess.run([sys.executable, str(STREAM), "--candidate", str(bad),
                               "--test-dir", str(self.test_dir)], capture_output=True, text=True)
        self.assertEqual(fail.returncode, 1)
        self.assertIn("1. bad.tsv: required S1 entity(ies) missing: S1-12", fail.stdout)

    def assertCaught(self, text, fragment):
        issues = self.issues(text)
        self.assertTrue(any(fragment in i for i in issues), issues)

    def test_duplicate_row(self):
        self.assertCaught(GOOD + "S1-3\tS2-7\n", "duplicate source1_entity_id row(s): S1-3")

    def test_missing_s1(self):
        self.assertCaught(GOOD.replace("S1-7\tS3-9\n", ""), "required S1 entity(ies) missing: S1-7")

    def test_unknown_s1_row(self):
        self.assertCaught(GOOD + "S1-999\t\n", "S1 ID that is not in the test set: S1-999")

    def test_s1_id_in_list(self):
        self.assertCaught(GOOD.replace("S1-12\t\n", "S1-12\tS1-5\n"), "Source-1 IDs (self-matches): S1-5")

    def test_unknown_id(self):
        self.assertCaught(GOOD.replace("S1-12\t\n", "S1-12\tS2-999\n"),
                          "not in the test Source-2/3 files: S2-999")
        self.assertCaught(GOOD.replace("S1-12\t\n", "S1-12\tS3-8\n"),
                          "not in the test Source-2/3 files: S3-8")  # S2-8 exists, S3-8 does not
        self.assertCaught(GOOD.replace("S1-12\t\n", "S1-12\tS2-07\n"),
                          "not in the test Source-2/3 files: S2-07")

    def test_duplicate_id_in_list(self):
        self.assertCaught(GOOD.replace("S1-7\tS3-9\n", "S1-7\tS3-9,S3-9\n"),
                          "repeated ID inside a candidate_entity_ids list for: S1-7")

    def test_bad_header(self):
        self.assertCaught(GOOD.replace("candidate_entity_ids", "matched_entity_ids"), "unexpected header")
        self.assertCaught(GOOD.replace("\t", ",", 1), "COMMA-separated")

    def test_header_case_insensitive(self):
        self.assertEqual(self.issues(GOOD.replace("source1_entity_id", "Source1_Entity_ID")), [])

    def test_crlf(self):
        self.assertCaught(GOOD.replace("\n", "\r\n"), "carriage return")

    def test_space_after_comma(self):
        self.assertCaught(GOOD.replace("S2-7,S3-7", "S2-7, S3-7"), "without an S2-/S3- prefix:  S3-7")

    def test_malformed_row(self):
        self.assertCaught(GOOD + "S1-3 S2-7\n", "malformed row (no tab)")

    def test_missing_pool_source(self):
        (self.test_dir / "test_source3.tsv").unlink()
        self.assertCaught(GOOD, "test_source3.tsv")


class TestWriteCandidateFile(OutputsBase):
    def test_end_to_end(self):
        # pool: S2-7, S2-8, S2-40, S3-7, S3-9 (pool_idx 0..4)
        src = np.array([2, 2, 2, 3, 3], dtype=np.int8)
        num = np.array([7, 8, 40, 7, 9], dtype=np.uint32)
        parts = self.dir / "candidates" / "TEST"
        parts.mkdir(parents=True)

        def part(name, rows):
            s1, pool, rank = (np.array(c) for c in zip(*rows))
            pd.DataFrame({"s1_idx": s1.astype(np.int32), "pool_idx": pool.astype(np.uint32),
                          "rank": rank.astype(np.int16),
                          "p_rerank": np.linspace(0.9, 0.1, len(s1)).astype(np.float32)}
                         ).to_parquet(parts / f"{name}.parquet", index=False)
            (parts / f"{name}._SUCCESS").write_text("")

        # rows deliberately out of order; S1 idx 1 and 4 get no candidates
        part("part-00001", [(3, 4, 3), (3, 2, 1), (3, 1, 2)])
        part("part-00000", [(2, 4, 1), (0, 3, 2), (0, 0, 1)])
        out = self.dir / "outputs" / "candidate_pairs.tsv"
        out.parent.mkdir()
        sizes = write_candidate_file(parts, S1, src, num, out)
        self.assertEqual(out.read_bytes().decode("utf-8"), GOOD)
        self.assertEqual(sizes.tolist(), [2, 0, 1, 3, 0])
        self.assertEqual(validate_candidates(str(out), str(self.test_dir)), [])
        self.assertFalse(out.with_name(out.name + ".tmp").exists())

        (parts / "part-00002.parquet").write_bytes((parts / "part-00000.parquet").read_bytes())
        with self.assertRaisesRegex(RuntimeError, "_SUCCESS"):
            write_candidate_file(parts, S1, src, num, out)
        (parts / "part-00002._SUCCESS").write_text("")
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            write_candidate_file(parts, S1, src, num, out)

    def test_matching_empty(self):
        out = self.dir / "matching_empty.tsv"
        write_matching_empty(S1, out)
        self.assertEqual(out.read_bytes().decode("utf-8"),
                         "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{i}\t\n" for i in S1))


if __name__ == "__main__":
    unittest.main()
