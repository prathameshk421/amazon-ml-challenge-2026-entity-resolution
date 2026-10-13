"""M1 ingest asserts on a tiny fake dataset (plan §9 integration, §5.1)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ber.ingest import IngestError, ingest_split

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
FILES = {
    "train/train_source1.tsv": HEADER + "S1-10\tAcme \"Co\"\t1 Main St\tUS\n"
                                        "S1-7\tRam Stores\t\tIndia\n"
                                        "S1-99\tSolo\tNowhere\tUS\n",
    "train/train_source2.tsv": HEADER + "S2-5\tACME CO\t1 main st\tUS\n"
                                        "S2-6\tराम स्टोर्स\t\tIndia\n",
    "train/train_source3.tsv": HEADER + "S3-5\tAcme Company\t1 Main Street\tUS\n"
                                        "S3-8\tOther\tElsewhere\tUS\n",
    "train/train_ground_truth.tsv": "source1_entity_id\tmatched_entity_ids\n"
                                    "S1-10\tS3-5,S2-5\n"
                                    "S1-7\tS2-6\n"
                                    "S1-99\t\n",
    "test/test_source1.tsv": HEADER + "S1-1\tLa Maison\t3 r de la Paix\tFrance\n",
    "test/test_source2.tsv": HEADER + "S2-1\tLa Maison SAS\t3 rue de la Paix\tFrance\n",
    "test/test_source3.tsv": HEADER + "S3-1\tMaison\t\tFrance\n",
}


class IngestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data, self.out = self.root / "data", self.root / "work"
        for rel, text in FILES.items():
            self.write(rel, text)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel, text):
        p = self.data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")

    def append(self, rel, line):
        self.write(rel, FILES[rel] + line)

    def test_valid_train(self):
        stats = ingest_split(self.data, "train", self.out,
                             {"s1": 3, "s2": 2, "s3": 2, "gt_links": 3, "gt_singletons": 1,
                              "gt_max_nmatch": 2})
        rec = self.out / "records"
        s1 = pd.read_parquet(rec / "train_s1.parquet")
        pool = pd.read_parquet(rec / "train_pool.parquet")
        gt = pd.read_parquet(rec / "train_gt.parquet")
        self.assertEqual(s1["entity_id"].tolist(), ["S1-10", "S1-7", "S1-99"])
        self.assertEqual(s1["name_raw"][0], 'Acme "Co"')
        self.assertEqual(str(s1["s1_idx"].dtype), "int32")
        self.assertEqual(pool["pool_idx"].tolist(), [0, 1, 2, 3])
        self.assertEqual(pool["src"].tolist(), [2, 2, 3, 3])
        self.assertEqual(pool["num"].tolist(), [5, 6, 5, 8])
        self.assertEqual((str(pool["pool_idx"].dtype), str(pool["src"].dtype),
                          str(pool["num"].dtype)), ("uint32", "int8", "uint32"))
        self.assertEqual(list(zip(gt["s1_idx"], gt["pool_idx"])), [(0, 0), (0, 2), (1, 1)])
        self.assertEqual(np.load(rec / "train_s1_nmatch.npy").tolist(), [2, 1, 0])
        self.assertEqual(stats["files"]["s2"]["empty"]["addr"], 1)
        self.assertEqual(stats["gt_orphan_pool"], 1)
        self.assertFalse(list(rec.glob("*.tmp")))

    def test_valid_test_split(self):
        stats = ingest_split(self.data, "test", self.out, {"s1": 1, "s2": 1, "s3": 1})
        self.assertEqual(stats["pool"], 2)
        self.assertFalse((self.out / "records" / "train_gt.parquet").exists())

    def test_expected_count_mismatch(self):
        with self.assertRaises(IngestError):
            ingest_split(self.data, "test", self.out, {"s1": 2, "s2": 1, "s3": 1})

    def test_five_fields(self):
        self.append("train/train_source2.tsv", "S2-9\ta\tb\tUS\textra\n")
        with self.assertRaises(IngestError):
            ingest_split(self.data, "train", self.out)

    def test_bad_prefix(self):
        self.append("train/train_source2.tsv", "S3-9\ta\tb\tUS\n")
        with self.assertRaises(IngestError):
            ingest_split(self.data, "train", self.out)

    def test_leading_zero(self):
        self.append("train/train_source2.tsv", "S2-09\ta\tb\tUS\n")
        with self.assertRaises(IngestError):
            ingest_split(self.data, "train", self.out)

    def test_duplicate_id(self):
        self.append("train/train_source3.tsv", "S3-8\tdup\tb\tUS\n")
        with self.assertRaises(IngestError):
            ingest_split(self.data, "train", self.out)
        self.assertFalse((self.out / "records" / "train_pool.parquet").exists())

    def test_bad_header(self):
        self.write("test/test_source1.tsv", FILES["test/test_source1.tsv"].replace("country", "cc", 1))
        with self.assertRaises(IngestError):
            ingest_split(self.data, "test", self.out)

    def test_gt_unknown_id(self):
        self.write("train/train_ground_truth.tsv",
                   FILES["train/train_ground_truth.tsv"].replace("S2-6", "S2-12345"))
        with self.assertRaises(IngestError):
            ingest_split(self.data, "train", self.out)

    def test_gt_two_owners(self):
        self.write("train/train_ground_truth.tsv",
                   FILES["train/train_ground_truth.tsv"].replace("S1-99\t\n", "S1-99\tS2-5\n"))
        with self.assertRaises(IngestError):
            ingest_split(self.data, "train", self.out)

    def test_gt_missing_s1_row(self):
        self.write("train/train_ground_truth.tsv",
                   FILES["train/train_ground_truth.tsv"].replace("S1-99\t\n", ""))
        with self.assertRaises(IngestError):
            ingest_split(self.data, "train", self.out)


if __name__ == "__main__":
    unittest.main()
