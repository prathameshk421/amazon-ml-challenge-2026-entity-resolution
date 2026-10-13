"""Country x singleton 80/20 assignment matches the dataset TSV split."""

import unittest

from ber.splits import s1_train_val_ids


class TestS1Split(unittest.TestCase):
    def test_four_strata_round_and_disjoint(self):
        # 10 per cell -> round(0.2*10)=2 val, 8 train
        eids, country, nmatch = [], [], []
        n = 0
        for ctry in ("India", "US"):
            for nmatch_v in (0, 3):
                for i in range(10):
                    eids.append(f"S1-{n}")
                    country.append(ctry)
                    nmatch.append(nmatch_v)
                    n += 1
        train, val = s1_train_val_ids(eids, country, nmatch)
        self.assertEqual(len(val), 8)
        self.assertEqual(len(train), 32)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(eids))
        train2, val2 = s1_train_val_ids(eids, country, nmatch)
        self.assertEqual(train, train2)
        self.assertEqual(val, val2)


if __name__ == "__main__":
    unittest.main()
