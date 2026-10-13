"""Unit tests for legacy/src/metric.py. Run: .venv/bin/python legacy/src/test_metric.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import f05_single, macro_f05, candidate_recall


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(("PASS" if ok else "FAIL"), name, "got=%.6f want=%.6f" % (got, want))
    if not ok:
        raise SystemExit("FAILED: " + name)


# Spec example: P=2/3, R=1.0 -> 0.714285... (spec rounds to 0.714)
pred = {"S2-00047", "S2-00193", "S3-00812"}
truth = {"S2-00047", "S3-00812"}
check("spec example", f05_single(pred, truth), 0.7142857142857143, tol=1e-3)

# empty/empty -> 1.0
check("empty/empty", f05_single(set(), set()), 1.0)
# empty truth + non-empty pred -> 0.0
check("singleton over-predict", f05_single({"S2-1"}, set()), 0.0)
# non-empty truth + empty pred -> 0.0
check("miss all", f05_single(set(), {"S2-1"}), 0.0)
# perfect prediction -> 1.0
check("perfect", f05_single({"S2-1", "S3-2"}, {"S2-1", "S3-2"}), 1.0)
# partial: P=0.5 R=0.5 -> F0.5 = 1.25*.25/(.125+.5) = 0.3125/0.625 = 0.5
check("half/half", f05_single({"a", "b"}, {"a", "c"}), 0.5)
# no overlap -> 0.0
check("no overlap", f05_single({"a"}, {"b"}), 0.0)

# macro average includes entities with zero candidates (missing key = empty)
truth_d = {"e1": {"a", "b"}, "e2": set(), "e3": {"c"}}
pred_d = {"e1": {"a"}}  # e2 missing (=empty, correct singleton), e3 missing (=empty, wrong)
m = macro_f05(pred_d, truth_d)
print("macro:", m)
# e1: P=1,R=0.5 -> 1.25*.5/(.25+.5)=0.625/0.75=0.8333; e2: 1.0; e3: 0.0
check("macro F0.5", m["F0.5"], (0.8333333333 + 1.0 + 0.0) / 3, tol=1e-6)
check("singleton_acc", m["singleton_acc"], 1.0)
assert m["n_entities"] == 3 and m["n_singleton"] == 1 and m["n_nonsingleton"] == 2

# candidate recall
cr = candidate_recall({"e1": {"a", "x"}, "e3": set()}, truth_d, all_entities=["e1", "e2", "e3"])
print("cand recall:", cr)
check("pair recall", cr["pair_recall"], 1 / 3)
check("entity recall", cr["entity_recall"], 0.5)

print("ALL METRIC TESTS PASSED")
