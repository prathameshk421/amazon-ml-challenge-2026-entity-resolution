"""Collect licenses of allowlisted libraries from local pip metadata (offline).
Writes artifacts/metrics/licenses.json; prints markdown table for reports/licenses.md.
Run: .venv/bin/python legacy/src/collect_licenses.py
"""

import importlib.metadata as md
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MET = os.path.join(ROOT, "artifacts/metrics")

LIBS = ["pandas", "numpy", "scikit-learn", "scipy", "rapidfuzz", "jellyfish",
        "lightgbm", "xgboost", "sentence-transformers", "torch", "transformers",
        "huggingface-hub", "matplotlib", "tqdm", "pyarrow"]


def main():
    rows = {}
    for lib in LIBS:
        try:
            m = md.metadata(lib)
            lic = (m.get("License-Expression", "") or "").strip()
            if not lic:
                for k, v in m.items():
                    if k == "Classifier" and "License ::" in str(v):
                        lic = str(v).strip()
                        break
            if not lic:
                lic = (m.get("License", "") or "?").strip().split("\n")[0][:120]
            rows[lib] = {"version": m.get("Version", "?"),
                         "license": lic[:200] if lic else "?"}
        except md.PackageNotFoundError:
            rows[lib] = {"version": "NOT INSTALLED", "license": "n/a"}
    with open(os.path.join(MET, "licenses.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("| lib | version | license |")
    print("|---|---|---|")
    for lib, r in rows.items():
        print("| %s | %s | %s |" % (lib, r["version"], r["license"]))


if __name__ == "__main__":
    main()
