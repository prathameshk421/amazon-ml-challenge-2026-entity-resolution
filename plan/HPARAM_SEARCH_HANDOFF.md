# Running the hyperparameter search on another machine

For your friend's agent. This runs `plan/checks/hparam_search.py` in **case A**: tunes LightGBM's
settings for the stage-3 matcher on `MATCH_TRAIN` (180,000 S1), the training set the current best model
(`x1_strsim`) was fit on. It reproduces `x1_strsim` as an internal sanity check, then searches for better
hyperparameters and reports whether anything beats it by a real, confidence-checked margin. **It does not
train on the bigger `MATCH_ALL` dataset (case B)** — that data isn't ready to hand off yet, it's still
being generated on the original machine. Case A is fully self-contained and ready right now.

Nothing here needs a GPU. It needs CPU cores (more = faster) and about 8-16 GB RAM to be safe.

---

## 1. Get the code

```bash
git clone https://github.com/prathameshk421/amazon_ml_challenge.git
cd amazon_ml_challenge
git checkout submission-v2
```

Read `AGENTS.md` and `plan/ROUND_CAP_EVIDENCE.md` before this run finishes. Short version: a fixed
training-round cap that looked safe at a smaller data size has already been measured, twice, to be too
low at a bigger one — every fold's `best_iteration` needs checking against its cap, and if it's at or
near the cap, that fold didn't actually converge, it got cut off. `hparam_search.py` already accounts for
this (its cap scales with `learning_rate` and retries a truncated fold once at 2x), but the console
output should still be read carefully rather than trusted blindly — if the final report says any trial
was "TRUNCATED", say so when reporting results back rather than treating that trial's number as final.

## 2. Set up Python

Match versions if possible (not strictly required, but this pins what was tested):
Python 3.14, `lightgbm==4.7.0`, `scikit-learn==1.9.1`, `numpy==2.5.3`, `pandas==3.0.6`,
`pyarrow==25.0.1`. The repo's own pin list:

```bash
python3 -m venv .venv
.venv/bin/pip install -r submission/code/business_entity_resolution/requirements.txt
```

## 3. Copy in the data files (not in git — get these from the other machine)

These are **the only files needed** for this specific script. Total ~433 MB. Preserve the exact relative
paths under the repo root (create the `work/...` directory structure exactly as shown):

```
work/match/features/MATCH_TRAIN/          (71 MB  -- prebuilt 68-feature parquet + meta.json)
work/match/features/MATCH_VALID/          (8.7 MB -- same, for the confirmation check)
work/match/runs/x1_strsim/                (72 MB  -- models/matcher.json, OOF_per_entity.npy, queries.npy;
                                            used only to verify the search's own baseline reproduces
                                            x1_strsim exactly before trusting any comparison)
work/rerank/model_v1.txt                  (2.7 MB)
work/rerank/model_v1_meta.json            (4 KB)
work/records/train_gt.parquet             (45 MB  -- labels)
work/records/train_s1.parquet             (97 MB  -- country, for the guard-segment check)
work/records/train_s1_nmatch.npy          (4.2 MB -- match-count segment)
work/norm/train_s1.parquet                (83 MB  -- name text, for the "name unseen" segment)
work/norm/train_s1_chain.npy              (8.4 MB -- chain-size segment)
work/norm/train_pool_store/indic.npy      (9.8 MB -- just this one file, not the whole store)
work/splits/query_sets.json               (31 MB  -- which S1 belong to which slice)
```

Ask for these specific paths, not a generic "send me your work folder" — the full `work/` tree on the
original machine is many GB (candidate files, full index, raw pool text) and almost none of it is needed
for this particular script.

Zip command for whoever is sending these (run on the original machine, from the repo root):
```bash
zip -r hparam_handoff.zip \
  work/match/features/MATCH_TRAIN work/match/features/MATCH_VALID work/match/runs/x1_strsim \
  work/rerank/model_v1.txt work/rerank/model_v1_meta.json \
  work/records/train_gt.parquet work/records/train_s1.parquet work/records/train_s1_nmatch.npy \
  work/norm/train_s1.parquet work/norm/train_s1_chain.npy work/norm/train_pool_store/indic.npy \
  work/splits/query_sets.json
```
Then on the receiving machine, `unzip hparam_handoff.zip` from the repo root — it recreates the same
paths automatically.

## 4. Sanity check before the real run

```bash
export PYTHONPATH=submission/code/business_entity_resolution/src
.venv/bin/python -c "
from ber import io_utils, match
fp = match.bucketing_fingerprint(io_utils.load_config())
X, s1, pool, meta = match.load_features('MATCH_TRAIN', fp)
print('MATCH_TRAIN:', meta['n_queries'], 'S1,', X.shape[0], 'rows,', X.shape[1], 'features')
assert meta['n_queries'] == 180000 and X.shape[1] == 68
print('looks right')
"
```
If this errors on a missing file, re-check step 3 — something didn't get copied to the exact right path.

## 5. Run the search

```bash
cd amazon_ml_challenge
export PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks
.venv/bin/python plan/checks/hparam_search.py 2>&1 | tee work/logs/hparam_search.log
```

`CASE = "A"` is already set at the top of the script — don't change it (case B needs data that doesn't
exist yet on this machine). Leave `configs/match.json` and `configs/blocking.json` untouched; the script
only reads them, never writes them.

**What it does, in order** (see the script's own docstring and `plan/FEATURE_EXPERIMENTS.md` §2 for the
full methodology):
1. Trains the current shipped defaults (`learning_rate=0.05, num_leaves=63, ...`) inside the script and
   checks it reproduces `x1_strsim`'s saved out-of-fold scores exactly (deterministic training). If this
   check fails or reports a nonzero difference, **stop and report that** before trusting anything else —
   it means something about the environment or data doesn't match.
2. Screens 16 random hyperparameter configs (cheap, 3-fold) against that baseline.
3. Confirms the top 3 with a full 5-fold run, and does a real, statistical comparison (paired difference
   + 95% confidence interval, same method as every other experiment in this project) against the
   defaults, plus a `MATCH_VALID` confirmation.
4. Prints a final ADOPT or REJECT verdict, and writes the full detail to
   `artifacts/metrics/hparam_search.json`.

Expect this to take a while — round counts scale up for low learning rates (up to 8,000+ rounds per
fold), so a single trial can run much longer than `x1_strsim`'s original ~5 minutes. Let it run to
completion; there's no need to babysit it.

## 6. What to send back

Just these two things:
- `artifacts/metrics/hparam_search.json` (the full result)
- The console output / `work/logs/hparam_search.log`

Do not touch anything under `work/candidates/`, `work/index/`, or any file named `TEST` or
`HOLDOUT_TEST` — none of those are needed for this script and touching validation-S1 or test labels for
anything but a one-time official score is against the contest rules this project follows strictly. If the
script or anything else asks for those, stop and ask first rather than generating them.
