"""End-to-end driver: contest TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv (submission 18).

Run from the run folder: the folder that contains student_resource/dataset/{train,test}/. Intermediates go to
work/, metrics to artifacts/metrics/, one log per step to logs/<step>.log, final files to output/.

  python code/business_entity_resolution/src/reproduce.py --list
  python code/business_entity_resolution/src/reproduce.py                      # every step, models retrained
  python code/business_entity_resolution/src/reproduce.py --models shipped     # training steps skipped
  python code/business_entity_resolution/src/reproduce.py --from drace_untrained
  python code/business_entity_resolution/src/reproduce.py --only merge

--models train   (default) trains every model from the training data, as the submission was built.
--models shipped copies the trained models in code/business_entity_resolution/models/ into work/ and skips every
                 step marked "train". The cross-encoder epoch-1 adapter (ce_rec) is always taken from models/: it was
                 trained with src/ce/train_ce_rec.py, which reproduce.py does not run (README section 5).

Steps marked GPU run the cross-encoder / e5 code on the local device (CUDA; MPS and CPU work but are far slower).
Step order is the order the submission was built in, including the intermediate submission-9 and submission-13
predictions that later steps read (DRACE seeds, group features, exclusion lists).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
PKG = SRC.parent
if (SRC / "ce").is_dir():  # submission zip layout
    CE, PIPE, MODELS = SRC / "ce", SRC / "pipeline", PKG / "models"
else:  # team repository layout (package at submission/code/business_entity_resolution/)
    ROOT = PKG.parents[2]
    CE, PIPE, MODELS = ROOT / "ce", ROOT / "plan" / "checks", PKG / "models"
JOB = CE / "hf_job"
PY = sys.executable
EXTRA = "DRACD_TEST,DRACD5_TEST,SEED5_TEST,DRACD7_TEST,DRACD10_TEST"


def ber(*args):
    return [PY, "-m", "ber.cli", *args]


def py(script, *args, env=None):
    return ([PY, "-u", str(script), *args], env or {})


def score(sets):
    """c0 cross-encoder p_ce for work/ce/pairs_<SET>.parquet -> work/ce/ce_rec_c0/<SET>.parquet."""
    return py(JOB / "score_job.py", env={"CE_LOCAL": "work", "CE_ADAPTER": "work/ce/ce_rec_c0/adapter",
                                         "CE_NAME": "ce_rec_c0", "CE_SETS": sets})


def retrieve(tag, k, out_tag=None, query_tag=None, model_dir=None):
    """e5 embeddings + per-country exact cosine top-k -> work/retrieval/<out_tag>_topk.parquet."""
    env = {"RETR_LOCAL": "work", "RETR_TAG": tag, "RETR_K": str(k), "RETR_OUT_TAG": out_tag or tag}
    if query_tag:
        env["RETR_QUERY_TAG"] = query_tag
    if model_dir:
        env["RETR_MODEL_DIR"] = model_dir
    return py(JOB / "retrieve_job.py", env=env)


def copy_file(src, dst):
    def f():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")
    return f


def copy_tree(src, dst):
    def f():
        if Path(dst).exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"copied {src}/ -> {dst}/")
    return f


def train_c0_adapter():
    """Continue ce_rec for 1 epoch on DEV_REST_REST fold 0 (ce/train_continue.py), then install the adapter."""
    out = Path("work/ce/c0_train")
    return [py(CE / "train_continue.py", "--pairs", "work/ce", "--base", "BAAI/bge-reranker-v2-m3",
               "--init-adapter", str(MODELS / "ce_rec_adapter"), "--fold", "0", "--out", str(out),
               "--budget-hours", "8.5"),
            copy_tree(out / "adapter", "work/ce/ce_rec_c0/adapter")]


def threshold_073():
    """full_c0_rgrp_t073 = full_c0_rgrp with only decision.threshold changed 0.70 -> 0.73 (submission 16 on)."""
    src, dst = Path("work/match/runs/full_c0_rgrp"), Path("work/match/runs/full_c0_rgrp_t073")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    p = dst / "models" / "matcher.json"
    d = json.loads(p.read_text())
    d["decision"]["threshold"] = 0.73
    p.write_text(json.dumps(d, indent=1))
    print(f"{p}: decision {d['decision']}")


def install_shipped():
    """--models shipped: trained models from code/business_entity_resolution/models/ into work/."""
    pairs = [(MODELS / "stage1" / "model_v1.json", "work/stage1/model_v1.json")]
    pairs += [(MODELS / "rerank" / f, f"work/rerank/{f}") for f in ("model_v1.txt", "model_v1_meta.json", "cut_v1.json")]
    for s, d in pairs:
        copy_file(s, d)()
    copy_tree(MODELS / "ce_rec_c0_adapter", "work/ce/ce_rec_c0/adapter")()
    for run in ("full_c0", "full_c0_rgrp"):
        copy_tree(MODELS / "matcher" / run, f"work/match/runs/{run}/models")()
    if (MODELS / "e5_distil").is_dir():
        copy_tree(MODELS / "e5_distil", "work/retrieval/e5_distil")()
    else:
        print("models/e5_distil not in this package: step 'distill' is a train step; drace_distilled needs --models train")


def finish():
    """Official validator (if the contest utils folder is present) and sha1 of the matching file."""
    out = Path("output")
    val = Path("student_resource/utils/validate_submission.py")
    if val.exists():
        subprocess.run([PY, str(val), "--matching", str(out / "matching_results.tsv"), "--candidate",
                        str(out / "candidate_pairs.tsv"), "--test-dir", "student_resource/dataset/test",
                        "--check-ids"], check=True)
    else:
        print(f"{val} not found: official validator skipped")
    h = hashlib.sha1((out / "matching_results.tsv").read_bytes()).hexdigest()
    print(f"sha1 output/matching_results.tsv {h} (submission 18: 7334635c...)")


# name, kind ("data" runs in both modes, "train" is skipped with --models shipped), device, what, actions
STEPS = [
    ("ingest", "data", "CPU", "read contest TSVs, S1 train/validation split",
     [ber("ingest", "--split", "train"), ber("ingest", "--split", "test"), ber("splits")]),
    ("normalize", "data", "CPU", "text normalization; abbreviation maps; transliteration dictionary (train pairs)",
     [ber("mine-abbrev", "--split", "train"), ber("normalize", "--split", "train", "--no-dict"), ber("learn-dict"),
      ber("normalize", "--split", "train"), ber("mine-abbrev", "--split", "test"), ber("normalize", "--split", "test")]),
    ("index_train", "train", "CPU", "stage-1 inverted index, train pool",
     [ber("build-index", "--split", "train")]),
    ("index_test", "data", "CPU", "stage-1 inverted index, test pool",
     [ber("build-index", "--split", "test")]),
    ("stage1", "train", "CPU", "stage-1 logistic blocking model",
     [ber("stage1-train")]),
    ("stage2", "train", "CPU", "stage-2 LightGBM re-ranker and cut",
     [ber("rerank-build"), ber("rerank-train"), ber("rerank-cut")]),
    ("candidates_train", "train", "CPU", "candidate lists + features: MATCH_TRAIN, MATCH_VALID, DEV_REST_180K, DEV_REST_REST",
     [ber("candidates", "--set", "MATCH_TRAIN"), ber("candidates", "--set", "MATCH_VALID"),
      ber("match-features", "--set", "MATCH_TRAIN"), ber("match-features", "--set", "MATCH_VALID"),
      py(PIPE / "build_dev_rest.py")]),
    ("candidates_test", "data", "CPU", "candidate lists + features: TEST",
     [ber("candidates", "--set", "TEST"), ber("match-features", "--set", "TEST")]),
    ("ce_pairs_train", "train", "CPU", "cross-encoder pair files for training sets; DEV_REST_REST fold-1 file",
     [py(CE / "prepare_pairs.py", "--sets", "DEV_REST_180K", "MATCH_TRAIN", "MATCH_VALID", "DEV_REST_REST"),
      py(PIPE / "split_rest_folds.py")]),
    ("ce_pairs_test", "data", "CPU", "cross-encoder pair file for TEST",
     [py(CE / "prepare_pairs.py", "--sets", "TEST")]),
    ("ce_train", "train", "GPU", "cross-encoder c0: ce_rec LoRA continued 1 epoch on DEV_REST_REST fold 0",
     train_c0_adapter()),
    ("ce_score_train", "train", "GPU", "c0 scores: MATCH_VALID (AUC gate), MATCH_TRAIN, DEV_REST_REST fold 1",
     [score("MATCH_VALID,MATCH_TRAIN,DEV_REST_REST_F1"),
      copy_file("work/ce/ce_rec_c0/DEV_REST_REST_F1.parquet", "work/ce/ce_rec_c0/DEV_REST_REST.parquet")]),
    ("ce_score_test", "data", "GPU", "c0 scores: TEST",
     [score("TEST")]),
    ("matcher_s9", "train", "CPU", "LightGBM full_c0 (68 features + c0 score); its predictions seed DRACE",
     [py(PIPE / "train_final_comp.py", "full_c0", "cex_p:ce_rec_c0", env={"REST_FOLD": "1", "EXCLUDE_SETS": "DEV_REST_180K"})]),
    ("predict_s9", "data", "CPU", "full_c0 on TEST -> work/outputs/full_c0/matching_results.tsv",
     [ber("exp-predict", "--run", "full_c0")]),
    ("rev_train", "train", "GPU", "reverse-name retrieval features, training sets",
     [py(PIPE / "rev_prep.py", "prep", "train"), retrieve("rev_train", 20),
      py(PIPE / "rev_prep.py", "feats", "MATCH_TRAIN", "MATCH_VALID", "DEV_REST_REST")]),
    ("rev_test", "data", "GPU", "reverse-name retrieval features, TEST",
     [py(PIPE / "rev_prep.py", "prep", "test"), retrieve("rev_test", 20), py(PIPE / "rev_prep.py", "feats", "TEST")]),
    ("grp_train", "train", "CPU", "first-pass OOF decisions and group-profile features, training sets",
     [py(PIPE / "firstpass_oof.py"), py(PIPE / "grp_feats.py", "--firstpass", "MATCH_TRAIN", "MATCH_VALID", "DEV_REST_REST")]),
    ("grp_test", "data", "CPU", "group-profile features, TEST (group = full_c0 acceptances)",
     [py(PIPE / "grp_feats.py", "--firstpass", "TEST")]),
    ("matcher_s12", "train", "CPU", "LightGBM full_c0_rgrp (85 features), the live matcher",
     [py(PIPE / "train_final_comp.py", "full_c0_rgrp", "cex_p:ce_rec_c0", "rev", "nfreq", "grp_full",
         env={"REST_FOLD": "1", "EXCLUDE_SETS": "DEV_REST_180K"})]),
    ("predict_s12", "data", "CPU", "full_c0_rgrp on TEST at threshold 0.70 (S13 base) and 0.73 (S18 base)",
     [ber("exp-predict", "--run", "full_c0_rgrp"), threshold_073, ber("exp-predict", "--run", "full_c0_rgrp_t073")]),
    ("drace_untrained", "data", "GPU", "DRACE: untrained e5, S1 top 5 + copy expansion top 2; c0 scores",
     [py(PIPE / "drace_test.py", "prep"), retrieve("drace_test", 5),
      py(PIPE / "drace_test.py", "pairs", "--k1", "5", "--k2", "2"), score("DRACE_TEST")]),
    ("distill", "train", "GPU", "e5-small distilled from c0 (InfoNCE + MarginMSE) -> work/retrieval/e5_distil",
     [py(PIPE / "distill_e5.py", "--dump"), py(JOB / "distill_job.py", env={"D_LOCAL": "work"})]),
    ("drace_distilled", "data", "GPU", "DRACE: distilled e5 S1 top 10, copy expansion ranks 3-5; c0 scores",
     [py(PIPE / "drace_test.py", "s1_queries"),
      retrieve("drace_test", 10, out_tag="dracd10_test", query_tag="dracd_test_s1", model_dir="work/retrieval/e5_distil"),
      py(PIPE / "drace_test.py", "pairs_extra", "--topk-tag", "dracd10_test", "--set", "DRACD_TEST", "--k1", "2"),
      py(PIPE / "drace_test.py", "pairs_extra", "--topk-tag", "dracd10_test", "--set", "DRACD5_TEST", "--k1", "5",
         "--skip", "DRACD_TEST"),
      score("DRACD_TEST,DRACD5_TEST"),
      py(PIPE / "drace_test.py", "merge", "--t", "0.995", "--base", "work/outputs/full_c0_rgrp/matching_results.tsv",
         "--out", "work/outputs/s13", "--tag", "drace_test_s13", "--extra", "DRACD_TEST,DRACD5_TEST"),
      py(PIPE / "seed5_test_pairs.py", "--s13", "work/outputs/s13/matching_results.tsv"),
      py(PIPE / "drace_test.py", "pairs_extra", "--topk-tag", "dracd10_test", "--set", "DRACD7_TEST", "--k1", "7",
         "--skip", "DRACD_TEST,DRACD5_TEST,SEED5_TEST"),
      py(PIPE / "drace_test.py", "pairs_extra", "--topk-tag", "dracd10_test", "--set", "DRACD10_TEST", "--k1", "10",
         "--skip", "DRACD_TEST,DRACD5_TEST,SEED5_TEST,DRACD7_TEST"),
      score("SEED5_TEST,DRACD7_TEST,DRACD10_TEST")]),
    ("merge", "data", "CPU", "matcher pairs + DRACE pairs with c0 >= 0.995, one-owner -> output/",
     [py(PIPE / "drace_test.py", "merge", "--t", "0.995", "--base", "work/outputs/full_c0_rgrp_t073/matching_results.tsv",
         "--out", "output", "--tag", "drace_test_final", "--extra", EXTRA),
      finish]),
]


def run_action(action, log, env_base):
    if callable(action):
        action()
        return
    if isinstance(action, tuple):
        cmd, extra = action
    else:
        cmd, extra = action, {}
    env = dict(env_base, **extra)
    shown = " ".join(f"{k}={v}" for k, v in extra.items()) + (" " if extra else "") + " ".join(
        Path(c).name if i == 0 else str(c) for i, c in enumerate(cmd))
    print(f"$ {shown}", flush=True)
    log.write(f"$ {shown}\n")
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        sys.stdout.write(line)
        log.write(line)
        log.flush()
    if p.wait() != 0:
        raise SystemExit(f"command failed (exit {p.returncode}) after {time.time() - t0:.0f}s: {shown}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", choices=("train", "shipped"), default="train")
    ap.add_argument("--from", dest="start", help="start at this step")
    ap.add_argument("--to", dest="stop", help="stop after this step")
    ap.add_argument("--only", help="comma-separated steps to run")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for name, kind, dev, what, _ in STEPS:
            print(f"{name:18s} {kind:5s} {dev:3s}  {what}")
        return
    for n in filter(None, [a.start, a.stop] + (a.only.split(",") if a.only else [])):
        if n not in names:
            raise SystemExit(f"unknown step {n!r}; --list shows the steps")
    sel = names[names.index(a.start) if a.start else 0: names.index(a.stop) + 1 if a.stop else len(names)]
    if a.only:
        sel = [n for n in names if n in a.only.split(",")]
    if a.models == "shipped":
        sel = [n for n in sel if dict((s[0], s[1]) for s in STEPS)[n] == "data"]
    if not Path("student_resource/dataset/test").is_dir():
        raise SystemExit("student_resource/dataset/test not found: run from the folder that holds the contest data")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(CE), str(JOB), str(PIPE)] +
                                        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    Path("logs").mkdir(exist_ok=True)
    if a.models == "shipped":
        install_shipped()
    print(f"steps: {', '.join(sel)}", flush=True)
    t_all = time.time()
    for name, kind, dev, what, actions in STEPS:
        if name not in sel:
            continue
        print(f"\n>>> {time.strftime('%H:%M:%S')} step {name} ({dev}): {what}", flush=True)
        t0 = time.time()
        with open(Path("logs") / f"{name}.log", "a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} step {name}\n")
            for action in actions:
                run_action(action, log, env)
        print(f">>> step {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nall selected steps done in {(time.time() - t_all) / 3600:.2f} h")


if __name__ == "__main__":
    main()
