"""Build the final contest package <team>_submission.zip from the repository (run from the repository root).

  .venv/bin/python submission/make_submission_zip.py [--sub 18] [--team VadaPav] [--stage <dir>]

Zip layout (contest instructions):
  output/matching_results.tsv, output/candidate_pairs.tsv      submission/output_v<N>/ (the leaderboard upload)
  code/business_entity_resolution/
    README.md, requirements.txt, configs/, tests/              package files
    src/ber/                                                   stages 1-3
    src/ce/, src/ce/hf_job/                                    cross-encoder, e5 retrieval and distillation
    src/pipeline/                                              stage-3 feature groups, matcher training, DRACE
    src/reproduce.py                                           end-to-end driver
    models/                                                    trained models of the submission
  Documentation_template.md
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

PKG = Path("submission/code/business_entity_resolution")
CE_FILES = ["common.py", "prepare_pairs.py", "train.py", "train_ce_rec.py", "train_continue.py"]
JOB_FILES = ["score_job.py", "retrieve_job.py", "ann_search.py", "distill_job.py", "collect.py"]
PIPE_FILES = ["build_dev_rest.py", "dev_rest_180k.py", "dev_rest_full.py", "split_rest_folds.py", "rev_prep.py",
              "firstpass_oof.py", "grp_feats.py", "train_final_comp.py", "drace_test.py", "seed5_test_pairs.py",
              "distill_e5.py"]
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
MODELS = {  # destination under models/ -> source files
    "stage1": ["work/stage1/model_v1.json"],
    "rerank": ["work/rerank/model_v1.txt", "work/rerank/model_v1_meta.json", "work/rerank/cut_v1.json"],
    "ce_rec_adapter": [f"cross_encoder_lora/adapter/{f}" for f in ADAPTER_FILES],
    "ce_rec_c0_adapter": [f"work/ce/ce_rec_c0/adapter/{f}" for f in ADAPTER_FILES],
    "matcher/full_c0": ["work/match/runs/full_c0/models/matcher.json", "work/match/runs/full_c0/models/model_fold0.txt"],
    "matcher/full_c0_rgrp": ["work/match/runs/full_c0_rgrp/models/matcher.json",
                             "work/match/runs/full_c0_rgrp/models/model_fold0.txt"],
}
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".pytest_cache")


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="18")
    ap.add_argument("--team", default="VadaPav")
    ap.add_argument("--stage", default="work/submission_zip")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stage = Path(a.stage)
    if stage.exists():
        shutil.rmtree(stage)
    code = stage / "code" / "business_entity_resolution"
    src = code / "src"

    (stage / "output").mkdir(parents=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(Path(f"submission/output_v{a.sub}") / f, stage / "output" / f)
    shutil.copy2("submission/Documentation_template.md", stage / "Documentation_template.md")

    for f in ("README.md", "requirements.txt"):
        shutil.copy2(PKG / f, code / f) if code.exists() else (code.mkdir(parents=True), shutil.copy2(PKG / f, code / f))
    shutil.copytree(PKG / "configs", code / "configs", ignore=SKIP)
    shutil.copytree(PKG / "tests", code / "tests", ignore=SKIP)
    shutil.copytree(PKG / "src" / "ber", src / "ber", ignore=SKIP)
    shutil.copy2(PKG / "src" / "reproduce.py", src / "reproduce.py")
    (src / "ce" / "hf_job").mkdir(parents=True)
    for f in CE_FILES:
        shutil.copy2(Path("ce") / f, src / "ce" / f)
    for f in JOB_FILES:
        shutil.copy2(Path("ce/hf_job") / f, src / "ce" / "hf_job" / f)
    (src / "pipeline").mkdir()
    for f in PIPE_FILES:
        shutil.copy2(Path("plan/checks") / f, src / "pipeline" / f)
    for dst, files in MODELS.items():
        (code / "models" / dst).mkdir(parents=True)
        for f in files:
            shutil.copy2(f, code / "models" / dst / Path(f).name)
    # The ce_rec adapter was saved with a machine-local base_model_name_or_path; point the staged copy
    # at the public Hub id (weights untouched; the c0 adapter already stores the Hub id).
    ce_rec_cfg = code / "models" / "ce_rec_adapter" / "adapter_config.json"
    cfg = json.loads(ce_rec_cfg.read_text())
    cfg["base_model_name_or_path"] = "BAAI/bge-reranker-v2-m3"
    ce_rec_cfg.write_text(json.dumps(cfg, indent=2) + "\n")

    out = Path(a.out or f"submission/{a.team}_submission.zip")
    out.unlink(missing_ok=True)
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(stage).as_posix())
    print(f"{out}: {len(files)} files, {out.stat().st_size / 2**20:.1f} MB")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        print(f"output/{f} sha1 {sha1(stage / 'output' / f)}")


if __name__ == "__main__":
    main()
