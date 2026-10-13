# Research log

Lab notebook for every experiment: what we tried, what the number did, what we concluded.
Negative, flat, and failed runs are logged too — they are the part people forget and then re-run.

**Rules**

1. One entry per run or decision, **newest first**. Append; never rewrite an old entry. If a later run
   shows an entry was wrong, add a new entry that says so and link back (`see M007`).
2. Every number cites the `artifacts/metrics/*.json` that holds it (AGENTS rule 6). No file, no number.
   A run that crashed or was killed is logged as `FAILED` with the reason.
3. Prose is the point: hypothesis before the run, interpretation after. "Δ +0.0004" alone is not an entry.
4. Update **Current best** in the same edit as any `ADOPTED` entry.
5. IDs: `E0–E6` stage-1 blocking, `R*` stage 2, `M###` stage-3 matcher/feature runs
   (`exps/M###.json`, see `plan/EXPERIMENT_ROUTINE.md`), `N*` measurement-only, `S*` submissions.
6. `reports/experiments.md` (generated) is the table; this file is the narrative. PROGRESS.md tracks
   phases, not experiments.

**Entry template**

```
### YYYY-MM-DD · M### · <short title> · ADOPTED | PARKED | DROPPED | FAILED | INFO
- Parent: M### · Commit: <sha> · Metrics: artifacts/metrics/<file>.json · Wall: <min>
- Hypothesis: <pattern seen, in which file/tag count> → <feature/change> should move <metric> because <mechanism>.
- Change: <exact feature names / params / rule>.
- Result: F0.5 <parent> → <new>, paired Δ <x> [CI lo, hi]; fixed <n> / broken <n>; segments that moved.
  Error tags: <tag> FP <a>→<b>, FN <a>→<b>. Mini-lockbox (if adopted): <value>.
- Learned: <what this says about the data or model, including "no effect">.
- Next: <the follow-up this suggests, or "none">.
```

---

## Current best

| Stage | Set | Metric | Value | Entry | Metrics file |
|---|---|---|---|---|---|
| 1 blocking | DEV_B 10K | F0.5 ceiling | 0.99550 | E6 | `s1_E6.json` |
| 1+2 candidates | Val S1 441,363 | F0.5 ceiling / mean size | 0.99276 / 5.43 | R2 | `rerank_eval_HOLDOUT_TEST.json` |
| 3 matcher | MATCH_TRAIN 180K OOF | entity F0.5 | 0.97412 | M000 | `match_train.json` |
| 3 matcher | MATCH_VALID 20K | entity F0.5 | 0.97449 | M000 | `match_eval_MATCH_VALID.json` |
| 3 matcher | MATCH_TRAIN 180K OOF | entity F0.5 | 0.98107 (x1_strsim 0.97807; test-like Δ +0.00186) | M005 | `exp_M005.json`, `universe_owner_M005.json` |
| 3 matcher | MATCH_VALID 20K | entity F0.5 | 0.98162 | M005 | `match_valid_confirm_M005.json` |
| 3 matcher | MATCH_TRAIN 180K OOF | entity F0.5 | 0.98211 (comp + num) | M016 | `exp_M016.json` |
| 3 matcher | MATCH_ALL held-out 165K | entity F0.5 | 0.98360 (full_comp; full_single 0.98161 same S1) | M005-full | `match_train_full_comp.json` |
| 3 matcher | MATCH_ALL − DEV_REST_180K held-out 147K | entity F0.5 | 0.98537 (full_ce, + p_ce; full_single 0.98162 same S1) | S6-file | `match_train_full_ce.json` |
| 3 matcher | MATCH_ALL − DEV_REST_180K held-out 147K | entity F0.5 | 0.98551 (full_ce_nf, + nfreq) | S7-file | `match_train_full_ce_nf.json` |
| submission | test (public LB) | entity F0.5 | **0.984075 (S18)**; S17 0.984028; S16 0.983958; S14 0.983741; S13 0.983604; S12 0.983064; S11 0.981623; S10 0.98135; S7 0.976205; S6 0.976128; S2 0.966797; S1 0.966109; S3 0.966 (rounded) | S10-LB | `drace_test.json`, `reports/test_usage_log.md` |

## Open questions

- **Target (set 2026-09-27 12:55): public LB 0.98**, i.e. +0.0038 over S7 (0.976205). Two uploads left. Plan: slot 1 = S9 (c0, LightGBM on ~837K S1) if its calibration-slice gain over full_single ≥ ~+0.0045; slot 2 = S10 (c0 + c1 cross-fit, LightGBM on ~1.47M S1, + nfreq). Estimated S9 ≈ +0.001 on the LB (d); 0.98 is above every estimate so far.
- Internal→leaderboard gap −0.0092 (S6; was −0.0148 at S2). N03/N04 explain ~0.008 from list statistics; ~0.006 is not visible without labels. Plan: `plan/RCA_RESEARCH_PLAN.md`.
- Paired SE of a feature change at 180K. Unmeasured until the first M run (N01 only covers one-owner).
- How much the OOF score is inflated by `translit_dict` being mined from DICT_SRC labels (VALIDATION_PLAN P4). Unmeasured.

---

## Log

### 2026-09-28 · SMOKE3 · c0 rescoring drift vs reference, cause open · INFO
- Rescoring 2,000 MATCH_VALID rows with the shipped c0 adapter (T4 torch 2.6, and Mac MPS) vs
  `work/ce/ce_rec_c0/MATCH_VALID.parquet`: same adapter sha1 `7398d2ce`; AUC 0.99906; mean |diff| 0.0011,
  max 0.446; 7/2000 flips at 0.995, 3 at 0.73. Not truncation (all top-5 diffs ≤ 86 tokens). Cause not
  established: numerics vs. a different pair serialization in the reference run (2 of the top 5 have empty
  pool addresses). Submission files unaffected.

### 2026-09-28 · SMOKE2 · CUDA smoke of submission package, retry (unpinned deps) · INFO
- Job `pkg-cuda-smoke4` (6abaafe052d0dbd7f1da3c38), t4-small, COMPLETED in 98 s running (~$0.011).
  Two earlier attempts failed in minutes on setup (no `huggingface_hub` in image python; `pip` vs `python`
  interpreter split; missing `-r`; numpy pin needs Python >= 3.12 vs image 3.11). Retry kept image torch
  2.6.0+cu124 and installed the rest unpinned.
- 2a PASS (PIP_EXIT=0, torch 2.6.0+cu124 on Tesla T4 kept). 2b PASS: `device cuda, amp torch.float16`,
  5 steps (fold 0: 956 train pairs; held-out logloss 0.02203 -> 0.02007), `/tmp/c0_smoke/adapter/
  adapter_model.safetensors` present (13,661,132 B). Tokenizer loaded from the base, confirming the fix.
- 2c MIXED: SMOKE AUC 0.99907; n=2000 maxdiff=0.417245 meandiff=0.001088 flips073=3 flips995=7. Against the
  rubric (flips995=0, maxdiff ~1e-3) this is not clean: 7 pairs cross 0.995 between the shipped-adapter
  rescoring (unpinned transformers, fp32 merge on T4) and the original A100 run. Not root-caused; candidates:
  library-version drift from the unpinned install, fp32-vs-bf16 merge dtype, T4-vs-A100 numerics.
- Bundle `smoke/` deleted from the hub repo. No score claims; the 2c gap needs a look before DRACE depends on it.

### 2026-09-28 · SMOKE · CUDA smoke of submission package on HF t4-small · FAILED (infra, no score)
- Job `pkg-cuda-smoke` (6abaaecf52d0dbd7f1da3bf8), image pytorch:2.6.0-cuda12.4, 45m cap (~$0.30 worst case).
  Failed in minutes at setup: the image's `python` has no `huggingface_hub` (bundle download failed) and `pip`
  vs `python` resolved to different interpreters (PIP_EXIT=0, yet `import torch` failed on missing `pyarrow`).
  Fix for retry: `python -m pip` throughout and pre-install `huggingface_hub` before the download step.
- Bundle `smoke/` uploaded then deleted from the hub repo. No scores produced; nothing to claim.

### 2026-09-27 · S18-LB · Submission 18 leaderboard: S17 + distilled S1 ranks 8-10, 0.984075 · ADOPTED
- Log: `reports/test_usage_log.md` · Metrics: `drace_test_s18.json` · Chain: `plan/checks/run_s18.sh` (log `logs/run_s18.log`)
  · File: `submission/output_v18/matching_results.tsv` (sha1 7334635c…).
- Test: 1,347,669 new pairs (ranks 8-10) scored on h200 (2.9 min); DRACE additions 84,402 (S17 83,104); final 5,812,655.
- Result: public LB 0.984075 vs S17 0.984028: +0.00005. Train evidence +0.00001 (seed5_eval.json, seed5+d10 vs seed5+d10_r67);
  transfer ≈ 5x on a tiny base — distilled-retriever depth keeps transferring above train size.
- Ops: two failed launches first (retrieve_job needs ann_search.py and pandas in the job image; fixed). The CPU-tokenisation
  prefetch did not change embedding speed (19K records/s on h200 both ways).
- Test-side diagnostic (`logs/country_shift_s17.log`): France predicted 3.243 pairs/S1 vs India 3.365, US 3.385 (train truth
  3.46/S1), France candidate lists 9.05 vs 5.6. Possible France recall gap; not measurable without France labels.

### 2026-09-27 · OW1 · DRACE ownership: fewest-matches-first instead of highest c0 · DROPPED (nothing to decide)
- Metrics: `owner_rule.json` · Script: `plan/checks/owner_rule.py` · Log: `logs/owner_rule.log`.
- S17-equivalent on train: 6,870 DRACE pairs; only 5 pool records claimed by > 1 S1. fewest-first −0.00001, most-first −0.00000.

### 2026-09-27 · S17-LB · Submission 17 leaderboard: S16 + distilled S1 ranks 6-7, 0.984028 (rank 663) · ADOPTED
- Log: `reports/test_usage_log.md` · File: `submission/output_v17/matching_results.tsv` (sha1 3adb3960…).
- Result: public LB 0.984028 vs S16 0.983958: +0.00007. Train evidence +0.00004; transfer ≈ 1.75x (distilled-retriever
  changes keep transferring above their train size, like S12/S13; seed expansion did not, S14).

### 2026-09-27 · I1-test · Profile-query (iterative) retrieval on top of S17 · DROPPED (overlap)
- Scripts: `plan/checks/iter_test_prep.py`, `plan/checks/iter_gate_prep.py`, `ce/hf_job/iter_job.py` (now env-parametrised,
  batched profile sum) · Log: `logs/iter_gate_prep.log`. HF iter-test (h200) launched 23:15 and cancelled after the gate.
- Train: profile top-5 pairs not reached by any S17 channel: 60,178, of which 83 true (ceiling ~+0.00003). I1's +0.00030
  oracle was measured against S13's channels; seed ranks 3-5 and distilled S1 ranks 6-7 already recover almost all of it
  (both also search around accepted records).

### 2026-09-27 · A2 · DRACE c0 admission cut-off under test-like weighting (S17 channels) · DROPPED (flat)
- Metrics: `admit_testlike.json` · Script: `plan/checks/admit_testlike.py` · Log: `logs/admit_testlike.log`.
- Test-like weighted F0.5 0.98864-0.98867 for T 0.98-0.997; drops at 0.998+ (0.999: 0.98853). Halves chose 0.99 / 0.995;
  cross-fitted vs 0.995 −0.00001. Keep 0.995.

### 2026-09-27 · S17-file · S16 + distilled-e5 S1 query ranks 6-7 · INFO (upload candidate)
- Parent: S16-file, D10 (ranks 6-7 restriction), SD5 · Metrics: `drace_test_s17.json` · Chain: `plan/checks/run_s17.sh`
  (log `logs/run_s17.log`) · HF (h200): dracd7-test retrieval 12.8 min (~19-21K pool records/s, CPU tokenisation-bound),
  dracd7-test-c0 scoring 853,936 pairs in 2.0 min (7,333 pairs/s, 2x the A100).
- Base full_c0_rgrp_t073 (threshold 0.73). DRACE additions 83,104 (S16 80,800; +2,304, +0.0013 per S1); final 5,811,357
  pairs; candidates 9.35 per S1. Validator PASS; sha1 3adb3960….
- Train evidence: seed<=5 + D10 ranks 6-7 vs seed<=5: +0.00004 (seed5_eval.json). Expectation (d): LB +0.00002 to +0.0001.
- Also: retrieve_job.py now tokenises batches ahead on CPU threads (RETR_PREFETCH, default 16); smoke set top-5 identical
  to the old code (100% overlap). Next channel checked: distilled seed expansion ranks 1-2 has only 158 true pairs among
  185,273 new train pairs (`logs/dseed_prep.log`), ceiling ~+0.00005: not scored.

### 2026-09-27 · ER1 · Rescue S1 left empty with their best sub-0.995 DRACE pair · DROPPED (flat)
- Metrics: `empty_rescue.json` · Script: `plan/checks/empty_rescue.py` · Log: `logs/empty_rescue.log`.
- S16-equivalent 0.98875; 10,495 S1 end with no match, 10,126 of them (96%) truly have no copy. Admitting the S1's best
  DRACE pair at c0 >= T2: 0.98 adds 9 pairs (6 true) +0.00002; 0.8 −0.00008; 0.5 −0.00022. Cross-fitted (both halves
  chose 0.98) +0.00002.
- Learned: empty predictions are already almost always right; there is nothing to rescue there.

### 2026-09-27 · TH3 · Per-country LightGBM threshold (test-like weighted, cross-fitted by S1 half) · DROPPED (flat)
- Metrics: `thr_country.json` · Script: `plan/checks/thr_country.py` · Log: `logs/thr_country.log`.
- Halves chose global 0.73 both times; per-country US 0.73/0.72, India 0.72/0.78 (unstable). Cross-fitted per-country vs
  global −0.00002. Both countries are flat between 0.72 and 0.76 (US 0.98888, India 0.98801).

### 2026-09-27 · N14 · How many rejected name-only true pairs are same-name ties (c0 retrain ceiling) · INFO
- Data: M027 OOF (threshold 0.73, one-owner) joined with rev_MATCH_TRAIN (pool name -> 20 nearest S1 names, same country).
  Log: `logs/ambiguity_probe.log` (log only; no metrics JSON written).
- True rejected name-only pairs 7,582: true S1 is the unique nearest S1 name 6.2%; nearest but tied with another S1 (within
  0.02 cos) 30.7%; another S1 name is closer 63.1%. Accepted true name-only (14,141): unique nearest 70.0%.
- Learned: for ~94% of the largest loss bucket, another S1 looks at least as plausible from the name alone, so a better
  pairwise model (retrained c0) can reach at most ~470 of these pairs (~+0.0003 oracle on train). Not worth the last HF
  credit or the deadline risk; c0 retrain DROPPED for tonight.

### 2026-09-27 · FINAL · S16 promoted to the live model · INFO
- work/match/models now holds full_c0_rgrp at threshold 0.73 (was full_single, 0.69). Rebuild chain for S16:
  `plan/checks/run_s16.sh`. The D10 test chain referenced in D10 as `run_s14.sh` is now `plan/checks/run_d10_test.sh`
  (never run to completion; S14 was built by the SD5 steps, see S14-file).

### 2026-09-27 · TH2 · Threshold under test's list-length distribution (train S1 reweighted, test inputs only) · INFO
- Metrics: `thr_testlike.json` · Script: `plan/checks/thr_testlike.py` · Log: `logs/thr_testlike.log`.
- Mean candidate list length train 5.42, test 6.13; bucket weights 6-7 1.30, 8-9 1.99, 10-14 4.16, >= 15 52.75 (few train S1:
  that bucket is noisy).
- On M027's scale (chosen 0.73 = live 0.70): test-like weighted F0.5 peaks flat at 0.74-0.76 (0.98861), falls from 0.78 on
  (0.82: −0.00013, 0.86: −0.00038). Maps to live 0.71-0.73, i.e. S16's 0.73 is at the upper end of the train-derived optimum.
- Decision: no further threshold raise; S16 is the final unless a train-side rule says otherwise.

### 2026-09-27 · S16-LB · Submission 16 leaderboard: S14 with threshold 0.73, 0.983958 · ADOPTED
- Log: `reports/test_usage_log.md` · File: `submission/output_v16/matching_results.tsv` (sha1 77149494…).
- Result: public LB 0.983958 vs S14 0.983741: +0.00022. Train evidence was flat (TH1 +0.00000), so the gain is test-specific:
  consistent with N04 (longer, more uncertain test lists favour a stricter threshold).
- Caution: stepping the threshold further by LB feedback would be tuning on test (AGENTS rule 5). Any further change must
  come from a train-side rule (e.g. threshold chosen on train S1 whose lists look like test's).

### 2026-09-27 · S16-file · S14 with the LightGBM threshold raised 0.70 -> 0.73 · INFO (upload candidate, coin flip)
- Parent: S14-file, TH1, N04 · Metrics: `drace_test_s16.json` · Log: `logs/run_s16_t073.log` · Run: `work/match/runs/full_c0_rgrp_t073`.
- Test: base pairs 5,728,253 (S14 5,749,212; −20,959, −0.012 per S1); DRACE additions 80,800; final 5,809,053. Validator PASS;
  sha1 77149494….
- Evidence: TH1 +0.03 over the chosen threshold +0.00000 on train; N04 (older model) favoured a higher threshold on lists > 5,
  which test has more of. Expectation (d): LB within ±0.0002 of S14.

### 2026-09-27 · S15-file · S14 with the LightGBM threshold lowered 0.70 -> 0.65 (user request) · INFO (upload candidate, not recommended)
- Parent: S14-file, TH1 · Metrics: `drace_test_s15.json` · Log: `logs/run_s15_t065.log` · Run: `work/match/runs/full_c0_rgrp_t065`
  (copy of full_c0_rgrp, only `decision.threshold` changed).
- Test: base pairs 5,754,915 (S14 base 5,749,212; +5,703, +0.0033 per S1); DRACE additions 80,770; final 5,835,685; 101,111
  empty S1 (S14 101,278). Validator --check-ids PASS; sha1 12736468….
- Expectation: TH1 says a 0.05 lower threshold is −0.00004 [−0.00011, +0.00004] on train (singletons −0.00137). LB flat to
  slightly negative (d).

### 2026-09-27 · TH1 · LightGBM decision threshold under the full S14 pipeline, MATCH_TRAIN · INFO
- Metrics: `thr_sweep_s14.json` · Script: `plan/checks/thr_sweep_s14.py` · Log: `logs/thr_sweep_s14.log`.
- M027 OOF + S14 DRACE additions; reference = M027's chosen threshold 0.73 (full_c0_rgrp's own choice is 0.70).
  0.68 (i.e. −0.05): −0.00004 [−0.00011, +0.00004]; 0.65: −0.00006 [−0.00015, +0.00004]; 0.60: −0.00022; 0.55: −0.00048;
  0.75: +0.00000; 0.78: −0.00005. Lower thresholds lose on singleton S1 (0.65: −0.00186 on that segment).
- Learned: the fitted threshold is already at the F0.5 optimum with DRACE in place; lowering it adds false matches
  that F0.5 punishes, mostly on S1 with no true copy.

### 2026-09-27 · AD1 · Address-only lexical retrieval (binary TF-IDF over address tokens, S1 -> pool top-K) · DROPPED
- Metrics: `addr_eval.json` · Scripts: `plan/checks/addr_retr.py`, `plan/checks/addr_eval.py` · Logs: `logs/addr_retr_train.log`
  (77 s for 180K S1 x 10.3M pool, df <= 3000), `logs/addr_eval.log`.
- Hypothesis (N13): 1,224 missed true pairs have an invented pool name but an address; c0 would admit them.
- Result: of 6,709 true pairs missed by every S14 channel (incl. D10), top-10 recovers 106 at 6.29 new pairs/S1 (true rate
  0.009%); oracle +0.00004. Top-3 recovers 40, oracle +0.00002. Not scored with c0 (no point).
- Learned: the invented-name copies also have corrupted addresses (median address Jaccard 0.29 on missed pairs; examples
  drop house numbers or city); an address-token query sees hundreds of same-street neighbours and ranks the copy low.

### 2026-09-27 · N13 · What the pairs missed by every S14 channel look like · INFO
- Metrics: `miss_profile.json` · Scripts: `plan/checks/miss_profile.py`; category counts and c0 check in `logs/miss_types.log`.
- Missed (MATCH_TRAIN, after lists + untrained e5 S1@5/seed@5 + distilled S1@5): 7,143 true pairs. Pool address empty 5,295;
  invented name (name Jaccard 0) with address 1,224; other address-bearing 436; Indic-script name 188.
- Missed vs found (address-bearing): median address Jaccard 0.29 vs 0.67; shared digit token 68% vs 89%.
- c0 on in-list invented-name pairs with address: 80.7% of 22,688 true at >= 0.995, 8 of 32,347 false — c0 can confirm them,
  but neither dense (e5) nor lexical address retrieval (AD1) finds them.

### 2026-09-27 · S14-LB · Submission 14 leaderboard: S13 + seed expansion ranks 3-5 0.983741 · ADOPTED
- Log: `reports/test_usage_log.md` · Metrics: `drace_test_s14.json`, `seed5_eval.json` · File:
  `submission/output_v14/matching_results.tsv` (sha1 a81c3cd5…).
- Result: public LB 0.983741 vs S13 0.983604: +0.00014. Train evidence +0.00028; transfer ≈ 0.5x, unlike S12/S13 (2.1-2.3x).
  Test got 0.66x train's added pairs per S1 (+0.0039 vs +0.0059), which explains most of the shortfall.
- Learned: seed-expansion depth does not carry the India-heavy test bonus that distilled S1 retrieval did; do not assume
  a 2x transfer for retrieval changes by default.

### 2026-09-27 · S14-file · S13 + untrained-e5 seed expansion ranks 3-5 (SD5) · INFO (upload candidate)
- Parent: S13-file, SD5 · Metrics: `drace_test_s14.json` · Log: `logs/run_s14_seed5.log` · HF: seed5-test-c0 (1,944,039 pairs).
- Same matcher and base as S12/S13 (full_c0_rgrp); DRACE merge at c0 >= 0.995 with extra sets DRACD_TEST, DRACD5_TEST,
  SEED5_TEST. DRACE additions 80,783 (S13 74,017; +0.0039 per S1 vs +0.0059 per S1 on train); final 5,829,995 pairs;
  candidates 8.86 per S1. Official validator --check-ids PASS; sha1 a81c3cd5….
- Expectation (d): train +0.00028; test adds 0.66x train's per-S1 rate; S12/S13 transferred 2.1-2.3x. LB ~0.9838-0.9842.
- Name-only check (`logs/nameonly_probe.log`, M027 OOF): 121,154 in-list name-only pairs, 17.9% true, p_cal AUC 0.9718;
  7,582 true rejected (median p_cal 0.30). The highest-scored false pairs include exact-name copies ("Williams Patriot",
  "Pediatric Vanguard Partners LLC"), i.e. same-name S1 competing for one pool record: not separable from the name alone.

### 2026-09-27 · N12 · Oracle upside of every unused retrieval channel after S13 + seed ranks 3-5 · INFO
- Metrics: `channel_oracle.json` · Script: `plan/checks/channel_oracle.py` · Log: `logs/channel_oracle.log`.
- Calibration: untrained seed ranks 3-5 oracle +0.00050, realised +0.00028 (ratio 0.56).
- Remaining oracle (accept only the new true pairs): untrained seed 6-10 +0.00022 at 10.9 new pairs/S1; seed 6-20 +0.00045
  at 33.5; distilled seed 1-5 +0.00019 at 5.6; distilled S1 6-7 +0.00011 at 1.7; distilled S1 6-10 +0.00017 at 4.4;
  untrained S1 6-10 +0.00019 at 3.4; untrained S1 6-20 +0.00043 at 12.1.
- Learned: after seed ranks 3-5, no single cached channel has a realisable train gain above ~+0.0001 per ~2M test pairs.
  0.985 on the LB (+0.0014 over S13; ~+0.0006 train at the S12/S13 transfer) is not reachable from retrieval depth alone.

### 2026-09-27 · SD5 · Untrained-e5 seed expansion widened from rank 2 to rank 5 (S14 candidate) · ADOPTED (train gate)
- Parent: S13-file · Metrics: `seed5_eval.json` · Scripts: `plan/checks/seed5_eval.py`, `plan/checks/seed5_test_pairs.py`
  · Logs: `logs/seed5_eval.log`, `logs/seed5_test_pairs.log`. Train pairs were already c0-scored (GATE2_TRAIN); test
  retrieval already existed (drace_test_topk, seed ranks <= 5), so only test scoring was new (HF seed5-test-c0).
- Result vs S13-equivalent 0.98846: seed<=3 +0.00013 [+0.00011, +0.00014]; seed<=5 +0.00028 [+0.00026, +0.00031]
  (added 5,621 -> 6,691, 6,553 true); seed<=5 + D10 +0.00033 [+0.00030, +0.00036]; seed<=5 + D10 ranks 6-7 +0.00032.
- Test: 1,944,039 new pairs to score (seed rank 3: 402,644; 4: 618,633; 5: 922,762).
- Learned: copy-to-copy (seed) expansion recovers more than widening the S1 query: seed ranks 3-5 give 3x D10's gain.

### 2026-09-27 · D10 · Distilled-e5 S1 query widened from top-5 to top-10 (S14 candidate) · ADOPTED (train gate)
- Parent: S13-file, R1 · Metrics: `d10_eval.json`, `deep_probe.json` · Scripts: `plan/checks/d10_prep.py`,
  `plan/checks/d10_eval.py`, `plan/checks/deep_probe.py`; HF d10-train-c0 (855,825 pairs) · Logs: `logs/d10_eval.log`,
  `logs/deep_probe.log`. Test chain: `plan/checks/run_s14.sh` (log `logs/run_s14.log`).
- Hypothesis: 29.6% of the true pairs missed by every channel belong to S1 with > 5 true pool records (`admit_sweep.json`);
  a top-5 S1 query cannot return them all. Oracle (accept only the new true pairs) for distilled S1@10 +0.00045, @20
  +0.00063, @50 +0.00087 (`deep_probe.json`).
- Result: distilled ranks 6-10: 863,099 scored pairs (1,244 true); at c0 >= 0.995, 987 pass (928 true, precision 0.94).
  S13-equivalent 0.98846 -> +0.00010 [+0.00008, +0.00012] (US +0.00004, India +0.00019, singleton +0.00000); candidates
  3.87 -> 8.65 new per S1. Adaptive depth (only S1 with >= n S13 acceptances) is strictly worse: n=2 +0.00007, n=3 +0.00006,
  n=5 +0.00004.
- Learned: c0 realises 22% of the S1@10 oracle. @20 would cost ~24M test pairs to score (~110 min on one A100): out of budget.
- Next: S14 = S13 + distilled ranks 6-10 on test. Expected LB (d, using the S12/S13 2.1-2.3x transfer): +0.0002, i.e. ~0.9838.

### 2026-09-27 · N11 · Retrieval-side research sweep: reverse / iterative / name-only retrieval, c0 admission boundary · INFO
- Metrics: `admit_sweep.json`, `rev_name_probe.json`, `deep_probe.json` · Scripts: `plan/checks/admit_sweep.py`,
  `plan/checks/rev_name_probe.py`, `plan/checks/iter_eval.py` (I1, `logs/iter_eval.log`; stopped before writing its JSON,
  numbers below are from the log only).
- c0 admission (DRACE T sweep 0.9-0.999 on the S13 config): flat. Best in-sample T 0.99 +0.00002; cross-fitted (T chosen
  per S1 half) vs 0.995 −0.00001. DROPPED.
- Name-only (empty pool address) pairs: 64% of the 8,755 true pairs missed by every channel. c0 cannot confirm them: of
  85,306 name-only DRACE pairs (485 true, AUC 0.971), only 2 true have p >= 0.9 (9 false). So more name-only recall
  (name-only dense retrieval, reverse pool->S1 name retrieval) cannot pass the admission gate. Reverse-name top-20 covers
  2,372 of 5,594 missed name-only records; the true S1 is rank 1 for 175. DROPPED (retrieval); the bottleneck is scoring
  evidence for name-only pairs, same conclusion as N10.
- Iterative retrieval (I1, profile vector = S1 + mean of accepted records, and hop-2 from DRACE records, distilled e5):
  oracle at top-5 +0.00030 at 0.62 new cand/S1; top-2 +0.00010. Below the plain deeper S1 query (D10 oracle +0.00045).
  DROPPED.

### 2026-09-27 · O1 · c0 p_ce override / veto / logit blend on in-list pairs (+ per-source cap, D10 cost gates), MATCH_TRAIN · DROPPED (flat)
- Metrics: `c0_override.json` · Script: `plan/checks/c0_override.py` · Logs: `logs/c0_override.log`; ad-hoc probes (no
  metrics file, log only) `logs/probe_struct.log`, `logs/d10_cosgate.log`.
- Setup: S13-equivalent base as admit_sweep (M027 OOF, threshold 0.73 + one-owner, then DRACE at c0 >= 0.995): 0.98846.
  Rules: accept rejected pairs with p_ce >= A and p_cal >= B; zero accepted pairs with p_ce <= V and p_cal < C; logit
  blend (1-w)·logit(p_cal) + w·logit(p_ce) with threshold sweep. Rule chosen on one S1 half, scored on the other.
- Result: best in-sample +0.00001 [−0.00004, +0.00005]; cross-fitted −0.00001. Only 35 rejected pairs have p_ce >= 0.99
  (57% true); no accepted pair has p_ce <= 0.01. LightGBM already absorbs c0 on in-list pairs.
- Per-source structure: true copies per S1 range 0-5 in S2 and 0-6 in S3, widely spread; no pool record has two true S1.
  No count cap to exploit.
- D10 cost gates (855,825 D10 train pairs, 434 true): distilled cosine does not separate (keep 95% of true → keep 94.5% of
  pairs). Rank does: ranks 6-7 hold 265 true (61%) in 330,215 pairs (38%).
- Learned: stage-3 decision rules are exhausted (threshold, expected-F, twin rescue, empty-address threshold, veto, blend,
  c0 override all flat). Remaining gains are on the retrieval side.
- Next: if D10 test scoring cost or time is the constraint, score S1 ranks 6-7 only.

### 2026-09-27 · I1 · Iterative retrieval: profile query and hop-2 recursion (distilled e5), MATCH_TRAIN · PARKED (weak)
- Metrics: `iter_eval.json` · Scripts: `plan/checks/iter_prep.py`, `ce/hf_job/iter_job.py` (HF a100, 14 min), `plan/checks/iter_eval.py`
  · Log: `logs/iter_eval.log`. Refs: Whang et al. SIGMOD 2009 (iterative blocking); ANCE-PRF (CIKM 2021) and Li et al.
  TOIS 2023 (dense pseudo-relevance feedback).
- Setup: S13-equivalent on 180K MATCH_TRAIN S1 (M027 + DRACE channels at c0 >= 0.995; F0.5 0.98846; accepted set built
  from model decisions only). Profile query = normalize(e5(S1) + mean e5(accepted records)), 169,505 S1; hop-2 = the
  5,621 DRACE-added records query their neighbours. A pair counts as new only if no current channel retrieves it
  (stage-2 lists, untrained S1@5, seed@2, distilled S1@5). 8,755 true pairs are missed by every channel (64% name-only,
  49% India).
- Result (new pairs per S1, true rate, missed pairs recovered, oracle F0.5): profile top-5 0.56, 0.81%, 814 (9.3%),
  +0.00030; profile top-10 4.33, 0.19%, 1,504 (17.2%), +0.00049; hop-2 top-10 0.19, 0.85%, 297 (3.4%), +0.00009;
  both top-5 0.62, 0.82%, 914, +0.00032. Recovered pairs are 75% India; name-only recovered 35 of ~5,600 at top-5.
- Learned: confirmed records add little new retrieval evidence beyond the S1 + copy channels. The hypothesis that a
  profile reaches name-only copies fails (a name-only record has no more in common with the profile than with the S1);
  the small gain is India transliteration, overlapping the distilled retriever. Hop-2 recursion has almost nothing to
  start from (DRACE adds 0.03 records per S1). Ceiling at a reasonable candidate cost (+0.6 per S1) ≈ +0.0003 internal
  before precision losses.
- Next: not run on test with the remaining ~$1.4 unless a cheap c0 precision check (~$0.1) shows most of the oracle survives.

### 2026-09-27 · S13-LB · Submission 13 leaderboard: distilled-e5 S1 query top-5 0.983604 · ADOPTED
- Log: `reports/test_usage_log.md` · Metrics: `drace_test_s13.json`, `gate2_union.json` · File:
  `submission/output_v13/matching_results.tsv` (sha1 e18433d2…).
- Result: public LB 0.983604 vs S12 0.983064: +0.00054. Train evidence for the change: +0.00026 (variant C vs B);
  transfer ≈ 2.1x, the same pattern as S12 (retrieval-side gains, mostly India, exceed their train size on test).
  Cost: candidates 7.22 -> 7.73 per S1.
- Learned: the distilled retriever keeps paying as its depth grows; retrieval is still the lever that transfers best.

### 2026-09-27 · S13-file · S12 with the distilled-e5 S1 query widened to top-5 (R1 variant C) · INFO (upload candidate)
- Parent: S12-file · Metrics: `drace_test_s13.json` · Chain: `plan/checks/run_s13.sh` (log `logs/run_s13.log`) · HF:
  dracd5-test retrieval (distilled e5, test S1 top-5) + dracd5-test-c0 scoring (892,479 new pairs, 4.0 min).
- Same matcher and base as S12 (full_c0_rgrp). New S1@3-5 pairs: 892,479 (0.515 per S1); 15,898 at c0 >= 0.995
  (India 13,052, US 1,539, France 1,307). DRACE additions 74,017 (S12 58,315); final 5,823,229 pairs; candidates 7.73
  per S1 (S12 7.22). Predicted pairs per S1 vs S12: India +0.016, US +0.002, France +0.005. Validator PASS; sha1 e18433d2….
- Train evidence: variant C vs B +0.00026 (gate2_union.json: C +0.00075, B +0.00049 over current DRACE).

### 2026-09-27 · S12-LB · Submission 12 leaderboard: S11 + group profile + distilled-e5 DRACE S1@2 0.983064 · ADOPTED
- Log: `reports/test_usage_log.md` · File: `submission/output_v12/matching_results.tsv` (sha1 5952d600…).
- Result: public LB 0.983064 vs S11 0.981623: +0.00144. Internal expectation ≈ +0.0006 (matcher +0.00013 on the
  calibration slice, R1 variant B +0.00049 on train). Transfer ≈ 2.3x — the distilled-retrieval additions are 92% India
  on test (7,887 of 8,545), and test has more India S1 (47% vs 40% of train); retrieval-side gains keep transferring
  above their internal size (S10 too).
- Next: S13 = distilled S1 query widened to top-5 (R1 variant C, +0.00075 vs +0.00049 on train, +1.8 cand/S1 on train).

### 2026-09-27 · S11-LB · Submission 11 leaderboard: S10 + reverse-name ownership + nfreq 0.981623 · ADOPTED
- Log: `reports/test_usage_log.md` · File: `submission/output_v11/matching_results.tsv` (sha1 49ac8e49…).
- Result: public LB 0.981623 vs S10 0.98135: +0.00027. Internal: calibration slice +0.0003 over S9, 180K OOF +0.00037
  (M035). Transfer ≈ 0.75-0.9x; the train-vs-test shift of the reverse features (nearest-name rate 24% vs 18%) did
  not hurt.
- Next: upload S12 (adds group profile + distilled-e5 S1@2 retrieval).

### 2026-09-27 · S12-file · full_c0 + rev + nfreq + group profile (grp_full), DRACE + distilled-e5 S1@2 · INFO (upload candidate)
- Parent: S11-file, M036, R1 · Metrics: `match_train_full_c0_rgrp.json`, `exp_predict_TEST_full_c0_rgrp.json`,
  `drace_test_s12.json`, `firstpass_oof.json` · Chains: `plan/checks/run_s12_mac.sh`, `plan/checks/run_s12_hf.sh`,
  merge `logs/merge_s12.log` · HF: dracd-test retrieval (distilled e5, test S1 top-2) + dracd-test-c0 scoring.
- Group features for training rows come from a 5-fold first pass over all 4,544,045 training rows (folds 989-1,156 of
  12,000, none truncated); TEST group features from S9's predictions.
- Matcher: best_iteration 1190/12000 (converged). Calibration slice (same 83,523 S1 as S9/S11): 0.98643, +0.00477
  [0.00442, 0.00511] over full_single (S11 +0.00464, S9 +0.00434). Decision threshold 0.70 + one-owner.
- Test: distilled S1@2 retrieval gave 160,492 new pairs (0.093 per S1); 8,545 at c0 >= 0.995 (India 7,887, France 329,
  US 329). DRACE additions 58,315 (S11 49,844); final 5,807,527 pairs; candidates 7.22 per S1. Predicted pairs per S1 vs
  S11: India +0.011, US −0.005, France −0.001. Official validator --check-ids PASS; sha1 5952d600….

### 2026-09-27 · R1 · c0 distilled into multilingual-e5-small for DRACE retrieval (research pack Doc 2 #1) · ADOPTED (variant B)
- Metrics: `gate1_gate1_distil.json`, `gate2_union.json` · Scripts: `plan/checks/distill_e5.py --dump`, `ce/hf_job/distill_job.py`
  (HF a100, 6.8 min training + gate-1 retrieval), `plan/checks/gate2_prep.py --tag gate1_distil`, `plan/checks/gate2_union.py`
  · Model: HF repo `prathameshk421/ber-ce-scoring/e5_distil/` (+ train_meta.json) · Logs: `logs/gate1_distil_eval.log`,
  `logs/gate2_union.log`.
- Training: 150K DEV_REST_REST fold-1 S1 (disjoint from MATCH_TRAIN), 435,454 triplets (S1, true pool record incl. 12,666
  blocking misses, hardest false in-list candidate by c0); loss InfoNCE (t 0.05, in-batch) + MarginMSE on the c0 p margin;
  1 epoch, batch 128, lr 2e-5. MarginMSE 0.85 -> 0.26.
- Gate 1 (MATCH_TRAIN vs full train pool): S1@5 recall of blocking-missed true pairs 21.7% -> 35.5% (India 24.4% -> 47.7%,
  US 17.7% -> 17.7%) at 1.29 -> 2.12 new candidates per S1; seed expansion worse for US (34.5% -> 26.9%).
- Gate 2 (base M027, c0 >= 0.995, one-owner): A = current DRACE +0.00220 vs M027 (1.81 new cand/S1). B = A + distilled
  S1@2: +0.00049 [0.00040, 0.00058] vs A at 2.08 cand/S1 (India +0.00120, US +0.00001, singleton −0.00069). C = A +
  distilled S1@5: +0.00075 [0.00065, 0.00085] at 3.87 cand/S1. c0 AUC on the distilled S1 pairs 0.99987.
- Decision: B (most gain per extra candidate). Next: test-side distilled S1 retrieval (K=2) + c0 scoring, merged into S12.

### 2026-09-27 · V1 · Hard address-number vetoes (research pack Doc 1 #4) on M035 OOF · DROPPED
- Log: `logs/veto_probe.log`. Accepted pairs hit by a veto are almost all true: digit conflict 27,669 pairs precision 0.993
  (Δ −0.02086); house-number mismatch 87,696, 0.996 (−0.07835); long-number mismatch 65,674, 0.993 (−0.04372); digit
  conflict & name >= 0.95 20,823, 0.994 (−0.01419).
- Learned: the generator corrupts numbers on true copies (e.g. 4900 vs 490), so number conflict is weak evidence; LightGBM
  already weighs it. Also noted: greedy one-owner is already the optimal assignment (no S1 capacity limit), so the
  research pack's Hungarian / optimal-transport step has nothing to gain.

### 2026-09-27 · S11-file · full_c0 + reverse-name ownership (rev) + nfreq, then DRACE · INFO (upload candidate)
- Parent: S9-file, M031, M034, M035 · Metrics: `match_train_full_c0_rev.json`, `exp_predict_TEST_full_c0_rev.json`,
  `drace_test_s11.json` · Chain: `plan/checks/run_submission_s11.sh` (log `logs/run_submission_s11.log`) · Density check:
  `logs/s11_density_check.log`.
- Change: same training rows / seeded split as full_c0 (REST_FOLD=1, DEV_REST_180K excluded) + groups `rev`, `nfreq`;
  DRACE merge unchanged (S1@5 + seed@2 pairs from S10, c0 >= 0.995, pool records owned by the new base excluded).
- Result: calibration slice (same 83,523 S1 as S9): +0.00464 [0.00429, 0.00499] over full_single vs S9's +0.00434
  (≈ +0.0003 over S9 on identical S1). Test: 5,752,290 base pairs (S9 5,739,077) + 49,844 DRACE = 5,802,134; 7.13
  candidates per S1. Official validator --check-ids PASS; sha1 in the chain log.
- Shift check: extra name-only accepted pairs per S1 +0.0032 on train (their precision 0.967 -> 0.971) vs +0.0044 on
  test (1.4x). Worst case if all extra test acceptances were wrong ≈ −0.0003.
- Supporting runs: M031 rev +0.00028 [0.00016, 0.00039] (band +0.00099); M034 nfreq +0.00020 [0.00009, 0.00031];
  M035 both +0.00037 vs M027; M033 specialists +0.00021 (parked: needs full-scale 5-fold specialists).

### 2026-09-27 · D1 · DRACE-specific LightGBM scorer vs fixed c0 >= 0.995 (Phase 8), MATCH_TRAIN · DROPPED (flat)
- Metrics: `drace_calib.json` · Script: `plan/checks/drace_calib.py` · Log: `logs/drace_calib.log`.
- Change: on the pairs test used (S1 rank <= 5 or seed rank <= 2; 309,681 pairs, 5,050 true), LightGBM on c0 p/logit,
  cosines, ranks, source flags, number of seed paths, best seed p_cal, name/address token-set, empty address; trained on
  one S1 half, scored on the other, threshold chosen on the scoring model's own half.
- Result: AUC c0 0.99962, model 0.99951. Rule vs M027 +0.00220; model vs M027 +0.00219; model vs rule −0.00001
  [−0.00004, 0.00002].
- Learned: retrieval provenance adds nothing over c0 on DRACE pairs; keep the fixed threshold.

### 2026-09-27 · M032 · X3 sibling group retested on the c0 base · PARKED (small)
- Parent: M027 · Metrics: `exp_M032.json` · Log: `logs/exp_M032.log` · Folds 410/359/562/250/257 of 3000.
- Result: OOF 0.98563, Δ +0.00012 [0.00002, 0.00021]; band +0.00028 [−0.00008, 0.00063]; singleton −0.00098 [−0.00176,
  −0.00010] (inside the −0.002 guard); US +0.00018, India +0.00002.
- Learned: c0 absorbed most of X3's old +0.00074. Not worth computing sib at full scale for the last slot.

### 2026-09-27 · N10 · S10 loss breakdown on train + two cheap probes (name-twin rescue, empty-address threshold) · INFO
- Metrics: `loss_s10.json` · Scripts: `plan/checks/loss_s10.py`; probes in `logs/twin_probe.log`, `logs/seg_threshold_probe.log`.
- S10-equivalent (M027 OOF + DRACE@0.995) on 180K MATCH_TRAIN S1: F0.5 0.98801. Oracle gain by bucket: FN_list (true pair
  in list, LightGBM rejects) 9,878 pairs +0.00546 (77% empty pool address); FN_never (found by nothing incl. e5 top-50)
  4,900 +0.00271 (73% empty); FP_lgb 1,395 +0.00192; FN_retr50 (e5 rank 6-50) 3,554 +0.00162; FN_drace (retrieved, c0 <
  0.995) 1,216 +0.00049; FP_drace 87 +0.00011.
- Name-twin rescue (accept a rejected candidate whose normalised name equals an accepted match's): precision 0.084, every
  variant negative (−0.00031 to −0.04347); 0 empty-address candidates are exact twins (copies carry altered names). DROPPED.
- Separate threshold for empty-address pairs: cross-fitted −0.00010 [−0.00016, −0.00003]. DROPPED.
- Learned: the largest remaining loss is empty-address pairs the matcher cannot separate; decision-rule tweaks do not help,
  only better evidence (cross-encoder quality on name-only records) or more retrieval can.

### 2026-09-27 · S10-LB · Submission 10 leaderboard: S9 + DRACE 0.98135 · ADOPTED
- Log: `reports/test_usage_log.md` · Metrics: `drace_test.json`, `gate2_c0.json`, `match_train_full_c0.json` ·
  File: `submission/output_v10/matching_results.tsv` (sha1 d6eb3ab2…).
- Result: public LB 0.98135 vs S7 0.976205: +0.00515. Internal expectation ~0.9885 (S9 slice 0.98600 + G2 +0.0025), so
  the internal→LB gap shrank from −0.0093 (S7) to ~−0.0071. S9 alone was not uploaded, so the split of the +0.00515
  between c0 (S9) and DRACE is not measured; DRACE's train gain alone (+0.0025) is about half.
- Learned: recall-side changes (DRACE: dense retrieval + copy expansion + cross-encoder filter) transfer to test at least
  as well as internal numbers predict; the 0.98 target is met.
- Next: keep the last slot as a reserve; document the pipeline for the final review (candidate_pairs.tsv 7.13 per S1).

### 2026-09-27 · S10-file · DRACE on TEST on top of S9 (S1@5 + seed@2, c0 >= 0.995) · INFO (upload candidate)
- Parent: S9-file, G2 · Metrics: `drace_test.json` · Scripts: `plan/checks/drace_test.py`, `plan/checks/run_drace_test.sh`,
  `plan/checks/run_drace_score_merge.sh` · HF Jobs: drace-test-retrieval (a100, 35.3 min), drace-test-c0-v2 (a100, 8.2 min);
  first scoring attempt failed in 59 s (score_job required a label column; fixed). Logs: `logs/run_drace_test*.log`,
  `logs/run_drace_score_merge.log`.
- Retrieval: 9,969,589 test pool records, 7,471,621 queries (1,732,544 S1 + 5,739,077 S9 accepted pairs as seeds), per
  country; 10,286,575 retrieved -> 1,735,134 new pairs (1.00 per S1) after removing in-list pairs and pool records S9
  already assigns. c0 p_ce: median 6e-5, 99th percentile 0.99994 (bimodal).
- Added at p_ce >= 0.995 with one-owner: 49,859 pairs (0.0288 per S1; train G2 at 0.995: 0.0307 per S1). By country:
  India 36,408 (0.045 per S1), France 7,294 (0.028), US 6,157 (0.009) — same order as train (India gains most).
  Final 5,788,936 pairs (S9 5,739,077); 101,305 empty S1; candidates 7.13 per S1 (S9 6.13).
- File: `submission/output_v10/matching_results.tsv` (sha1 d6eb3ab2…), `candidate_pairs.tsv` includes every scored
  DRACE pair; official validator --check-ids PASS.
- Expectation (d): internal ~0.9885 (S9 slice 0.98600 + G2 +0.0025); LB ~0.979-0.981 with the historical −0.009 gap.

### 2026-09-27 · TC · Transitive-consistency cleaning (TransClean-style) on M027 + DRACE, MATCH_TRAIN · DROPPED (flat)
- Parent: G2 · Metrics: `tc_train.json` · Script: `plan/checks/tc_train.py` · HF Job tc-train-c0 (a100, 8.7 min scoring) ·
  Log: `logs/tc_train.log`. Paper: TransClean, arXiv 2506.04006 (+24.42 F1 on their multi-source data, (b)).
- Change: for S1 with >= 3 predicted pool records, c0 scores every ordered pool-pool pair among them (1,863,832);
  sibling score = mean two-order p; drop a record when sibling score < tau and own score < cap; cross-fitted by S1 half.
- Result: base 0.98801; 527,310 records with >= 2 siblings, only 0.25% FP. c0 pool-pool AUC 0.831 (vs 0.999 S1-pool);
  sibling-score AUC true-vs-FP 0.892. Every rule drops more TP than FP (tau 0.3: 535 FP, 6,668 TP). Cross-fitted Δ
  +0.00000 [−0.00002, 0.00003] (both halves chose tau 0.05, cap 0.9).
- Learned: the base has too few FPs inside multi-match sets for consistency cleaning to pay, and c0 does not transfer
  to pool-vs-pool comparison. Not applied to test (saved ~$2 of scoring).

### 2026-09-27 · M030 · MLP-PLR (numeric periodic embeddings, tuned schedule) instead of / blended with LightGBM · DROPPED (flat)
- Parent: M027 · Metrics: `exp_M030.json`, `exp_M030b.json` (w 0.5, fixed before the run), `exp_M030b25.json`,
  `exp_M030b75.json` · Script: `plan/checks/nn_exp.py --run M030 --arch plr --lr 1e-3 --wd 1e-4 --batch 4096
  --max-epochs 30 --patience 5 --cosine` (laptop MPS, 600 s) · Log: `logs/nn_exp_M030.log` · Models: `work/match/runs/M030/models/`.
- Hypothesis: M029 was undertrained (lr 2e-3 fixed, patience 2, fold 3 stopped at epoch 4); PLR embeddings (Gorishniy
  et al. 2022) close most of the MLP-vs-GBDT gap on tabular data, and a near-equal NN might help a blend.
- Result: epochs 14-24 of 30 (none hit the cap); early-stop logloss 0.0277-0.0289 vs LightGBM 0.0261-0.0282. Alone 0.98528,
  Δ −0.00023 [−0.00038, −0.00008]; blend 0.5 0.98559, Δ +0.00008 [−0.00002, 0.00018]; 0.25 +0.00005 [−0.00001,
  0.00012]; 0.75 −0.00003 [−0.00016, 0.00009]. Correlation with LightGBM raw p 0.99887.
- Learned: a properly trained NN nearly matches LightGBM but learns the same function from the same 69 features; blending
  adds nothing measurable. Model family is not where the remaining loss is (third confirmation after XGB-BLEND, M029).

### 2026-09-27 · M029 / M029b · MLP instead of / blended with LightGBM on M027's setup · DROPPED
- Parent: M027 · Metrics: `exp_M029.json`, `exp_M029b.json` · Script: `plan/checks/nn_exp.py` (laptop MPS, 131 s) ·
  Log: `logs/nn_exp_M029.log` · Models: `work/match/runs/M029/models/` (mlp_fold*.pt, prep_fold*.pkl).
- Change: MLP 69 -> 256 -> 256 -> 128 -> 1 (SiLU, dropout 0.1, AdamW 2e-3, batch 2048), quantile-normal inputs + NaN
  indicators fit per fold on fit rows; early stop on M027's early-stop S1 (patience 2). Same folds, isotonic, decision.
- Result: MLP alone 0.98441, Δ −0.00111 [−0.00128, −0.00094]; 0.5/0.5 raw-probability blend 0.98533, Δ −0.00019
  [−0.00031, −0.00007]. Early-stop logloss per fold 0.0291–0.0334 (fold 3 stopped at epoch 4). Correlation with
  LightGBM raw p 0.99707.
- Learned: on these engineered features the MLP is worse and too correlated with LightGBM to help a blend (same finding
  as XGB-BLEND). No neural matcher in the submission.

### 2026-09-27 · G2 · Gate 2: c0-filtered dense-retrieval pairs added to M027 (recall rescue), MATCH_TRAIN · ADOPTED (for the final submission)
- Parent: M027, G1 · Metrics: `gate2_c0.json` · Scripts: `plan/checks/gate2_prep.py`, `plan/checks/gate2_eval.py`,
  HF Job gate2-c0 (a100-large, 4.6 min scoring; MATCH_VALID AUC 0.99897) · Log: `logs/gate2_eval.log`.
- Hypothesis: G1 recovers 44% of missed true pairs at S1@5 + seed@5 but at 0.7% positive rate; c0 (AUC 0.999 on list pairs)
  may still separate them because retrieval look-alikes differ in ways a cross-encoder reads (numbers, addresses).
- Change: add a new pair when p_ce(c0) >= t (and cos >= c); one-owner kept (pool records M027 already accepts are not
  added; a pool record wanted by several S1 goes to the highest p_ce). Threshold cross-fitted: chosen on one hash half
  of the S1, scored on the other, swapped. c0 never trained on a MATCH_TRAIN S1.
- Result: 983,236 new pairs -> 928,868 after one-owner (6,662 true); c0 AUC on them 0.9996. Cross-fitted: F0.5 0.98552 ->
  0.98802, paired Δ +0.00251 [0.00239, 0.00264], 5,031 S1 changed; both halves chose t 0.99, c 0. US +0.00105 [0.00096,
  0.00115], India +0.00469 [0.00443, 0.00498], band +0.00253 [0.00232, 0.00274], singleton −0.00049 [−0.00098,
  −0.00010] (inside the −0.002 guard). At t 0.99: 5,660 added, TP 5,542, FP 118 (97.9% precision). Plateau: every t from
  0.9 to 0.998 gives +0.00235 to +0.00251; the cosine filter never helps. 81% of G1's +0.00309 oracle.
- Learned: the largest single gain of the project so far, on the loss bucket (stage-1 misses) no earlier change touched.
  The cross-encoder is precise enough on retrieval look-alikes that a plain threshold works; no LightGBM retrain needed
  for this gain.
- Caveat: test lists are denser (5.75 vs 4.68 pool records per S1) and France is unseen; FP rate on test may be higher.
  The plateau allows a stricter test threshold (0.995: +0.00249 internal) at almost no internal cost.
- Next: same pipeline on TEST on top of S9 (seeds = S9's accepted pool records), t 0.995.

### 2026-09-27 · G1 · Gate 1: dense retrieval recall of missed true pairs (multilingual-e5-small, train side) · INFO (pass)
- Metrics: `gate1_gate1.json` · Scripts: `plan/checks/gate1_prep.py`, `ce/hf_job/retrieve_job.py` (HF a100-large, 15.6 min,
  ~$0.65), `plan/checks/gate1_eval.py` · Log: `logs/gate1_eval.log`.
- Setup: 180,000 MATCH_TRAIN S1 + 597,934 seed queries (pool records M027 OOF accepts, threshold 0.73 + one-owner) against
  all 10,320,219 train pool records, per country (US 6.19M, India 4.13M), exact cosine top-50. Text "query: name | addr".
- Sanity: M027 F0.5 recomputed 0.98552 (= exp_M027.json); in-list true pairs retrieved by S1 query @50: 97.8%.
- Result: 15,117 of 621,534 true pairs (2.43%) are missing from the final candidate lists. Recovered (new pairs per S1,
  positive rate, oracle F0.5 gain if only the recovered true pairs were added to M027):
  S1@5 21.7% (1.29, 1.4%, +0.00196); S1@10 37.2% (5.80, 0.5%, +0.00284); S1@50 54.0% (45.5, 0.1%, +0.00400);
  seed@5 38.8% (4.77, 0.7%, +0.00220); S1@5+seed@5 44.1% (5.46, 0.7%, +0.00309); S1@10+seed@10 53.8% (19.2, 0.2%,
  +0.00371); S1@20+seed@20 60.7% (47.8, 0.1%, +0.00420). India recall > US at every k.
- Learned: recall passes the 25% bar; copy expansion finds as many missed pairs as the S1 query. The problem moves to
  precision: at 5+5 only 1 new pair in ~140 is true, and F0.5 break-even is ~3.6 TP per FP (N09).
- Next: gate 2 — c0 scores the S1@5+seed@5 new pairs (983K); pick a threshold on train labels; realistic F0.5 gain.

### 2026-09-27 · S9-file · full_c0: 68 live features + ce_rec_c0 p_ce, MATCH_TRAIN + MATCH_VALID + DEV_REST_REST fold 1 · INFO (upload candidate)
- Parent: M027 · Metrics: `match_train_full_c0.json`, `exp_predict_TEST_full_c0.json` · Chain:
  `plan/checks/run_submission_c0_full.sh` (log `logs/run_submission_c0_full.log`; first attempt died with the session
  while waiting, restarted 13:05) · Fold-1 scores: HF Job c0-score-f1-l4 (l4x1, 60.2 min, 984 pairs/s, AUC 0.99895).
- Change: `REST_FOLD=1 EXCLUDE_SETS=DEV_REST_180K train_final_comp.py full_c0 cex_p:ce_rec_c0`; DEV_REST_REST fold 0 (c0's
  training S1) and DEV_REST_180K (ce_rec's) dropped; same seeded split for every other S1.
- Result: best_iteration 1202/12000 (converged), fit 3,864,095 rows. Calibration slice 83,523 S1: 0.98600 vs full_single
  0.98166, paired Δ +0.00434 [0.00402, 0.00467], 3,200 S1 changed (full_ce +0.00375, full_ce_nf +0.00389 on their larger
  slices — not the same S1). Decision: threshold 0.72 + one-owner.
- Test: 5,739,077 pairs; 102,497 empty S1; mean predicted size France 3.22 / India 3.29 / US 3.38. File:
  `submission/output_v9/matching_results.tsv` (sha1 cb77cd04…); official validator --check-ids PASS.
- Learned: c0's gain mostly survives the smaller LightGBM training set (~+0.0004-0.0005 over S7 on this measure).
- Next: upload as slot 1 (expected LB ~0.9766-0.9772, (d)).

### 2026-09-27 · M028 · ce_rec and ce_rec_c0 scores as two columns · DROPPED (flat)
- Parent: M027 · Metrics: `exp_M028.json` · Log: `logs/exp_M028.log` · Folds 402/400/462/309/319 of 3000 (none truncated).
- Hypothesis: disagreement between the two cross-encoders marks uncertain pairs.
- Result: OOF 0.98545 vs M027 0.98552, Δ −0.00007 [−0.00017, 0.00003], 1,168 S1 changed.
- Learned: c0 is ce_rec continued, so the older score carries no extra information. A second CE feature needs a
  model trained on different data or a different architecture to have a chance.

### 2026-09-27 · S8-file · M027 on TEST: 68 live features + ce_rec_c0 p_ce, MATCH_TRAIN 5-fold models · INFO (upload candidate)
- Parent: M027 · Metrics: `exp_M027.json` (OOF 0.98552), `exp_predict_TEST_M027.json` · Chain:
  `plan/checks/run_submission_c0.sh` (log `logs/run_submission_c0.log`) · TEST scores: HF Job c0-score-test
  (a100-large, 49.6 min, 10,613,056 pairs, 0 non-finite, mean p 0.551). exp-predict 291 s.
- Decision: threshold 0.73 + one-owner. Test: 5,725,343 pairs (S6 5,744,247; S7 5,757,382); 102,386 empty S1;
  mean predicted size France 3.22 / India 3.29 / US 3.36, ~6% empty in each. File:
  `submission/output_v8/matching_results.tsv` (sha1 9d5494d7…); official validator --check-ids PASS.
- Caveat: LightGBM trained on 180K S1, S6 on 1,475,458. Learning curve without CE: +0.0014 per doubling
  (`learning_curve.json`); with p_ce unmeasured. S8 can land below S6 on the LB. S9 (full_c0, ~837K S1) is the
  intended upload; S8 is a cheap read of c0 at small scale.

### 2026-09-27 · M027 · ce_rec_c0 score in place of ce_rec (A with the better cross-encoder) · ADOPTED
- Parent: M020 · Metrics: `exp_M027.json` · Log: `logs/exp_M027.log` · Wall: 1.5 min · Folds 422/383/454/394/370 of 3000
  (none truncated). Scores: HF Job c0-score-train (a100-large, `ce/hf_job/score_job.py` CE_ADAPTER=adapter_c0,
  CE_OUT=scores_c0) — MATCH_VALID gate AUC 0.99897, MATCH_TRAIN AUC 0.99893; job cancelled after MATCH_TRAIN to save
  credits (DEV_REST_REST not scored; its fold 0 is c0 training data anyway).
- Hypothesis: M020–M026 put all CE value in the per-pair score; c0 has −21% MATCH_VALID logloss (`ce_rec_c0_eval.json`).
- Change: `cex_p:ce_rec_c0` (69 columns); c0 never saw a MATCH_TRAIN S1.
- Result: OOF 0.98552 vs M020 0.98446, Δ +0.00106 [0.00091, 0.00121], 2,941 S1 changed; band +0.00321
  [0.00266, 0.00374]; US +0.00090 [0.00072, 0.00111], India +0.00130 [0.00106, 0.00152], singleton +0.00255
  [0.00127, 0.00382]. No segment worse. Decision: threshold 0.73 + one-owner.
- Learned: a better per-pair CE score transfers into LightGBM ~10x more than any CE-derived feature family.
  More CE training data is the lever (held-out loss still falling at the end of c0).
- Next: c0 TEST scores (HF c0-score-test running) + training rows for a full-scale model; mini-lockbox MATCH_VALID
  is c0-scored already on Kaggle (work/ce/ce_rec_c0/MATCH_VALID.parquet).

### 2026-09-27 · M020–M026 · CE-derived feature families A–G on top of p_ce (ce_rec) · DROPPED (flat; keep ce_p alone)
- Parent: x1_strsim (68 base columns) for all seven; paired also vs M020 (A) · Commit: 9673aec + uncommitted
  `exp.py` `cex_<family>:<ce>` groups · Metrics: `exp_M020.json` … `exp_M026.json`, `cefe_ablation.json` ·
  Scripts: `plan/checks/run_cefe_ablation.sh`, `plan/checks/cefe_ablation.py` · Log: `logs/run_cefe_ablation.log` ·
  Wall: 20 min (7 × ~3 min) · Folds 377–791 of 3000, none truncated.
- Hypothesis: the cross-encoder scores pairs one at a time but the decision is per S1 list; list-relative CE
  statistics (rank, margins, list shape), CE-vs-stage-2 disagreement and CE × single-evidence products should
  add F0.5 over p_ce alone, most on the uncertain band. Evidence before the run: (c) only.
- Change: A `ce_p` only; B + logit/entropy/confidence; C + rank, top-k flags, gaps to top1/next/previous,
  t1−t2, t2−t3; D + list mean/std/min/max, z, percentile, softmax share and HHI, normalised entropy, counts
  ≥ 0.5/0.7/0.8/0.9, same-name (n_tset ≥ 0.95) count and rank; E + p_ce − p_rerank, |·|, CE rank − stage-2 rank,
  top-1 agreement, margin difference, p_ce − p_ce of stage-2 top1; F + p_ce × n_tset / a_tset / country match /
  a_house / a_num_jacc / p_rerank / 1/stage-2 rank / l_n / empty address / log_chain / source; G = B..F (50 cols).
  p_ce is ce_rec, trained on DEV_REST_180K only, so out-of-sample on every MATCH_TRAIN S1 (step "OOF CE" holds).
- Result (OOF entity F0.5, 180,000 S1; Δ vs A paired, 95% CI):
  A 0.98446 (vs base +0.00639 [0.00610, 0.00669]); B 0.98445 −0.00001 [−0.00010, 0.00009];
  C 0.98442 −0.00004 [−0.00016, 0.00008]; D 0.98449 +0.00003 [−0.00009, 0.00016]; E 0.98449 +0.00003
  [−0.00007, 0.00014]; F 0.98438 −0.00008 [−0.00019, 0.00003]; G 0.98443 −0.00003 [−0.00016, 0.00011];
  M018 (p, rank, gap — the S6/S7 group) 0.98438 −0.00007 [−0.00017, 0.00004].
  Band (25.1% of S1): every family within ±0.00017 of A, all CIs cross 0. Empty-address S1 (47.3%): within
  ±0.00013, all cross 0. hi_name (≥ 2 same-name candidates, 87.8%): within ±0.00009.
  Singletons (5.7%): C +0.00147, D +0.00245, E +0.00137, G +0.00265 (CIs > 0) — paid back elsewhere
  (overall flat); G has FP 1,777 vs A 1,861 and FN 10,571 vs 10,454, i.e. an operating-point shift, not new
  information.
- Redundancy (`cefe_ablation.json` → redundancy): top-gain CE columns in G are products with ce_p
  (ce_x_prr, ce_x_s2rank, ce_x_cmatch) and ce_logit — Spearman 0.86–1.00 with ce_p; the list/rank columns have
  low standalone AUC (0.56–0.70) and small gain. The tree gets the list context from the 14 existing l_* columns
  (built on p_rerank) plus ce_p.
- Learned: all of the CE gain is in the per-pair score. LightGBM with the 68 base columns already rebuilds any
  list-relative view of p_ce it needs; margins, list normalisation, disagreement and interactions add nothing
  measurable (|Δ| ≤ 0.00008). Per-S1 context is not where the remaining loss is.
- Next: no larger-set run for any family (step 8 has no winner). Put effort into p_ce quality: ce_rec_c0 scores
  (HF Jobs c0-score-train / c0-score-test running), then "68 + ce_p(c0)" vs M020 on the same folds. For full-scale
  models, `cex_p` can replace `ce:` (A ≥ M018 by +0.00007, CI crosses 0 — no reason to change S7).

### 2026-09-27 · S7-LB · Submission 7 leaderboard: full_ce_nf (full_ce + nfreq) 0.976205 · ADOPTED
- Log: `reports/test_usage_log.md` · Model metrics: `match_train_full_ce_nf.json` · File: `submission/output_v7/matching_results.tsv`
- Result: public LB 0.976205 vs S6 0.976128: +0.000077. Internal held-out gain +0.00014 (0.98537 → 0.98551), so the LB
  kept ~55% of it. Internal→LB gap 0.98551 − 0.976205 = 0.0093 (S6: 0.0092).
- Learned: nfreq transfers, at about half size; as small as M019 predicted. Gap unchanged: nfreq does not touch the
  test-heavier error types.
- Next: full_ce_nf is the base for submission 8 (ce_rec_c0 p_ce + fold-1 adapter).

### 2026-09-27 · CE-C0 · ce_rec continued on DEV_REST_REST fold 0 (ce_rec_c0) · INFO (standalone win; not yet in a matcher)
- Kernel: `prathameshkinagi/ber-ce-qwen-v2m3` v2 (2×T4, 04:09 → 10:57, 8.97 GPU-h used of the week). Code:
  `ce/train_continue.py`, `ce/kaggle_continue/`. Outputs: `work/ce/kaggle_out/qwen_v2m3_v2/`, copied to
  `work/ce/ce_rec_c0/` (adapter, train_meta.json, MATCH_VALID.parquet). Metrics: `ce_rec_c0_eval.json`.
  v1 FAILED in 7 min (Kaggle peft vs preinstalled torchao 0.10); v2 uninstalls torchao.
- Change: ce_rec LoRA (bge-reranker-v2-m3) continued for 1 epoch on DEV_REST_REST fold 0 (top bit of
  s1_idx·0x9E3779B1; 3,285,477 pairs, 605,841 S1), lr 5e-5 linear decay, batch 32, fp16 + GradScaler on one T4.
  161 pairs/s, 6.48 h, 0 skipped batches, peak VRAM 6.16 GB.
- Result: fold-0 held-out (30,000 pairs) logloss 0.0514 → 0.0410, AUC 0.99835 → 0.99894, still falling at the end.
  MATCH_VALID (108,493 pairs, trained on by neither): AUC 0.99841 → 0.99896, logloss 0.0507 → 0.0402, PR-AUC
  0.99902 → 0.99935. Pool-address-empty subset (13,290 pairs; 75% of full_ce FN_in_list): AUC 0.932 → 0.946,
  PR-AUC 0.713 → 0.754, logloss 0.253 → 0.220. Alias subset AUC 0.992 → 0.994.
- Qwen3-Reranker-0.6B (GPU 0): FAILED at the first training step, CUDA OOM on the 14.5 GB T4 (batch 32, fp32 master
  weights, no gradient checkpointing). Zero-shot held-out AUC with the short prompt: 0.824. No comparison yet.
- Learned: more data of the same kind still helps the cross-encoder (−21% logloss on MATCH_VALID from 3.4× the
  pairs), most on the empty-address rows where full_ce loses most. Standalone win only; adoption needs the matcher run.
- Next: p_ce from ce_rec_c0 needs TEST + MATCH_TRAIN + fold 1 scored (~15M pairs); a fold-1 adapter restores
  fold 0 to LightGBM. Qwen retry needs gradient checkpointing + batch 16.

### 2026-09-27 · S7-file · full_ce_nf: full_ce + nfreq (M019), MATCH_ALL minus DEV_REST_180K · INFO (upload candidate)
- Parent: full_ce · Commit: 9673aec (+ uncommitted exp.py nfreq group) · Metrics: `match_train_full_ce_nf.json`,
  `exp_predict_TEST_full_ce_nf.json` · Chain: `plan/checks/run_submission_ce_nfreq.sh` (log
  `logs/run_submission_ce_nfreq.log`) · Wall: train 1168 s, exp-predict 561 s.
- Change: 71 full_ce features + `nf_pool_name_in_s1`, `nf_pool_name_in_pool`, `nf_s1_name_in_s1`; same seeded fit /
  early-stop / calibration split as full_ce.
- Result: best_iteration 2330/12000 (converged). Calibration slice 147,462 S1: 0.98551 vs full_ce 0.98537 (+0.00014,
  unpaired; paired vs full_single +0.00389 [0.00362, 0.00416] vs full_ce's +0.00375). Decision: threshold 0.67 + one-owner.
- Test: 5,757,382 pairs, 103,178 empty S1. File: `submission/output_v7/matching_results.tsv` (sha1 2dc021d1…);
  official validator --check-ids PASS.
- Learned: nfreq's gain at full scale matches M019 (+0.00018 at 180K). Small.
- Next: upload S7; expect roughly S6 + 0.0001 on the LB (d).

### 2026-09-27 · XGB-BLEND · XGBoost + LightGBM blend on full_ce (ALTERNATIVE_APPROACHES 1.6) · DROPPED (flat)
- Parent: full_ce · Commit: 9673aec (+ uncommitted `plan/checks/train_xgb_blend.py`, `exp.py` xgb_blend hook) ·
  Metrics: `match_train_full_ce_xgb.json` · Log: `logs/train_xgb_blend_full_ce_xgb.log` · Wall: 32 min (XGBoost fit 28 min)
- Hypothesis: a second GBDT implementation (XGBoost hist, lossguide 63 leaves, eta 0.05, same 71 features, same fit /
  early-stop / calibration S1 as full_ce) makes partly independent errors, so averaging raw probabilities should
  raise F0.5 on the band rows. Evidence before the run: (c) only; the repo plan rated it low.
- Result: XGBoost best_iteration 2788/12000 (converged). Calibration slice 147,462 S1, every predictor put through the
  same isotonic + threshold + one-owner procedure (LightGBM re-scored = recorded 0.98537 exactly):
  XGBoost alone 0.98538 (Δ +0.00001 [−0.00007, +0.00009]); blend w=0.5 (fixed before the run) 0.98537
  (Δ +0.0000002 [−0.00007, +0.00007], 585 S1 changed); w=0.25 / 0.75 also inside ±0.00008.
  AUC 0.999485 vs 0.999484; raw-probability correlation 0.99975.
- Learned: on these features the two libraries learn the same function (corr 0.99975), so averaging removes no error.
  The remaining loss is in the features and the recall ceiling, not in LightGBM's variance. A 3-seed LightGBM bag
  (RCA plan G2) should be flat for the same reason — not worth running.
- Model: `work/match/runs/full_ce_xgb/models/` (LightGBM full_ce + xgb_fold0.ubj, w 0.5) is submittable via
  `exp-predict --run full_ce_xgb`, but not built or uploaded: expected LB change ≈ 0.
- Next: none on GBDT swaps/ensembles.

### 2026-09-27 · M019 · nfreq: whole-name frequency in the split's own S1 / pool tables · PARKED (small, positive)
- Parent: M018 · Metrics: `exp_M019.json` · Wall: 11 min · Folds 644/398/555/583/357 of 3000 (none truncated).
- Hypothesis: N09 — among empty-address same-name rows, positive rate 67% when the name is on exactly 1 S1 vs 8% on 4+.
- Result: OOF 0.98456 vs 0.98438, Δ +0.00018 [0.00007, 0.00029]; band +0.00067 [0.00023, 0.00109].
- Learned: the idf features already carry most of the name-frequency signal. Keep for the next full run only if bundled.

### 2026-09-27 · M018 · p_ce group on the MATCH_TRAIN 5-fold protocol · ADOPTED (parent for p_ce-era runs)
- Parent: x1_strsim · Metrics: `exp_M018.json` · Wall: 5 min · Folds 423/589/720/477/473 of 3000 (none truncated).
- Result: OOF 0.98438 vs 0.97807, Δ +0.00631 [0.00603, 0.00662]; band +0.02160. (Full-scale S6: +0.00375, LB +0.00933.)

### 2026-09-27 · N09 · Headroom of full_ce + four probes (what could reach LB 0.98) · INFO
- Metrics: `ce_headroom_full_ce.json`, `ce_headroom_tags.json` · Scripts: `plan/checks/ce_headroom.py`, `ce_headroom_tags.py`;
  probes in logs/{empty_addr_probe,cut_probe,biencoder_probe,tta_probe}.log (scratch scripts, not in repo).
- Slice: 19,985 full_ce calibration S1 in MATCH_TRAIN/MATCH_VALID, F0.5 0.98613. Errors: FP 154, FN_in_list 1,048,
  FN_cut 521 (in K_min 20 list, cut from K5), FN_stage1 1,101. Oracle gains: FN_in_list +0.0053, FN_stage1 +0.0054,
  FP +0.0018, FN_cut +0.0017.
- FN_in_list: 75% pool address empty, 56% name token-set >= 95. Empty-address same-name rows are 28.9% positive
  (6,329 rows); p_ce separates them best (<=0.3: 9%, 0.7-0.9: 70%); exact text twins explain only 41 FN.
- FN_cut probe (CE on 521 cut positives + 3,000 of 301,252 cut negatives): at p_ce >= 0.8, 264 TP for ~100 FP
  (1 sampled negative x100, noisy); 1 FP costs ~3.6 TP under F0.5 here -> about break-even on p_ce alone.
- Bi-encoder probe (multilingual-e5-small, 100K random pool distractors): true record ranked first for 50% of
  FN_stage1 (63% top-3, 75% top-11) vs 91% for found pairs. Full pool is 103x larger: recall at k~50 unmeasured.
- Order-swap TTA: swapped-input AUC 0.521 on 10,126 uncertain MATCH_VALID pairs (model is order-dependent) — dead.
- CE learning curve (friend's log): held-out loss 0.0615 @240K pairs, 0.0537 @480K, 0.049 @970K (still falling).

### 2026-09-27 · S6-LB · Submission 6 leaderboard: full_ce (p_ce feature) 0.976128 · ADOPTED
- Log: `reports/test_usage_log.md` · Model metrics: `match_train_full_ce.json` · File: `submission/output_v6/matching_results.tsv`
- Result: public LB 0.976128 (rank 741) vs S2 0.966797: +0.009331. Internal held-out gain was +0.00375, so the LB gain
  is ~2.5x the internal one. Internal→LB gap 0.98537 − 0.976128 = 0.0092 (S2: 0.0148).
- Learned: first internal gain that transferred in full (S2 kept ~10%, S3 negative). A per-pair text score from a
  multilingual cross-encoder transfers to test; cross-list statistics (comp) did not. Test-heavier error types
  (France, transliteration) are plausibly where p_ce helps most — unmeasured without labels.
- Next: full_ce is the new base. Candidates: p_ce on DEV_REST_180K via a second adapter trained on other S1
  (restores 180K S1 of training rows); cross-encoder recall channel for FN_MISSED (42% of loss, N08).

### 2026-09-27 · S6-file · full_ce: live features + cross-encoder p_ce group, MATCH_ALL minus DEV_REST_180K · INFO (upload candidate)
- Metrics: `match_train_full_ce.json` · Chain: `plan/checks/run_submission_ce.sh` (log `logs/run_submission_ce.log`) ·
  Wall: HF a100-large scoring MATCH_TRAIN + DEV_REST_REST 34 min (~$1.42; AUC MATCH_TRAIN 0.99838), train 564 s,
  exp-predict 297 s.
- Change: 68 live features + `p_ce_rec`, `p_ce_rec_rank`, `p_ce_rec_gap` (exp group `ce:ce_rec`). DEV_REST_180K rows
  dropped (cross-encoder training S1: p_ce in-sample there); same seeded fit / early-stop / calibration split as
  full_single, so the calibration slice is out-of-sample for both models.
- Result: best_iteration 2334/12000 (converged). Calibration slice 147,462 S1: full_ce 0.98537 vs full_single 0.98162,
  paired Δ +0.00375 [0.00348, 0.00401], 5,300 S1 changed. Decision: threshold 0.69 + one-owner.
- Test: 5,744,247 pairs (S2 5,789,591); 103,719 empty S1. File: `submission/output_v6/matching_results.tsv`
  (sha1 c33d044e…); official validator --check-ids PASS.
- Caveat: internal→LB transfer has failed before (S3). p_ce is a per-pair score, not a cross-list statistic like
  comp, so the S3 failure mode (train-vs-test list structure) should not apply — untested until the LB number.
- Next: upload; log the LB score as S6-LB.

### 2026-09-27 · CE-FEAT-smoke · p_ce group in the final trainer, 50-round smoke (MATCH_TRAIN + MATCH_VALID only) · INFO
- Metrics: `match_train_smoke_ce.json` · Script: `EXCLUDE_SETS=DEV_REST_180K,DEV_REST_REST NUM_BOOST_ROUND=50
  train_final_comp.py smoke_ce ce:ce_rec` · Wall: 25 s train, 48 s exp-predict on TEST · Model deleted after the check.
- Purpose: exercise the new EXCLUDE_SETS / `ce:` group path end to end before the full run (not a candidate).
- Result: calibration slice 19,985 S1 (same seeded split as full_single, so out-of-sample for both):
  smoke 0.98383 vs full_single 0.98256, Δ +0.00127 [0.00045, 0.00213]; truncated by design (50/50 rounds).
  p_ce joins 100% of feature rows; standalone AUC MATCH_VALID p_ce 0.99841 vs p_rerank 0.98605.
- Next: full run `full_ce` on MATCH_ALL minus DEV_REST_180K (plan/checks/run_submission_ce.sh).

### 2026-09-27 · S5-file · Cross-encoder only (LoRA bge-reranker-v2-m3, ce_rec), threshold + one-owner · INFO
- Metrics: `ce_submit_ce_rec.json` · Scripts: `ce/hf_job/score_job.py` (HF Jobs a100-large, 49 min, ~$2.05),
  `ce/hf_job/collect.py`, `ce/ce_submit.py submit --ce ce_rec --fit-set MATCH_TRAIN_S20000` · Adapter: Friend B's
  1-epoch LoRA on DEV_REST_180K (branch submission-v3, `cross_encoder_lora/adapter/`).
- Scoring: bf16 on A100, ~3,700 pairs/s; 10,613,056 TEST pairs, 0 non-finite. Format gate: AUC MATCH_VALID 0.99841,
  MATCH_TRAIN_S20000 0.99835 (friend's held-out eval 0.9986). Local M3 MPS fp32 measured 76 pairs/s (not viable).
- Decision (fitted on 20,000 train S1, never on test): p_ce >= 0.79 + one-owner; fit F0.5 0.97650.
  MATCH_VALID (scoring only) F0.5 0.97552 — below M005 on MATCH_VALID (0.98162), above M000 (0.97449).
- Test: 5,637,038 pairs; ~3.2 predicted per S1 and ~6% empty in every country (France, India, US).
  File: `submission/output_v5/matching_results.tsv` (sha1 663f29bf…); official validator --check-ids PASS.
- Expectation: below S2 (0.966797) on the LB; uploaded as a standalone read of the cross-encoder.
- Next: p_ce as a LightGBM feature (MATCH_ALL minus DEV_REST_180K, where p_ce would be in-sample).

### 2026-09-27 · S4-file · comp + num on MATCH_ALL, hybrid file (diagnostic upload candidate) · INFO
- Metrics: `match_train_full_comp_num.json`, `final_hybrid_full_comp_num.json` · Scripts: `train_final_comp.py full_comp_num comp num`,
  `final_hybrid.py full_comp_num` · Wall: 33.5 min train + ~50 min files
- Result: best_iteration 5636/12000 (converged). Calibration slice: full_comp_num 0.98448 vs full_single 0.98161
  (+0.00287 [0.00263, 0.00310]); hybrid (threshold 0.64) 0.98324, +0.00163 [0.00145, 0.00179]. Test: hybrid
  5,817,998 pairs (S2 5,789,591; S3 5,827,052); official validator --check-ids PASS.
  File: `submission/output_v4/matching_results.tsv` (copy of work/outputs/full_comp_num_hybrid/).
- Caveat: S3 (same method without num) scored below S2 on the LB; num failed US→India LOCO (−0.00450). Upload
  only as an A/B diagnostic against S3, not as an expected improvement.

### 2026-09-26 · S3-LB · Submission 3 leaderboard: contention hybrid did not transfer · INFO
- Log: `reports/test_usage_log.md` (upload 23:05) · Model metrics: `final_hybrid.json`, `match_train_full_comp.json`
- Result: public LB shown as 0.966 (rounded, exact pending) vs S2 0.966797 — between −0.0003 and −0.0013, against
  an internal held-out gain of +0.00143 (hybrid) on the MATCH_ALL calibration slice.
- Learned: third internal→LB failure (S2 kept ~10%, S3 negative). Candidates: (1) contention distribution
  shift on contested rows (N07: test US/IN less contested, France 1.17 vs 0.46 mean log lists-per-record);
  (2) hybrid threshold 0.59 vs live 0.69 added 37,461 pairs on test (unmeasured whether FP);
  (3) internal validation does not reproduce the test pool's structure (4.68 vs 5.75 records per S1), so any
  cross-list feature is tuned to train's structure. Internal gains from cross-list features are not evidence
  for the leaderboard until a test-like validation set (RCA D3) exists.
- Next: do not upload the comp+num file; build the D3 densified validation set; re-judge M005/M010/M017 on it.

### 2026-09-26 · N08 · LOCO by feature group (item 14, France proxy) · INFO
- Metrics: `loco_groups.json` · Script: `plan/checks/loco_groups.py` · Wall: ~50 min (paused 8 min for the S3 deadline)
- Change: one LightGBM per direction on one country's MATCH_TRAIN S1 (90% fit / 10% early stop + isotonic),
  x1 threshold 0.67 + one-owner, scored on the other country; paired vs the 68-feature base.
- Result (US→India / India→US): base 0.95268 / 0.97216. +comp +0.00497 / +0.00506; +num −0.00450 / +0.00151;
  +sib +0.00131 / −0.00144; +dinv +0.00045 / +0.00021. Dropping: name_str +0.00039 / −0.00181, idf −0.00720 /
  −0.00274, addr_str −0.01234 / −0.00862, list −0.00062 / −0.00158, counts −0.00077 / −0.00184, context
  (src, country3, pool_indic) +0.00081 / −0.00005. All CIs in the file.
- Learned: contention transfers across countries; the address-number group is country-specific (US-trained
  number rules hurt India). No base family is dead weight except possibly `context` for US→India.
- Next: prefer comp-only for France-heavy uncertainty; if num is kept, consider it only with India+US training
  (already the case) and watch the S3 vs S4 leaderboard difference.

### 2026-09-26 · M017 · Specialists (converged, cap 10000) on top of comp + num · PARKED (needs canary + full-scale plan)
- Parent: M016 · Metrics: `exp_M017.json` · Wall: ~70 min (concurrent jobs)
- Result: OOF 0.98330 vs 0.98211, Δ +0.00119 [0.00098, 0.00141]; band +0.00424; singleton −0.00010 [−0.00196, 0.00186].
  Name specialist converged at 4056-4732/10000; no fold truncated.
- Learned: the specialist/disagreement stack is additive with comp + num. Stacked features need the label-shuffle
  canary (EXPERIMENT_ROUTINE §6) and 5-fold specialist OOF on MATCH_ALL (several hours) before a submission.
- Next: canary on MATCH_TRAIN, then plan the MATCH_ALL stack.

### 2026-09-26 · M015 · Fellegi-Sunter m/u log-likelihood feature (item 12) · DROPPED
- Parent: x1_strsim · Metrics: `exp_M015.json`
- Result: OOF 0.97813, Δ +0.00007 [−0.00008, 0.00022]; band +0.00025 (CI crosses 0).
- Learned: trees already read the same binned agreement pattern from the raw columns.

### 2026-09-26 · S3 · Submission 3 file (comp-only hybrid) · INFO
- Metrics: `final_hybrid.json`, `match_train_full_comp.json` · File: `submission/output_v3/matching_results.tsv`
  (sha1 b84e9e5e…, copy of work/outputs/full_hybrid/)
- Rule: rows whose pool record is in >= 2 test S1 lists take full_comp's p, others full_single's; threshold 0.59,
  one-owner off (chosen on the MATCH_ALL calibration slice). Slice: hybrid 0.98305, full_comp 0.98360, full_single
  0.98161; hybrid − single +0.00143 [0.00125, 0.00160].
- Test: 5,827,052 pairs (submission 2: 5,789,591); contested row share 32.1%; official validate_submission.py
  --check-ids PASS (also PASS for the pure full_comp file). Leaderboard score: pending upload.

### 2026-09-26 · N08 · FP/FN report with raw text, full_single + full_comp, MATCH_ALL calibration slice · INFO
- Parent: none · Metrics: `error_report.json`, `error_report_tags.json` · Wall: 27 min (feature load)
- Hypothesis: none (measurement, asked by the team). Rows: 165,545 train S1 held out from both final models.
- Result: F0.5 reproduced (full_single 0.98161, full_comp 0.98360). full_comp errors: FP 1,648, FN (in list)
  9,928, FN_MISSED (true link never in the candidate list) 14,037. Loss (sum of 1 − per-S1 F0.5 = 2,714) by
  error type per S1: FN_MISSED only 1,130 (42%), FN only 971 (36%), FP only 378 (14%), mixed 234.
  FN in list: 63% pool address empty, 54% name token-set ≥ 95, median p_cal 0.297. FN_MISSED: 59% India,
  26% name token-set < 50, 23% non-Latin text. FP: median p_cal 0.867, 23% disjoint address numbers, 39% name ≥ 95.
- Learned: candidate generation (FN_MISSED) is the largest single loss source on this slice, ~0.0068 F0.5,
  larger than the stage-1 share measured on DEV_B (0.0045, `s1_E6.json`). Not yet split into stage-1 miss vs
  stage-2 cut (K_max 60 / p_min 0.05).
- Next: split FN_MISSED by stage; bi-encoder kNN retrieval channel targets the name<50 / non-Latin part.
  Files: `work/reports/errors/errors_{full_single,full_comp}.parquet`.

### 2026-09-26 · M016 · comp + num combined · ADOPTED
- Parent: M005 · Metrics: `exp_M016.json` · Wall: 15 min (with a concurrent full-data job)
- Result: OOF 0.98211 vs M005 0.98107, Δ +0.00104 [0.00085, 0.00124]; band +0.00368 [0.00302, 0.00439]; singleton
  +0.00098 [−0.00078, 0.00255]. vs x1_strsim +0.00404 (separately +0.00300 and +0.00132: close to additive).
  No fold truncated.
- Next: full_comp_num on MATCH_ALL + hybrid files (`train_final_comp.py full_comp_num comp num`, `final_hybrid.py full_comp_num`).

### 2026-09-26 · M005-full · full_comp: live recipe + comp on MATCH_ALL · INFO (submission-3 candidate)
- Metrics: `match_train_full_comp.json` · Script: `plan/checks/train_final_comp.py` · Wall: 46.7 min
- Change: train_final_single.py unchanged (same seeded fit / early-stop / calibration split, params, cap 12000) plus
  the `comp` group; saved to `work/match/runs/full_comp/models/`, NOT copied to the live dir.
- Result: best_iteration 6524/12000 (converged). Held-out calibration slice (165,545 S1): 0.98360 vs full_single
  0.98161 on the same S1, Δ +0.00199 [0.00178, 0.00220]. Decision: threshold 0.67, one-owner off.
- Learned: the 180K gain carries to full scale at ~2/3 size. full_single's slice score has only slice-level
  one-owner, so part of the Δ is the one-owner coverage gap that test already closes (N05b: ~+0.0009).
- Next: `final_hybrid.py` (contested-only hybrid, N07) and test files.

### 2026-09-26 · N07 · M005 sensitivity to contention density; contested-only hybrid · INFO
- Metrics: `comp_density_check.json` (script `plan/checks/comp_density_check.py`); hybrid and bucket counts ad-hoc from
  `work/match/runs/{x1_strsim,M005}/oof.parquet` + `work/exp/groups/MATCH_TRAIN/comp.parquet` (no metrics file:
  numbers below are to be regenerated by `plan/checks/final_hybrid.py` on the MATCH_ALL slice before use).
- Hypothesis: test US/India rows are LESS contested than train (mean c_log_n_own 0.26 / 0.32 vs 0.46 train; France
  1.17), because the test pool adds unclaimed records. M005 learned contention at train density.
- Result: thinning the other S1 lists to 50% (mean 0.318) makes M005 −0.00229 [−0.00260, −0.00198] vs x1_strsim;
  25%: −0.00689. (Pessimistic: thinning also removes true owners, which test does not.) On M005 OOF, contested
  rows: FP 2280→869, TP +180; uncontested rows: FP 1376→1616, TP +2549 — M005 loosens uncontested rows.
  Hybrid (M005 p on contested rows, x1_strsim p on uncontested, one threshold 0.66 + one-owner): Δ +0.00213
  [0.00194, 0.00232] vs x1_strsim, band +0.00729 (M005 alone +0.00300).
- Learned: the contested-row gain is the robust part; the uncontested-row gain depends on train's orphan density.
- Next: submission 3 as the hybrid of full_comp (contested) and full_single (uncontested); pure full_comp as the
  alternative file.

### 2026-09-26 · M014 · 3-seed LightGBM average (item 17) · PARKED
- Parent: x1_strsim · Metrics: `exp_M014.json` · Wall: ~20 min
- Result: OOF 0.97824, Δ +0.00018 [0.00006, 0.00029]; band +0.00072 [0.00027, 0.00119]; singleton 0.0. No fold truncated.
- Learned: variance reduction is real but small; at MATCH_ALL scale it triples training time (~3 x 35 min).
- Next: only as a final add-on if time allows.

### 2026-09-26 · M013 · Residual LightGBM on the parent's uncertain band (item 10) · PARKED
- Parent: x1_strsim · Metrics: `exp_M013.json` · Wall: 20 s (band rows only)
- Change: second model on rows with x1_strsim OOF p_cal in [0.1, 0.9) (25% of S1), features + parent raw p
  (`par_p`), own isotonic on the band; other rows keep the parent's p.
- Result: OOF 0.97840, Δ +0.00033 [0.00017, 0.00049]; band +0.00131; India +0.00066, US +0.00011 (CI crosses 0).
  Folds converge at <= 152 rounds.
- Learned: a band specialist adds little over one model that already sees every row; the band loss is
  information-limited (M008 said the same). Below the +0.0005 minimum gain.
- Next: retry on top of the adopted feature set only if cheap.

### 2026-09-26 · M012 · One LightGBM per pool source S2 / S3 (items 8, 13) · DROPPED
- Parent: x1_strsim · Metrics: `exp_M012.json` · Wall: 7 min
- Result: OOF 0.97694, Δ −0.00113 [−0.00131, −0.00094]; band −0.00423; every segment worse.
- Learned: splitting halves the data per model; `src` as a split feature already captures source-specific noise.
- Next: none.

### 2026-09-26 · M011 · Name / address / context specialists stacked + disagreement (items 9, 11) · PARKED (truncated)
- Parent: x1_strsim · Metrics: `exp_M011.json` · Wall: ~20 min
- Change: three LightGBMs on name-only (n_*), address-only (a_*), context-only (p_rerank, rank, l_*, src, country3,
  pool_indic, log_chain) columns, 5-fold OOF on the same folds; sp_name, sp_addr, sp_ctx, sp_name_minus_addr stacked
  into the main model (test time: fold average).
- Result: OOF 0.97981, Δ +0.00174 [0.00149, 0.00199]; band +0.00686; India +0.00233, US +0.00134. The name
  specialist hit the 3000 cap in all 5 folds (2989-3000) — its OOF is under-trained, so this number is not the
  converged result. Main model converged at 625-793. Label-shuffle canary for stacked features not yet run.
- Learned: the disagreement/stacking signal is real at this size, but the run is not a clean measurement.
- Next: re-run with a specialist cap >= 8000, plus the canary, then combine with comp + num.

### 2026-09-26 · M010 · Structured address-number group (item 7) · ADOPTED (for combination)
- Parent: x1_strsim · Metrics: `exp_M010.json` · Wall: 8 min
- Change: group `num`: first address number equal / digit-dropped / different (nm_first_state) and log distance,
  any digit-drop pair, postal-like (>=5 digits) and unit-number agreement, number counts, shared numbers in the
  same order. Country-agnostic, no labels.
- Result: OOF 0.97939, Δ +0.00132 [0.00112, 0.00151]; band +0.00497; US +0.00146, India +0.00112, singleton
  +0.00333. nm_first_logdiff gain rank 7, nm_first_state 9. Folds <= 1978/3000.
- Learned: "first number" vs "any number" agreement and the numeric distance separate house-number conflicts that
  a_num_jacc and a_house miss. (X2's regex version measured +0.00047; this one is ~3x.)
- Next: M016 = comp + num.

### 2026-09-26 · M005 · Pool-side contention features (reverse retrieval, items 1/4/5) · ADOPTED
- Parent: x1_strsim · Metrics: `exp_M005.json`, `universe_owner_M005.json`, `match_valid_confirm_M005.json` · Wall: 6.3 min
- Hypothesis: N05 — FP rows are mostly records another S1 owns; in-band rows beaten by another S1's p_rerank by
  >=0.2 match 11% vs p_cal 0.28. Features on how each pool record is contested across ALL S1 candidate lists
  (train side: every train-split S1 with candidates incl. validation-S1 inputs, no labels; test side: all test S1).
- Change: group `comp` (`src/ber/exp.py`): c_log_n_own, c_rank_q, c_margin (own p_rerank − best other S1's),
  c_n_other_ge05, c_best_other, c_q_share_not_best, c_q_share_lose02. 75 features.
- Result: OOF 0.98107 vs 0.97807, Δ +0.00300 [0.00274, 0.00324]; band S1 +0.01115 [0.01020, 0.01208]; US +0.00261,
  India +0.00359, singleton +0.00422. Test-like decision (one-owner over all train S1 for both runs, competitors =
  full_single in-sample p): Δ +0.00186 [0.00162, 0.00208] (re-swept thresholds +0.00174); band +0.00750, non-band
  −0.00003 [−0.00013, 0.00005]. MATCH_VALID (once): 0.98162 vs 0.97848, Δ +0.00313 [0.00242, 0.00384].
  Gain ranks: c_margin 4, c_best_other 5, c_rank_q 13. Folds 1200-1758/3000, none truncated. One-owner no
  longer chosen (the features subsume it).
- Learned: about 60% of the gain is information one-owner cannot give (a soft, learned version of contention plus
  its interaction with list features); all of it sits in band S1, which is 35-57% of test S1 vs 21% of train.
  Test coverage of the signal matches train (32.1% vs 33.6% contested rows).
- Next: retrain with `comp` on MATCH_ALL (1.65M S1, cap >= 12000 per ROUND_CAP_EVIDENCE) for submission 3.

### 2026-09-26 · N07 · Kaggle T4 throughput probe for cross-encoder ce_rec_base (bge-reranker-base, full FT) · INFO
- Parent: none · Metrics: `ce_rec_base_probe_k0.json`, `ce_pairs_manifest.json` · Wall: 4 min on Kaggle 2×T4
- Hypothesis: a 278M cross-encoder full fine-tune plus scoring of MATCH_TRAIN/VALID/TEST fits in the free
  Kaggle quota (30.00 GPU-h/week, measured with `kaggle quota`).
- Change: none (measurement). Pair files: 5 sets, row counts equal to the feature tables, S1 overlap of
  DEV_REST_180K with MATCH_TRAIN / MATCH_VALID / DEV_REST_REST / HOLDOUT_TEST = 0.
- Result: one T4 trains at 193 pairs/s (batch 64, max_length 128, fp16 autocast, peak VRAM 7.5 GB) and scores
  at 1,623 pairs/s (1,190–1,390 including tokenization). Projected: 1 epoch 1.33 h, MATCH sets 0.09 h,
  TEST 0.91 h, DEV_REST_REST 0.59 h. Untrained bge-reranker-base ROC-AUC on 20k MATCH_VALID pairs 0.8365.
  Toy run (2,873 pairs, 30 steps): held-out AUC 0.817 → 0.924. All code paths (train save, 2-GPU shard
  scoring, merge) ran on Kaggle.
- Learned: the cost concern was unfounded at 278M: ~2.5 GPU-h total. Kaggle image: torch 2.10.0+cu128,
  transformers 5.0.0.
- Next: k2 (1 epoch + scoring) running; then `ce/evaluate.py` and an M run with group `ce:ce_rec_base`.

### 2026-09-26 · M009 · Copy-cluster support + S2/S3 triangulation, text only (items 5, 6) · DROPPED
- Parent: x1_strsim · Metrics: `exp_M009.json` · Wall: 7 min (group build 13 s)
- Hypothesis: true matches of one S1 are near-duplicates of each other (sb_n_dup 2.28 for y=1 vs 1.14 for y=0 on
  3,000 S1); candidate-to-candidate name/address token-set (>=90 / >=80) without X3's stacked pass-1 p.
- Result: OOF 0.97825, Δ +0.00018 [0.00001, 0.00036]; band +0.00071 [−0.00001, 0.00147]; singleton −0.00255
  [−0.00427, −0.00078] (guard breach). Best column sb_best_addr_sim gain rank 9. Folds 1650-2134/3000.
- Learned: X3's singleton failure is not only from stacking; any "my candidates agree with each other" evidence
  raises p for chain duplicates in singleton lists too. Small gain, same failure mode.
- Next: none unless combined with a candidate-to-S1 gate.

### 2026-09-26 · M008 · Upweight feature-ambiguous pairs x2 (hard-negative curriculum, item 3) · DROPPED
- Parent: x1_strsim · Metrics: `exp_M008.json` · Wall: 7 min
- Hypothesis: 56% of loss sits in band S1; doubling the weight of pairs that look ambiguous from features alone
  (n_tset>=0.9 with a_tset<0.6 or empty, or |gap to next p_rerank|<0.1 with p_rerank in [0.05,0.95]; 18.5% of rows,
  58% of band rows) should spend more splits there. (First spec used 0-100 thresholds on 0-1 columns and was
  fixed before the run.)
- Result: OOF 0.97792, Δ −0.00015 [−0.00029, −0.000004]; band S1 −0.00064 [−0.00120, −0.00003]. Folds 2168-2411/3000.
- Learned: reweighting by a feature rule moves splits away from rows the model already separates without helping
  the ambiguous ones; the band loss is information-limited, not attention-limited.
- Next: none.

### 2026-09-26 · M007 · Macro-aware S1 weighting 1/n_candidates (item 1) · DROPPED
- Parent: x1_strsim · Metrics: `exp_M007.json` · Wall: 6.5 min
- Hypothesis: logloss averages over pairs, the metric over S1; weight 1/n makes each S1 count once.
- Result: OOF 0.97817, Δ +0.00010 [−0.00011, 0.00031]; band +0.00038 [−0.00049, 0.00126]; singleton segment
  −0.0149 [−0.0175, −0.0126] (guard breach). Decision flipped to expected-F.
- Learned: singleton lists are short, so 1/n upweights them as rows but the decision then loses singletons; the
  overall effect is noise. Guard failure alone rejects it.
- Next: none.

### 2026-09-26 · M006 · Density-invariant list features (item 2, RCA M3) · PARKED
- Parent: x1_strsim · Metrics: `exp_M006.json` · Wall: 7 min
- Hypothesis: l_n / count list features scale with pool density (test denser); z-scores and shares within the list
  (d_prr_z, d_rank_frac, d_share_prr05, d_prr_minus_med, d_share_prr_close, d_ntset_z, d_atset_z, d_nidf_z) carry the
  same ordering in a density-free form.
- Result: OOF 0.97796, Δ −0.00011 [−0.00028, 0.00005]; band −0.00044 [−0.00103, 0.00018]. Folds 1528-1990/3000.
- Learned: no in-distribution gain; OOF cannot show the density benefit it targets. Checked for transfer in
  `loco_groups.json` (+dinv) instead.
- Next: LOCO diagnostic.

### 2026-09-26 · N06 · Decision variants: threshold surface (item 16), calibration variants (item 19) · INFO
- Metrics: `decision_variants_x1_strsim.json` · Script: `plan/checks/decision_variants.py` · Wall: ~6 min, no training
- Hypothesis: one global isotonic map and one threshold may be wrong for long or tied lists (N04 hinted +0.00019).
- Change: on x1_strsim OOF, everything cross-fitted over the 5 folds (fit on 4, apply to 1), one-owner in all.
  A global isotonic + threshold; B isotonic per list-size bucket; C Platt; D threshold per list-size bucket;
  E threshold per (list-size bucket x top-2 p_rerank gap < 0.1).
- Result: A 0.97788. B +0.00003 [−0.00002, 0.00008]; C +0.00002 [−0.00003, 0.00007]; D +0.00007 [0.00003, 0.00011];
  E +0.00016 [0.00004, 0.00029], band S1 +0.00064.
- Learned: calibration variants do nothing. A 2-D threshold surface is a small real gain concentrated in band S1.
  Test has more long/tied lists, so the surface would fire more often there (c).
- Next: re-run on the final parent; if E still holds, add a per-cell threshold to the decision rule.

### 2026-09-26 · R0 · Experiment runner reproduces x1_strsim · INFO
- Metrics: `exp_R0.json` · Code: `src/ber/exp.py` (`exp-train`, `exp-predict`) · Wall: 6.5 min
- Change: new runner that writes only to `work/match/runs/<id>/` (match-train --run also overwrites the live model),
  with feature groups, label-free sample weights, seeds, band-stratum paired delta.
- Result: OOF 0.97807, per-entity max |diff| vs x1_strsim 0.0; best_iteration 1651/1819/2011/1908/1574 = x1_strsim.
- Next: M005-M009.

### 2026-09-26 · N05b · Pool-competition rule on top of universe one-owner · DROPPED
- Metrics: `pool_competition_increment.json` · Script: `plan/checks/pool_competition_increment.py` · Wall: 55 min
- Hypothesis: N05's +0.00104 compared against one-owner that only saw the 165,545-S1 slice. Test-time match-predict
  runs one-owner over all 1.73M test S1, so the rule may already be in the submission.
- Change: competitors' p_cal = full_single on every train feature row (in-sample outside the slice, so one-owner here
  is optimistic); rule applied only on rows whose pool record is in no HOLDOUT_TEST/DEV_B list (84.1% of rows).
- Result: universe one-owner vs none +0.00094 [0.00085, 0.00103]. p_rerank rule on top: m=0 −0.00005
  [−0.00009, −0.00002] (316 rows dropped, 247 of them true); m=0.2 −0.00001 [−0.00003, 0.00001]. Test replay of the
  live decision keeps 5,789,591 pairs = submission 2 exactly. On test the rule would drop 8,995 rows (0.50% of S1;
  France 4,633).
- Learned: the across-list signal as a hard rule is already spent by one-owner. As model features (M005) it can
  still help, but MATCH_TRAIN OOF one-owner sees only 180K S1, so M005 must also be judged with universe one-owner.
- Next: M005 with a universe-one-owner evaluation.

### 2026-09-26 · N05 · Pool-side competition (reverse retrieval) signal check · INFO
- Metrics: `pool_competition_check.json` · Script: `plan/checks/pool_competition_check.py` · Wall: ~6 min
- Hypothesis: FP rows are mostly records another S1 owns (N03: 63%); how a pool record is contested across all S1
  lists should separate them without labels.
- Result (165,545-S1 reference slice, full_single): 33.6% of rows contested (test 32.1%). Of TPs, 0.08% are rows
  where q is not the p_rerank-best S1 for the record. In the band, rows beaten by >=0.2 match 11.1% vs mean p_cal
  0.28; rows q wins by >=0.2 match 67.6% vs 0.61. Rule "drop kept row if another S1 has higher p_rerank" +0.00104
  [0.00094, 0.00115] vs live without universe one-owner (superseded by N05b).
- Learned: a real residual signal inside the band; the rule's gain was mostly the one-owner coverage gap.
- Next: N05b.

### 2026-09-26 · N04 · Decision rule on the test-like stratum (train labels) · INFO
- Metrics: `dense_stratum_rules.json` · Script: `plan/checks/dense_stratum_rules.py` · Wall: ~5 min
- Hypothesis: test lists carry more uncertain candidates, so a different threshold or expected-F might suit that regime.
- Result (full_single calibration slice, 165,545 train S1): S1 with >=1 row p_cal in [0.1,0.9) are 20.8% of S1,
  F0.5 0.95018 vs 0.98974 for the rest, and carry 56% of the loss. Best threshold on that stratum = live 0.69;
  expected-F vs threshold +0.00024 [-0.00065, +0.00107]. Lists >5: best 0.79, +0.00019.
- Learned: no decision-rule quick win. The loss is in ambiguous lists, and test has 1.25-2.7x more of them.
- Next: `plan/RCA_RESEARCH_PLAN.md` tier 1.

### 2026-09-26 · N03 · Train-vs-test shift diagnostic (test inputs only) · INFO
- Metrics: `test_shift_diag.json` · Script: `plan/checks/test_shift_diag.py` · Wall: 8 min
- Hypothesis: the leaderboard gap (internal 0.98161 vs public 0.966797) comes from France, from the denser test
  pool (5.75 vs 4.68 records per S1, all countries), or from stage-2 recall.
- Result: uncertain-band rows per S1 0.254 (train ref) vs test US 0.473, India 0.337, France 0.912. Stage-2 list size
  5.42 vs 5.60/5.63/8.99. Self-estimated F0.5 (Bernoulli(p_cal), candidates only) 0.9883 ref (actual 0.9887) vs
  test 0.9812/0.9851/0.9637. A list-statistics performance estimator gives test 0.9749/0.9775/0.9552, weighted
  0.9732. Adversarial AUC 0.986 is an IDF-scale artifact (max idf 16.150 vs 16.115); 0 live-model splits fall in
  the gap. Stage-2 tail truncation: not seen (last-slot-positive 8.9% test vs 11.4% ref; it does not predict
  misses on train). FP rows on ref: 37% orphans, 63% records owned by another S1.
- Learned: test is harder mainly through more near-duplicate distractors per list and France. The re-ranker was
  also truncated at 400/400 rounds on 45k queries (`rerank_train.json`). ~0.006 of the gap is not visible without labels.
- Next: `plan/RCA_RESEARCH_PLAN.md` (D1-D4, M1 cross-encoder feature, M2 density augmentation, R1 re-ranker retrain).

### 2026-09-26 · S2 · Submission 2 leaderboard · INFO
- Commit: 215b197 · Metrics: `match_train_full_single.json`, `match_predict_TEST.json` · log: `reports/test_usage_log.md`
- Public LB 0.966797 (S1: 0.966109). Internal 0.98161 held-out. Internal +0.0071 over S1, LB +0.0007.
- Learned: ~10% of the internal gain transferred; see N03.
### 2026-09-26 · CE-SMOKE1 · cross-encoder smoke (Friend B track, details in work/ce/) · INFO
- Metrics: work/ce/ce_rec_smoke_bf16/run.json (not artifacts/metrics/ — ce track lives in work/ce/).
- Hypothesis: LoRA r=16 on bge-reranker-v2-m3 attention + full head, bf16, batch 16×accum 4,
  trained on DEV_REST_180K pairs, learns pair match (eval AUC on held-out 1k S1 groups).
- Change vs first attempt: fp16 → bf16 (fp16 run went NaN by step 50, eval AUC 0.5215 — FAILED,
  log work/ce/ce_rec_smoke_train.log) + grad clip 1.0 + fail-fast NaN abort.
- Result: 1536 steps / 392 s; train loss 0.40 → 0.12; EVAL loss 0.1192, AUC 0.9906.
  Thermals (threads=4, workers=0): CPU 52–65°C, GPU 60–82°C steady, no guard pause.
- Learned: bf16 is required for this head on Ada; thermal headroom is ample for the full epoch.
- Next: full 1-epoch run on 976,897 DEV_REST_180K pairs (~4 h projected).

### 2026-09-26 · N02 · Experiment routine and this log · INFO
- Metrics: none (planning).
- `plan/EXPERIMENT_ROUTINE.md` written: stage-3 experiments on frozen MATCH_TRAIN candidates, ~40 min
  per run, paired Δ vs parent, error files per run. Loss analysis from `match_train.json`: FN inside
  candidates 47% of OOF loss (0.01219 of 0.02588), FP-involving 29%, blocking misses 24%.
- Learned: the precision/recall trade-off is already set by the decision threshold. Most of the loss the
  matcher can still fix is missed true pairs with p_matcher median 0.33, so features matter more than
  weighting the loss.
- Next: build `ber/exp.py`, reproduce M000 through it (must equal 0.97412).

### 2026-09-26 · N01 · Noise floor on MATCH_VALID · INFO
- Commit: fd77ee7 · Metrics: `validation_noise.json`
- Per-entity F0.5 std 0.1006; 82.4% of entities score exactly 1.0. Unpaired SE of the mean: 0.00071 at
  20K, 0.00022 at 200K, 0.00015 at 441K, 0.000078 at 1.655M.
- One-owner on vs off, paired: Δ +0.000007 [−0.000009, +0.000032], 2 entities changed, diff std 0.0016.
- Learned: unpaired comparisons on 20K cannot resolve gains below ~0.002; paired comparisons on fixed
  folds are required. One-owner is neutral at 0.9% coverage; its effect on test is unmeasured.
- Next: paired bootstrap in every M run.

### 2026-09-26 · S1 · Submission 1 · INFO
- Commit: fd77ee7 · Metrics: `match_predict_TEST.json` · Wall: 30 min predict (+141 min test candidates, `candidates_TEST.json`)
- 1,732,544 S1 rows, 5,688,259 pairs, validator passed. Mean predicted list size: France 3.28,
  India 3.24, US 3.33; empty share 5.6% / 6.2% / 5.9%.
- Learned: France prediction statistics look like US/India's, but that shows only that the output looks
  normal, not that it is correct (no labels).
- Next: record the leaderboard score here.

### 2026-09-25 · M000 · Stage-3 matcher baseline (LightGBM, 50 features) · ADOPTED
- Commit: c050879 · Metrics: `match_train.json`, `match_eval_MATCH_VALID.json` · Wall: 16.5 min train
- Hypothesis: pair features (name/address token-set, IDF, numbers) plus list-context features over the
  stage-2 candidates beat thresholding p_rerank alone.
- Change: 5-fold GroupKFold LightGBM binary, lr 0.05, 63 leaves, early stop 100; isotonic calibration;
  decision rule chosen on OOF.
- Result: OOF F0.5 0.97412 vs p_rerank-threshold floor 0.94655 (+0.02757). Rules on OOF: threshold
  0.97388, expected-F 0.97407, threshold+one-owner 0.97412, expected-F+one-owner 0.97433. Chosen
  threshold 0.70 + one-owner because expected-F did not clear `expected_f_min_gain` 0.001. MATCH_VALID
  0.97449 (−0.00037 vs OOF, within noise). Pairwise precision 0.9916, recall in candidates 0.968.
- Learned: `p_rerank` dominates gain (4.6M), then list features `l_p_ratio_max` (0.79M), `l_p_gap_max`
  (0.39M): a pair's score relative to the rest of its list carries the most signal. Worst segments:
  match_1 0.913, chain ≥10 0.964, India 0.970.
- Next: feature backlog F1–F9 in `plan/EXPERIMENT_ROUTINE.md` §8.

### 2026-09-25 · R2 · Stage 1+2 on validation S1 (one-shot) · INFO
- Commit: 1a2fb63 · Metrics: `rerank_eval_HOLDOUT_TEST.json` · Wall: 30.5 min
- F0.5 ceiling 0.99276, pair recall 0.9755, entity recall 0.99835, mean candidate size 5.43 (max 41).
- Learned: matches DEV_B (ceiling 0.99334, size 5.43) within 0.0006; stage 1+2 did not overfit DEV_A/DEV_B.
  This is the only validation-S1 score so far; the matcher has not been scored there.

### 2026-09-25 · R1 · Stage-2 re-ranker and leave-one-country-out · ADOPTED
- Commit: 2150e55 · Metrics: `rerank_eval_DEV_B.json`, `rerank_loco.json`
- Result: DEV_B candidate size cut from ~265 (E6) to 5.43 at F0.5 ceiling 0.99334 (−0.0022 vs E6).
  LOCO: trained on US, scored on India 0.98492 vs all-country 0.98897 (−0.0041); trained on India,
  scored on US 0.99654 vs 0.99626 (+0.0003).
- Learned: the US-only model loses 0.004 on India, so a country missing from training costs something
  in stage 2. France on test is in that position.

### 2026-09-25 · E6 · Stage-1 confirmation on DEV_B · ADOPTED
- Metrics: `s1_E6.json` — DEV_B ceiling 0.99550 vs DEV_A 0.99483; gate (DEV_B ≥ DEV_A − 0.003) passed.

### 2026-09-25 · E5 · Chain-size quota multiplier · DROPPED
- Metrics: `s1_E5.json`
- Hypothesis: queries whose name belongs to a chain (≥10 pool records) need a larger quota.
- Result: no setting passed the gate. min_chain10 × 2: chain≥10 ceiling +0.0047 (gate 0.005) at +10%
  size; × 3: +0.0070 but +21% size (gate 15%).
- Learned: chains gain from more candidates, but the cost is paid by every query. Chain ≥10 is still a
  weak segment in stage 3 (0.964, M000). Revisit only if stage 3 cannot close it.

### 2026-09-25 · E4 · Per-channel quotas · ADOPTED
- Metrics: `s1_E4.json`
- Result: best ceiling 0.99662 at mean size 741; chosen LEARN150/NAME0/ADDR50/PAIR100/XK50/CAT20 at
  0.99571, mean size 291 (within 0.001 of best at 39% of its size).

### 2026-09-25 · E3b · Learned stage-1 channel (logistic) · ADOPTED
- Metrics: `s1_E3b.json` — ceiling at K=300: learned 0.99223 vs hand-weighted 0.97793 (+0.0143).

### 2026-09-25 · E3 · Per-family posting caps · ADOPTED (smaller caps)
- Metrics: `s1_E3.json`
- Result: larger caps gained ≤ 0.0004 ceiling at up to 2× latency (NT 0.005: 0.99621 at 11.4 ms vs
  0.001: 0.99588 at 5.5 ms). Kept the small caps.

### 2026-09-25 · E2 · Blocking-key families · ADOPTED (5 kept, 2 dropped)
- Metrics: `s1_E2.json`
- Kept NP, SP, NC, AB, XK (ceiling gains 0.0055, 0.0042, 0.0029, 0.0059, 0.0052). Dropped SK (2
  exclusive pairs, gain 0) and PH phonetic (2 exclusive pairs, gain 0.00008).
- Learned: phonetic keys add nothing the n-gram and token keys don't already find.

### 2026-09-25 · E1 · Hand-weighted channel baseline · INFO
- Metrics: `s1_E1.json` — top-50 ceiling 0.96972, pair recall 0.9193; pair recall@150 0.9374 passed the 0.91 gate.

### 2026-09-25 · E0 · Mined abbreviation dictionary · ADOPTED
- Metrics: `e0_abbrev.json`
- Result: true-pair address Jaccard 0.6989 → 0.7577 (+0.0588); random negatives 0.0219 → 0.0232
  (+0.0013). Gate: true rise ≥ 0.005 and larger than negative rise.
