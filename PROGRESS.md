START: Fri Sep 25 01:43:41 IST 2026
LAST_UPDATE: Fri Sep 25 14:30 IST 2026
REMOTE_NOTE_2026-09-25: linux setup: dataset symlinked via ../data, .gitignore fixed, .venv created, pip install running, missing bundle reported; local commit only, push left for user per M2
PULL_MERGE_2026-09-25: fetch+pull done, conflicts resolved, merge d55ad2a, now ahead 6. Push NOT done (policy-blocked + heavy files flagged, see assumptions). Manual slim+push commands left for user.
HOUSEKEEPING: COMMIT DONE 3a177a5 main clean; rename plus push left for user.
TEAMMATE BUNDLE: zip running in background (pid 96465, log logs/bundle_zip.log) -> teammate_share/teammate_bundle.zip split 1900m (M7 single-file cap).
TEAMMATE BUNDLE: DONE teammate_share/teammate_bundle.z01 (1.9G) + teammate_bundle.zip (1.1G); 51 files, 6.7G uncompressed; listing verified; bundle git-ignored, handoff only.
TEAMMATE BUNDLE V2 (no dataset): old parts deleted; zipping candidates + index + 4 parquets + files -> teammate_share/teammate_artifacts.zip (split 1900m) in background (pid 5401, log logs/bundle_zip2.log).
TEAMMATE BUNDLE V2: DONE teammate_artifacts.z01 (1.9G) + teammate_artifacts.zip (39M); 34 files, 4.2G uncompressed, no student_resource; listing verified; pid 5401 reaped.
PHASE 0: DONE
PHASE 1: DONE
PHASE 2: DONE
PHASE 3: IN-PROGRESS (candidates ready; TF-IDF fit next, then featurize/train/val)
PHASE 4: TODO
PHASE 5: SKIPPED(time: clock anomaly forced compressed path; see assumptions 09:30)
PHASE 6: TODO
NEXT ACTION: Phase 3 featurize + LightGBM OOF + val F0.5
PHASE 2: TODO
PHASE 3: TODO
PHASE 4: TODO
PHASE 5: TODO
PHASE 6: TODO
NEXT ACTION: Phase 1 EDA (structural 1a on fit part, then signals 1b, structure 1c)
REPOSITORY HOUSEKEEPING: Local uncommitted work is being committed; folder rename and remote push are deferred because project rules prohibit moving the project root or running git push.

BUCKETING DIAGNOSTICS (Fri Sep 25 14:01:43 IST 2026): DONE, no training. reports/bucketing_diagnostics.md, metrics diag_{scripts,blocking,breakdown,rerank,blocking2,quota}.json, aux index artifacts/index/aux.npz. Found normalize v1 erases Indic names (699K train pool). Next: bucket_prompt_v2.md Phase 1 (normalize v2).
LFS_MIGRATE_2026-09-25: `git lfs track "*.npy" "*.bin" "*.pkl" "*.parquet"` + commit 81b063e, then `git lfs migrate import --skip-fetch` rewrote 9 unpushed commits (npy 3.1G + bin 744M -> 3.8G LFS objects). INFO after: git push payload still ~123M JSON; LFS quota (1G free) would choke, so squashed instead.
SQUASH_2026-09-25: backup branch + `git reset --soft origin/main` (only --hard is denied) + single commit 69e9106. Tree verified identical (9ff68cb7). `migrate info` now: 1 commit, ~2.3M total, 0 LFS objects. Branch: ahead 1. Push left for human (git push* denied in-session).
NEXT ACTION: human runs `git push` (small, ~2.3M); optional later `git lfs prune && git gc` to reclaim .git space.
VALIDATION PLAN (2026-09-26): plan/VALIDATION_PLAN.md written (MATCH_ALL 1,655,458 5-fold OOF, paired bootstrap, leakage register L1-L10, val-S1 ledger). Noise floor measured: artifacts/metrics/validation_noise.json. Nothing implemented yet; next = plan §10 steps 1-4.
X4_ENTITY_GATE (2026-09-26): entity-level LightGBM gate stacked on frozen x1_strsim pass-1 p (match-gate-train command, plan/FEATURE_EXPERIMENTS.md X4). OOF F0.5 0.97823 vs x1_strsim 0.97807 (delta +0.00016, below +0.0005 min gain); MATCH_VALID delta +0.00033 CI lower bound -0.00001 (fails X4's extra condition); singleton guard OOF -0.00206 (breach). REJECTED, code/config restored, live matcher still x1_strsim (match-eval MATCH_VALID confirmed 0.97848). Details: plan/FEATURE_EXPERIMENTS.md §4 X4 note.
REPO_REORG_SUBMISSION_V2 (2026-09-26): branch submission-v2. .gitignore rewritten (*.zip/*.z01, .claude/, work/* except trained models, data/* except split script/write-up/manifest); legacy src/ -> legacy/src/, overnight-agent files -> archive/agent_runs/, split.md -> docs/, root README.md added. Git LFS: work/match/models (x1_strsim), work/match_k5_submission1/models (submission 1), work/rerank/model_v1.txt, submission/output/*.tsv (~400 MB). Tests: 168 passed (pytest 9.1.1). Not done: no history rewrite (old tok_idf.json 48 MB / split.json 35 MB blobs remain in main history); local .git garbage (~3 GB tmp_pack) and orphan .git/lfs cache (~3.6 GB) not cleaned.
FINAL_SUBMISSION (2026-09-27 22:40): S16 = full_c0_rgrp at LightGBM threshold 0.73 + DRACE (untrained e5 S1@5/seed@5, distilled e5 S1@5, c0 >= 0.995, one-owner). Public LB 0.983958 (S14 0.983741, S13 0.983604). Live model copied to work/match/models (matcher.json threshold 0.73). Rebuild: plan/checks/run_s16.sh. Retrieval research sweep N11-N13, AD1, D10, SD5, TH1-TH2 in RESEARCH_LOG.md.
