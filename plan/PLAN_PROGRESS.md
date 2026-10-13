# Planning progress (fresh-rebuild plan for candidate generation)

Purpose: context anchor for the planning session. The plan itself is `plan/BUCKETING_PLAN.md`.
Old `src/` = EDA-only code (reference, not reused as-is). New code lives in
`code/business_entity_resolution/` (matches the required submission layout).

## Status
- [x] 1. Verify dataset facts the plan depends on (`plan/checks/data_facts.json`, `romanize_quality.json`)
- [x] 2. Draft architecture + module specs (plan v1 draft)
- [x] 3. Draft evaluation protocol, experiments, acceptance gates
- [x] 4. Draft compute/memory budget for the friend's laptop
- [x] 4b. Reference romanizer built + validated (`plan/reference/translit_ref.py`, `ref_quality.json`, `goldens.json`):
      skeleton match 98.24% of Indic true pairs, token-equal 79.7%; fixed chillu + RRA cluster rules.
- [x] 5. Review pass 1 findings (being applied in v1.1):
  1. §5.2.1 prose had a leftover "hmm-check" example → make the reference implementation normative + goldens file.
  2. Chain rule used d_EN "0 if over cap" → disables the rule for the biggest chains. Use raw df.
  3. E2 keep-rule measured reach twice → also count ranking value (recall at default quotas).
  4. No full `blocking.json` → executor would invent keys. Add the full default JSON.
  5. CLI commands had no argument spec / set→split mapping. Add table.
  6. Official validator with the candidate file holds ~140M ID strings in Python sets (≈10–14 GB) → OOM risk. Add a streaming validator implementing the same rules; official run is best-effort.
  7. Stage-2 sampling said "top-100 by C_BOTH" without saying "among stage-1 selected negatives". Clarify; compute features only after sampling.
  8. Quota grid (486 configs) needs a vectorized method, otherwise minutes per query set. Specify rank-based method.
  9. "Next larger quota config" undefined → next point on the E4 Pareto list.
  10. Stage-2 must be retrained if stage-1 config changes → config hash check.
  11. Code transfer to the laptop not specified → git bundle/zip by a human.
  12. Postings hard limit rule was only in the risk table → move into M6 spec.
  13. Composite formulas when a family is dropped → skip the term; channel with no families → quota 0.
  14. Skeleton examples in §5.5.1 must match the reference (kansaltin now 'kansalting' → KNSLTNK).
  15. Malayalam/Tamil RRA clusters and chillu letters added to the romanizer.
- [x] 5b. All 15 findings applied → plan v1.1.
- [x] 6. Review pass 2 findings, applied:
  16. NC greedy edge-stripping was asymmetric ("happy nails llc" vs "happynails com") → fixed-variant NC keys (keys_ref.concat_variants), verified.
  17. Skeleton goldens in prose were wrong (v/f/w map to P: developers → TPLPRS, silver → SLPR) → corrected from the reference.
  18. Golden count in prose (57) wrong → 74 cases (68 names, 6 addresses).
  19. DEV_B usage order ambiguous (stage-1 vs stage-2) → two separate one-time checks; stage-1 change after P5 → rerun P5.
  20. `rerank-eval` semantics (needs candidates?) → runs stage-1+2 if missing.
  21. Cut rule needs RR_CAL stage-1 recall → `rr_cal_stage1.json` written by rerank-build.
- [x] 7. Final summary to user

## Verified facts (source in brackets)
- Every row of all 6 source files has exactly 4 tab-separated fields; no CR, no BOM; header
  `entity_id business_name business_address country`. Plain `line.rstrip("\n").split("\t")` is exact. [data_facts.json]
- Double-quote chars exist (up to 1,448 per file). pandas default quoting gives the same row
  count, but it can still alter field text, so the plan uses QUOTE_NONE / manual split. [data_facts.json]
- Row counts: train S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316.
- IDs match `^S[123]-\d+$`; no leading zeros; max number 999,999,995 (< 2^32); IDs unique per file;
  S2 and S3 numbers overlap (26,801 train, 25,008 test), so a record key = (source, number); no train/test ID overlap.
- No empty names or countries; empty addresses: train S2 168,967 / S3 175,916; test S2 129,408 / S3 136,098.
- Literal `<NULL>` inside addresses (a missing component): train S2 43,799, S3 43,572; test S2 35,024, S3 35,153. Never in names.
- `<CITY_NAME>` placeholder in names: train S1 116, S2 220 (+1 `<CIT_NAME>`); test S1 108.
- Countries: train {US, India}; test {US, India, France}; identical spelling in all sources.
- Non-ASCII letters in the pool: 9 Indic scripts (Devanagari, Telugu, Kannada, Bengali, Tamil, Gujarati,
  Malayalam, Oriya, Gurmukhi) + accented Latin + `º` (France). No Arabic/CJK/Cyrillic. S1 = Latin only.
- 752,869 train-pool names contain Indic letters. On 30K fit true pairs: romanized vs S1 name have the same
  token count 92.0%; share ≥1 identical token 77.3%; trigram Jaccard median 0.273. Tokens are English words
  written phonetically ('praivet'→private, 'kansaltin'→consulting), so a train-learned token dictionary works. [romanize_quality.json]
- ZWNJ (U+200C) inside Kannada/Malayalam words; it must be deleted, not turned into a space (it split 'san rais').
- Ground truth: 2,206,821 rows = every train S1; 123,247 singletons; 7,638,365 links; max 11; no record has 2+ owners; all linked IDs exist.
- Validator: header compared lower-cased; row ID list split on ',' (no spaces allowed); only `\n` is stripped,
  so Windows `\r\n` output would corrupt the last ID of every row. Candidate file is optional for the
  validator but required in the zip; matches ⊄ candidates is only a warning.
- blake2b 8-byte hash: 0.35 s per 1M keys; searchsorted of 200 keys in 50M sorted uint64: 0.004 ms.
- Measured on this Mac (M3): old stage-1 union 2.3 to 3.8 ms/query (numpy); old per-pair Python featurizer ~8.3K pairs/s with 6 workers.

## Decisions log (decision, why, alternatives rejected)
- D1 Target platform: friend's laptop under WSL2 Ubuntu 24.04, Python 3.12; code must ALSO run on native Windows/macOS (spawn-safe, mmap indexes, explicit utf-8, `\n` newlines). Rejected: fork-dependent shared memory (breaks on Windows spawn).
- D2 Indexes = per-family sorted uint64 blake2b key hashes + CSR postings, saved as .npy and memory-mapped. Rejected: Python dicts (non-shareable, 2-4 GB per worker), string arrays (memory).
- D3 Romanize Indic text inside normalization (stdlib unicodedata table, explicit consonant/vowel map), plus a train-derived token dictionary. Rejected: external transliteration libs/models (compliance), separate ROM channel (romanized tokens now live in the normal token space).
- D4 Stage-1 = per-channel quota union. Rejected: single additive score (measured worse: diag_blocking2/diag_quota).
- D5 Stage-2 = LightGBM candidate re-ranker with adaptive cut; its output = candidate_pairs.tsv. Rejected: hand formulas (diag_rerank: +0.001 @150).
- D6 Country is a re-ranker feature only (3-state), never a filter.
- D7 New split + disjoint query sets (DEV_A tune, DEV_B confirm, RR_TRAIN, RR_CAL, MATCH_POOL, VAL).
- D8 df caps are fractions of the indexed pool size (train 10.32M vs test 9.97M).

## Round 3 (user: split labelled train into train/test; hostile critique; upgrade) — IN PROGRESS
- [x] Critique subagent launched (writes plan/critique/CRITIQUE.md)
- [x] Split evidence (`plan/checks/split_design.json`):
  - test S1 names found among train S1 names: all 0.3526 (US 0.3591, India 0.4602, France 0.0001)
  - random 20% train holdout names found in the other 80%: all 0.378 (US 0.3348, India 0.4428)
    → random stratified entity split mimics official test for US/India; name-grouped split would not. France only via LOCO.
  - true links: 7,638,365, 100% same country; 0 with empty pool name; 337,018 with empty/<NULL>-only pool address
    (of 344,883 such pool records → empty-address records are almost always owned)
  - singleton rate US 0.0558, India 0.0559; match counts 1..6+ very similar per country
  - pool per S1: train 4.677, test 5.754; train orphans 2,681,854 (26.0%). If test had train's 3.461 links/S1,
    test orphans would be ≈ 3.97M (≈40%) — ESTIMATE, not measured.
- [x] Critique read; key numbers re-verified independently (`plan/checks/verify_critique.json`): France token shares,
      exact-name 0.2855 / chain≥10 0.2468, Indic usable token 0.2615 / usable skeleton 0.0554, d_EN median 31 — all reproduced.
- [x] Prototypes run (evidence for v2):
  - `norm_v21_check.json`: v2.1 (Gurmukhi fixes, letter collapse, st/ste classes) improves exact-name 0.2859→0.2960, no script regresses.
  - `abbrev_miner.json`: unsupervised per-country miner, cos≥0.95: France r→rue, av, all, imp, rte, ch, pl, q, bd, n→no, b→bis, t→ter (+1 error l→labbe, df 0.0006 → removed by target floor 0.001); train: state codes, dr, ave, apt, hwy, blvd.
  - `learned_stage1.json`: logistic stage-1 score top-150 pair recall 0.9529 vs hand 0.9308; quota union 0.9682 vs 0.9623 at equal size; F0.5 ceiling 0.9882 vs 0.9854.
- [x] Plan v2 written (`plan/BUCKETING_PLAN.md`; v1.1 kept as `BUCKETING_PLAN_v1.1.md`); review log §11.
- [x] Self-review of v2: fixed E0 needing an index before P3 (now index-free negatives), mine-abbrev independence from normalize,
      holdout size exact (441,365), name-unseen definition, critique-only numbers labelled "NOT re-run", Q1 budget contradiction.

## Reference artifacts (normative for the executor)
- plan/reference/translit_ref.py, keys_ref.py, goldens.json, keys_check.json (G on train pool: 268 tokens), ref_quality.json

## Open questions for the human (only things that cannot be decided from data)
- Q1 Submission zip size limit (candidate_pairs.tsv ≈ 13 bytes × mean × 1.73M; mean 100 ≈ 2.3 GB raw). Plan budget: mean ≤ 100.
- Q2 Exact laptop CPU model / OS (plan auto-sizes workers; WSL2 assumed).
