# LightGBM round cap (`num_boost_round`): measured evidence, not a guess

Status: reference, 2026-09-26. Read this before setting `num_boost_round`/`ROUND_BUDGET` for any stage-3
matcher training run, at any data scale. Every number below cites its `artifacts/metrics/*.json` or a run
log — no number here is estimated.

## The mistake this file exists to prevent

Twice in one session, a fixed round cap was set from the *previous* scale's experience and turned out to
be too low at the next scale up:
- `dev_rest_180k.py` used the shipped default `num_boost_round=3000` (fine at 180,000 S1) on a 360,000-S1
  combined run — 2 of 5 folds hit the cap without converging (`best_iteration` 2997 and 3000, `work/logs/dev_rest_180k.log`).
- `dev_rest_full.py` then raised the cap to 4000, reasoning only "the 360k run needed more than 3000" — at
  1,655,458 S1, 3 of 3 folds completed so far hit the cap again (`best_iteration` 4000, 3998, 3988,
  `work/logs/dev_rest_full.log`). The fix from the last scale was still too small for this one.

The pattern: **a round cap sized for one data scale is not safe to assume at the next scale.** This file
is the fix — a measured table, and a mechanical check to run after every training, not another guess.

## Measured data: `best_iteration` per fold, all at `learning_rate=0.05` (unchanged across every run below)

| Run | Total rows | Fit rows/fold (~72% of total) | Cap | `best_iteration` per fold | Any fold hit the cap? |
|---|---|---|---|---|---|
| k5_base (180,000 S1) | 975,699 | ~702,500 | 3000 | 2085, 1997, 1936, 1849, 1891 | No |
| k5_feat (180,000 S1) | 975,699 | ~702,500 | 3000 | 1448, 1703, 1927, 1901, 1724 | No |
| x1_strsim (180,000 S1) | 975,699 | ~702,500 | 3000 | 1651, 1819, 2011, 1908, 1574 | No |
| dev_rest_180k (360,000 S1) | 1,952,601 | ~1,405,900 | 3000 | 2997, 2424, 2566, 2727, **3000** | Yes, 2 of 5 |
| dev_rest_full (1,655,458 S1) | 8,979,983 | ~6,465,600 | 4000 | **4000**, **3998**, **3988**, **3996**, **4000** | Yes, 5 of 5 (OOF F0.5 0.98221 reported with this caveat) |

| train_final_single (1,655,458 S1, one model) | 8,979,983 | 7,633,743 | 12000 | **8208** (converged) | No |
| train_final_comp (1,655,458 S1, one model, 75 features incl. comp) | 8,979,983 | 7,633,743 | 12000 | 6524 (converged) | No (`match_train_full_comp.json`) |
| train_final_comp full_comp_num (1,655,458 S1, one model, 83 features incl. comp + num) | 8,979,983 | 7,633,743 | 12000 | 5636 (converged) | No (`match_train_full_comp_num.json`) |
| train_final_comp full_ce (MATCH_ALL minus DEV_REST_180K = 1,475,458 S1, one model, 71 features incl. p_ce group) | 8,003,081 | 6,803,405 | 12000 | 2334 (converged) | No (`match_train_full_ce.json`) |
| train_xgb_blend full_ce_xgb (XGBoost hist, lossguide 63 leaves, eta 0.05; same rows as full_ce) | 8,003,081 | 6,803,405 | 12000 | 2788 (converged) | No (`match_train_full_ce_xgb.json`) |
| train_final_comp full_ce_nf (MATCH_ALL minus DEV_REST_180K = 1,475,458 S1, one model, 74 features: full_ce + nfreq) | 8,003,081 | 6,803,405 | 12000 | 2330 (converged) | No (`match_train_full_ce_nf.json`) |
| exp M018 main (180,000 S1, + p_ce) | 975,699 | ~702,500 | 3000 | 423, 589, 720, 477, 473 | No (`exp_M018.json`) |
| exp M019 main (180,000 S1, + p_ce + nfreq) | 975,699 | ~702,500 | 3000 | 644, 398, 555, 583, 357 | No (`exp_M019.json`) |

| exp R0 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 1651, 1819, 2011, 1908, 1574 | No (`exp_R0.json`) |
| exp M005 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 1758, 1672, 1538, 1465, 1200 | No (`exp_M005.json`) |
| exp M006 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 1990, 1936, 1814, 1528, 1733 | No (`exp_M006.json`) |
| exp M007 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 1828, 2187, 2037, 1782, 1787 | No (`exp_M007.json`) |
| exp M008 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 2168, 2112, 2270, 2411, 1893 | No (`exp_M008.json`) |
| exp M009 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 1941, 1650, 1830, 2134, 2060 | No (`exp_M009.json`) |
| exp M010 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 1978, 1892, 1974, 1535, 1719 | No (`exp_M010.json`) |
| exp M011 spec_name (180,000 S1) | 975,699 | ~702,500 | 3000 | **2999**, **3000**, **2989**, **2994**, **2991** | Yes, 5 of 5 (`exp_M011.json`) |
| exp M011 spec_addr (180,000 S1) | 975,699 | ~702,500 | 3000 | 1933, 1947, 2197, 2057, 2216 | No (`exp_M011.json`) |
| exp M011 spec_ctx (180,000 S1) | 975,699 | ~702,500 | 3000 | 935, 1252, 1108, 1075, 1023 | No (`exp_M011.json`) |
| exp M011 main (180,000 S1) | 975,699 | ~702,500 | 3000 | 720, 625, 793, 745, 695 | No (`exp_M011.json`) |
| exp M012 main_part2 (180,000 S1) | 975,699 | ~702,500 | 3000 | 935, 1182, 1185, 1248, 1114 | No (`exp_M012.json`) |
| exp M012 main_part3 (180,000 S1) | 975,699 | ~702,500 | 3000 | 1318, 1256, 1415, 1121, 1248 | No (`exp_M012.json`) |
| exp M013 main (180,000 S1, band rows only) | 975,699 | ~702,500 | 3000 | 119, 116, 122, 135, 152 | No (`exp_M013.json`) |
| exp M020 main (180,000 S1, CE-derived family) | 975,699 | ~702,500 | 3000 | 578, 573, 761, 566, 424 | No (`exp_M020.json`) |
| exp M021 main (180,000 S1, CE-derived family) | 975,699 | ~702,500 | 3000 | 468, 613, 614, 566, 397 | No (`exp_M021.json`) |
| exp M022 main (180,000 S1, CE-derived family) | 975,699 | ~702,500 | 3000 | 657, 518, 633, 717, 377 | No (`exp_M022.json`) |
| exp M023 main (180,000 S1, CE-derived family) | 975,699 | ~702,500 | 3000 | 742, 576, 791, 438, 566 | No (`exp_M023.json`) |
| exp M024 main (180,000 S1, CE-derived family) | 975,699 | ~702,500 | 3000 | 587, 511, 554, 477, 447 | No (`exp_M024.json`) |
| exp M025 main (180,000 S1, CE-derived family) | 975,699 | ~702,500 | 3000 | 464, 527, 566, 420, 387 | No (`exp_M025.json`) |
| exp M026 main (180,000 S1, CE-derived family) | 975,699 | ~702,500 | 3000 | 545, 716, 561, 490, 427 | No (`exp_M026.json`) |
| exp M027 main (180,000 S1, cex_p ce_rec_c0) | 975,699 | ~702,500 | 3000 | 422, 383, 454, 394, 370 | No (`exp_M027.json`) |
| exp M028 main (180,000 S1, ce_rec + ce_rec_c0) | 975,699 | ~702,500 | 3000 | 402, 400, 462, 309, 319 | No (`exp_M028.json`) |

M011 note: the name-only specialist (n_* columns only) hit the 3000 cap in all 5 folds at 180,000 S1, while the
full-feature model converges at ~1500-2100: a weaker feature set needs MORE rounds at lr 0.05, so a specialist cap
must not be copied from the main model's. Re-run specialists with a cap >= 8000 and check again.

First converged measurement at MATCH_ALL scale: one model on ~7.6M fit rows at lr 0.05 needed 8208 rounds
(`artifacts/metrics/match_train_full_single.json`, `work/logs/submission2.log`) — about 4x the 180k-S1
runs (~2000) and 2x the 4000 cap the 5-fold MATCH_ALL run was cut at. For MATCH_ALL-scale training at
lr 0.05, start the cap at >= 12000.

Source files: `artifacts/metrics/match_train_{k5_base,k5_feat,x1_strsim}.json` (`folds[].best_iteration`),
`work/logs/dev_rest_180k.log`, `work/logs/dev_rest_full.log`.

## What this evidence actually supports, and what it doesn't

**Supported (a):** at a fixed learning rate, the number of rounds needed to converge rises as training
data grows, and the safe cap at one scale is not automatically safe at the next. The 180,000-S1 runs never
came close to their 3000 cap (max 2085); by 1,655,458 S1 every fold measured so far needed *more* than
4000.

**Not supported — do not over-fit a formula to two noisy points:** the 360,000-S1 run shows real
per-fold variance (2424 to 3000+ within the same run, same data scale) — which S1 happen to land in a
fold visibly affects how many rounds it needs, not just the row count. Do not compute a clean
multiplier (e.g. "rounds scale linearly with rows, so multiply by 9.2x") and trust it blindly. The
right level of confidence here is "rounds-needed grows with data size, roughly, with real fold-to-fold
noise" (c), not a precise predictive formula (that would be (d), a guess dressed as a measurement).

## The actual rule to follow (mechanical, not a guess)

1. **Never trust a training run without checking `best_iteration` against the cap for every fold.** If
   any fold's `best_iteration >= num_boost_round - early_stopping_rounds`, that fold did not converge —
   it was truncated, not early-stopped. Its OOF prediction is optimistic-in-the-wrong-direction
   (undertrained, likely worse than the model's true capability at that data size), not simply "a
   slightly worse but valid measurement."
2. **When starting a run at a new data scale, do not assume the previous scale's cap.** Start from this
   table's largest comparable point and go meaningfully higher, or use `hparam_search.py`'s
   `round_cap()`/retry-at-2x-cap logic (`plan/checks/hparam_search.py`, reviewed 2026-09-26) rather than a
   fixed number: it scales the cap by `learning_rate` and automatically retries a truncated fold at 2x the
   cap once before flagging it as still-truncated. That logic accounts for the learning-rate axis; combine
   it with a data-size-aware starting point from this table for the row-count axis, since the two effects
   are independent and this table only has evidence for the row-count one.
3. **After any run, append its `best_iteration` values to the table above**, with its own metrics file
   cited, before drawing a conclusion from its F0.5. If a fold hit the cap, say so explicitly next to the
   headline number, the way this file's own MATCH_ALL entry does — do not report a capped run's F0.5 as if
   it were the model's converged best.
4. **If a whole training set (`MATCH_ALL`, ~8.98M rows) will be trained again, allocate a much higher cap
   from the start** — the current dev_rest_full.py run at cap 4000 hit it on every fold measured so far,
   so a rerun should start at least 2-3x that (8000-12000) and still verify against rule 1, not assume
   that number is enough either.

## Open, unmeasured

- No fold at any scale in this table has been allowed to run to full, uncapped convergence — every "hit
  cap" fold's true convergence point is unknown, only bounded below by the cap it hit. This means the true
  shape of the rows-vs-rounds relationship is not yet known past the point where the cap starts binding.
- Whether the relationship is closer to linear, sub-linear (e.g. sqrt), or something else in rows has not
  been measured — the two available scale-up points are too few and too noisy (see above) to fit a curve.
