# OFFICIAL SPEC (condensed from problem_statement.pdf). If any other file disagrees with this one, THIS file wins.

## Task
Given business records from 3 sources with noisy fields and no shared IDs, find which S2/S3 records refer to the same real-world business as each Source 1 (S1) entity. S1 is the deduplicated reference. An S1 entity may match zero, one, or many records from S2 and S3.

## Files (all tab-separated; read with sep="\t")
- dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
- dataset/test/test_source1.tsv, test_source2.tsv, test_source3.tsv (no labels)
- Source columns: entity_id (prefix S1-/S2-/S3- gives the source), business_name, business_address, country
- Ground truth columns: source1_entity_id, matched_entity_ids (comma-separated S2-/S3- IDs; empty = singleton)
- country: train has US and India. TEST ALSO HAS FRANCE (unseen). Treat country as an open set of strings. Do not hard-code, filter, or one-hot on {US, India}. Every test entity, France included, must appear in the submission.

## Noise to expect
- Names: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix inconsistencies, DBA/trade names, punctuation (& vs and), word-order transposition, typos, transliterations.
- Addresses: abbreviations (Rd/Road, St/Street), transliteration variants, missing components (no PIN, no state), landmark references (Near SBI ATM), municipal numbering formats, component reordering.

## Outputs (both tab-separated, in output/)
1. matching_results.tsv: columns source1_entity_id, matched_entity_ids. Only this file is scored.
2. candidate_pairs.tsv: columns source1_entity_id, candidate_entity_ids. This is the LAST candidate set that the matching model actually scores. Every ID in matching_results must also appear here.
Rules for both: exactly one row per test S1 entity (empty list allowed); only S2-/S3- IDs that exist in the test set; no duplicate IDs within a list; no duplicate source1_entity_id rows; no self-matches to S1.
Validator (stdlib only), run from the student_resource/ directory:
  python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
It checks FORMAT ONLY. It does not compute a score.

## Final package (for the human, later)
zip with output/ (both TSVs), code/business_entity_resolution/{src/, README.md, requirements.txt}, Documentation_template.md. Code must reproduce both output files from the training/test data.

## Constraints
- Final model: MIT or Apache-2.0 licensed, up to 8B parameters.
- STRICTLY PROHIBITED: external data lookup. No commercial entity-resolution APIs/services, no government/registry lookups, no geocoding APIs to normalize addresses, no external data augmentation from internet sources. Violations = disqualification. Approaches and code are reviewed.

## Metric
F0.5 = (1.25 x P x R) / (0.25 x P + R), computed PER S1 ENTITY, then macro-averaged over all S1 entities in the evaluation set, singletons included.
Singleton (no true matches): predicting an empty list scores 1.0; predicting anything scores 0.0.
Example: predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> P = 2/3, R = 1.0, F0.5 = 0.714.
Leaderboard: public = subset of test, private = remainder; final ranking = private.

## Official tips
Blocking sets the recall ceiling. Use string-similarity features (Jaccard, Levenshtein, TF-IDF cosine) for names and addresses. Watch country-specific address patterns. F0.5 rewards precision over recall. Do not neglect singletons.
