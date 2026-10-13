# Cross-encoder / transformer scorer: notes for GPU exploration

Status: exploration only, not started, not approved for the shipped pipeline. This is a real model-family
change (LightGBM tree ensemble → neural network), so per `AGENTS.md` it needs explicit team sign-off
before it goes into a submission, even if it works. Prototyping it is fine; shipping it is a separate
decision.

---

## 1. The rules question first, because it's the one that actually matters

Contest spec (`notes/problem_statement.md`): *"Final model: MIT or Apache-2.0 licensed, up to 8B
parameters"* and *"STRICTLY PROHIBITED: external data lookup. No commercial entity-resolution
APIs/services, no government/registry lookups, no geocoding APIs, no external data augmentation from
internet sources."*

My read, with reasonable confidence but **not certainty**: a small pretrained transformer checkpoint,
downloaded once, run entirely offline, with its OWN weights licensed MIT or Apache-2.0, is allowed. The
"final model must be licensed, ≤8B params" line only makes sense as a rule if pretrained models are in
scope — there'd be nothing to license or count parameters of otherwise. The ban is about **hosted
services doing the work for you at inference time** (a paid API call) and **augmenting your training
data from the internet** (scraping extra business records), not about using a general-purpose pretrained
language model you run yourself, on your own hardware, on the contest's own data.

**Do not treat this as settled.** If there's an official rules channel or FAQ for this contest, this is
worth a direct, explicit question before investing real GPU time: *"May we fine-tune an open-weight,
locally-run pretrained transformer checkpoint on the contest's own training data?"* If the answer is no,
everything below is moot. Getting this wrong is exactly the kind of thing that risks disqualification,
which is worse than getting a lower score.

**Hard no's regardless of the answer above** (`AGENTS.md` rule 2, unambiguous):
- No calling any hosted model API (OpenAI, Anthropic, a hosted Hugging Face inference endpoint, etc.) —
  the model must run on your friend's own GPU, offline, no network calls during training or inference.
- No using the checkpoint to look up real business data, geocode addresses, or pull in outside sources.
- Whatever checkpoint is used, its own license must be checked and recorded (this project keeps a running
  license file at `reports/licenses.md` — add an entry there if this ever ships).

---

## 2. What problem this would actually target

The current matcher (LightGBM, 68 hand-built features: rapidfuzz ratios, TF-IDF cosine, Monge-Elkan,
phonetic codes, address-shape features, pool-uniqueness counts) already separates most pairs cleanly:
ROC-AUC 0.9975, PR-AUC 0.9985 on MATCH_VALID (`match_eval_MATCH_VALID.json`). The place a transformer
could plausibly add something new is the **uncertain band** — pairs whose calibrated probability lands
away from both 0 and 1, where the hand-built features don't cleanly separate true from false matches
(e.g. genuine paraphrase in how an address is written, or subtle semantic equivalence engineered features
miss). It is not likely to beat 68 well-chosen features on the easy majority of pairs — most contest
teams' gains from this kind of addition come from the hard tail, not a blanket replacement.

**Recommended framing: don't replace the matcher, add to it.** Train the transformer as its own scorer,
then feed its output probability into the existing LightGBM model as ONE more column — exactly the same
way stage 2's `p_rerank` already feeds into stage 3 today. This keeps the whole thing inside this
project's existing adoption methodology (same 5-fold OOF, same paired-CI rule, same guard segments) rather
than requiring a full architecture swap and its own separate validation story. A full replacement is
higher risk and harder to validate cleanly; a stacked feature is a much smaller, testable claim.

---

## 3. Candidate checkpoints (license-checked, all small enough to be fast even on a modest GPU)

| Model | Params | License | Language coverage | Note |
|---|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | Apache-2.0 | 50+ languages | Best default: built for semantic similarity, multilingual (relevant since test includes France, train includes India). |
| `distilbert-base-multilingual-cased` | 134M | Apache-2.0 | 104 languages | Solid general-purpose fallback if the sentence-transformer doesn't fine-tune well as a cross-encoder. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~23M | Apache-2.0 | English only | Already a cross-encoder architecture (built for exactly this shape of task: score how well two texts match), but English-only — see the normalization point below for why that might still work. |

Stay in the 20M–150M parameter range. There is no benefit to going toward the 8B ceiling here — the
inputs are two short text fields, not documents, and a bigger model just means slower inference for no
expected gain on this task.

**Useful shortcut: this pipeline already romanizes non-Latin script.** `textnorm.py` already converts
Indic-script names into Latin text before anything else touches them (`name_norm`/`addr_norm`). That
means an English-only cross-encoder fed the *normalized* text (not raw) may work fine without needing a
multilingual checkpoint at all — worth testing both ways rather than assuming multilingual is required.

---

## 4. How to structure it, concretely

**Reference architecture:** this is the "Ditto" pattern (Li et al., 2020, entity matching with pretrained
LMs) — serialize each record as text and let the transformer read both together:
```
COL name VAL Miller Interiors LLC COL addr VAL 18563 Laurie Lane, Waddell, AZ COL country VAL US
[SEP]
COL name VAL Miller Interiors LLC COL addr VAL nan COL country VAL US
```
Feed the concatenated pair through the transformer, take the `[CLS]`/pooled output, add a small
classification head (one linear layer), fine-tune with binary cross-entropy against the same labels this
project already uses.

**Data — use the exact same slices this project already uses, for the exact same leakage reasons:**
- Fit on `MATCH_TRAIN`'s candidate pairs (`work/candidates/MATCH_TRAIN/`, joined against
  `work/records/train_gt.parquet` for labels) — 975,699 rows, positive rate 0.6215, or the larger
  `MATCH_ALL` set if that gets adopted from the data-scaling work in progress.
- Confirm on `MATCH_VALID` (20,000 S1) — never tune on it, only check.
- **Never touch validation-S1 (`HOLDOUT_TEST`) or test labels.** Same rule as everything else in this
  project — those are scoring-only, once, at the very end.
- Use the same 5-fold `GroupKFold` grouped by `s1_idx` (`fold_seed=42`) that the rest of the matcher uses,
  so results are directly comparable and combinable with the existing pipeline.

**Before a big investment, do a cheap sanity check first:** fine-tune on a subset (50k–100k pairs), check
its standalone AUC/logloss against LightGBM's existing 0.9975 ROC-AUC. If it's clearly worse and highly
correlated with the features that already exist (e.g. it just rediscovers `n_tfidf_cos`), it's unlikely to
add much as a stacked feature. If it's capturing something the existing features don't — check by looking
at where it disagrees with the current model, specifically on the pairs LightGBM is least confident about
— that's the actual signal worth chasing.

---

## 5. What "success" looks like here

Same bar as every other experiment in this project (`plan/FEATURE_EXPERIMENTS.md` §2.6): add the
transformer's probability as one new column to the existing 68-feature LightGBM matcher, retrain, and
require a paired OOF gain of **at least +0.0005** with the 95% confidence interval entirely above zero, no
guard-segment (`country_US`, `country_India`, `singleton`) regression worse than 0.002. Nothing gets
adopted into the shipped pipeline without clearing that bar and without an explicit go-ahead, since this
is a model-family change.

CPU inference cost matters for the actual submission even if training happens on a GPU: test has
~10.6 million candidate-pair rows (`match_predict_TEST.json`, 5.4M matched pairs at submission 1's rate).
If this is ever adopted, inference needs to run in reasonable time on whatever hardware produces the final
submission — batch it, use fp16, and consider running it only on pairs the LightGBM model is least
confident about (mirroring the "uncertain band" framing above) rather than all 10M+ rows, to keep
inference cost bounded.
