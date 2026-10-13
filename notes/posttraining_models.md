# Post-training / Models — Entity Resolution Challenge

## 1. Is fine-tuning a big win here? (honest answer)

**Yes — more than almost anything else on the ideas list.** Here's why, concretely:

- Off-the-shelf embedding models are trained for general semantic similarity ("the cat sat on the mat" ≈ "a feline rested on the rug"). That is a *different* problem from entity resolution, where "Sharma Trading Pvt Ltd" and "Sharma Trading Co." need to score high, but "Sharma Trading Pvt Ltd" (Pune) and "Sharma Trading Pvt Ltd" (Mumbai) need to score low *despite being nearly identical strings*. General embeddings are not optimized to separate these — they're optimized for meaning, not identity.
- Found direct proof of this while researching (see §4): a team fine-tuned a small model specifically for entity resolution and measured **5.2x reduction in score overlap** between true-match and non-match distributions, precision going from 84.2% → 88.6%, and false positives in auto-match going from 131 → 0, purely from fine-tuning — same architecture, same size, only the training target changed. That overlap reduction is exactly what a precision-heavy metric like F0.5 rewards.
- It's cheap to do at your scale: you're not pretraining anything, you're fine-tuning a 20-30M parameter model on pairs derived from your own train_ground_truth.tsv — this trains in minutes to low hours on a single GPU, not something that eats your one day.
- It directly produces the "final model" the problem statement requires (MIT/Apache, ≤8B) — this isn't a side activity, it can BE your core deliverable, with algo/blocking work feeding it candidates.

**When it's not worth it:** if your classic feature-based classifier (edit distance + Jaccard + TF-IDF + phonetic features → GBT) already gets you close to the recall/precision the blocking stage allows, marginal fine-tuning gains may not be the best use of remaining hours vs. polishing normalization/blocking. Decide this **after** you have a working baseline and can see where errors actually are (see agent error-bucket analysis in ideas.md) — don't fine-tune blind.

**Recommended framing:** fine-tuning is your second lever, not your first. Baseline (blocking + feature classifier) gets you a submittable, scorable pipeline fast. Fine-tuning a small embedding model on your own train pairs is the highest-value upgrade once that baseline exists, because the gains are concentrated exactly where F0.5 punishes you hardest — cleaning up the near-miss cases (same name, wrong location; abbreviated vs. full names) that heuristics alone leave ambiguous.

---

## 2. RL — checked, honest answer: **not the right tool, here's what's actually used instead**

I looked into this specifically rather than assuming. The research reality:

- **True RL (reward/policy-based sequential decision making) is not an established approach for entity resolution.** This is a static classification/ranking problem — there's no environment, no sequential actions with delayed reward, no state transitions that RL's formalism actually buys you anything for. Forcing RL onto this would add complexity without a corresponding win.
- **What people actually use that *looks* RL-adjacent:**
  - **Active learning** — the real workhorse in this space. Instead of a reward signal, a model iteratively picks the *most informative* unlabeled pairs to get labeled/confirmed next (usually the ones nearest its current decision boundary), retrains, repeats. This is supervised learning with a smart sampling policy, not RL — but it's the closest thing to "an agent iteratively improving itself" that's actually proven in ER literature (e.g. Shao & Wang, "Active Blocking Scheme Learning for Entity Resolution," PAKDD 2018; Jain, Sarawagi & Sen, "DIAL: Deep Indexed Active Learning," PVLDB 2022).
  - **DIAL specifically** is worth knowing about: it jointly trains the *blocker* and the *matcher* together in an active-learning loop, because their paper found training them separately with the same negative examples hurts one or the other — blocker needs to keep hard-negatives close (for recall), matcher needs to push them apart (for precision). That's a genuinely useful insight for your pipeline even if you don't implement full DIAL: **don't reuse the exact same negative pairs for tuning your blocking threshold and your final classifier.**
  - **Contextual bandits** could, in theory, be used for adaptive per-country threshold selection (treat "which threshold to use for this country's records" as a bandit arm-selection problem) — but this is genuinely niche and likely more engineering than the marginal gain justifies in one day. Mention in your methodology doc as awareness, don't build it.

**Bottom line:** drop RL as a direct method. Use **active learning framing** for iterating your training pairs (see §3 — this is actually what the long-horizon agent should do), and note the DIAL blocker/matcher-negative-separation insight in your feature/training pipeline.

---

## 3. Designing a long-horizon agent skill to do this

The best real-world template for this already exists and is documented in §4 (the arctic-embed-xs-entity-resolution model card) — it describes almost exactly the loop you'd want an agent to run. Here's the skill design, adapted to your constraints:

### Loop structure (what the agent repeats each round)

1. **Run current pipeline** (blocking + matcher) on the training data, using current model/feature weights.
2. **Harvest signal from the results**:
   - Confirmed correct matches (compared against train_ground_truth) → **positive pairs**
   - High-scoring pairs the model got wrong, or that landed near the decision threshold → **hard negatives** (this is the critical part — random negatives are too easy and don't improve precision; near-miss negatives are what actually sharpens the boundary)
3. **Retrain/fine-tune** on the accumulated pair set (not just this round's — accumulate across rounds so the model doesn't forget earlier hard cases).
4. **Evaluate on a fixed holdout split** (carve this out once, up front, and never train on it — otherwise you can't trust round-over-round comparisons).
5. **Log the metric that matters** (validation F0.5, or score-overlap between match/non-match distributions — the arctic-embed card shows overlap is often a cleaner signal to watch than F1/F0.5 alone during training, since it tells you if the threshold is getting easier to set).
6. **Repeat.** Stop when the metric plateaus (the reference case converged around round ~17-22 of 26 attempted — diminishing returns are real, don't over-round on a hackathon clock).

### Concretely, as an agent skill/task spec:
- **Goal**: minimize score-overlap between match/non-match pair distributions on a fixed holdout, subject to not exceeding N rounds or T minutes of wall-clock time.
- **Tools it needs**: run blocking+matching pipeline, compute pairwise features/scores, compare against ground truth, extract hard negatives (score above threshold but label=no-match), fine-tune step (LoRA on a small embedding model is cheap enough to do many rounds — full fine-tune is not, given time budget), log metrics per round.
- **Stopping condition**: plateau detection (metric improvement < some epsilon for 2 consecutive rounds) or hard time budget — whichever first.
- **Human-in-loop checkpoint**: after each round, a one-line summary (precision/recall/overlap delta) so you can kill it early if it's not converging, rather than discovering wasted hours at the end.

This is a genuinely good use of an agent for tomorrow — it's exactly the "exhaustive but focused" execution you were describing, bounded by a clear metric and stopping rule so it doesn't wander.

---

## 4. Open-source models researched (HuggingFace + GitHub)

### Purpose-built for this exact task
**`themelder/arctic-embed-xs-entity-resolution`** (HuggingFace) — this is worth knowing about directly, found via research, not from memory:
- Fine-tuned version of `Snowflake/snowflake-arctic-embed-xs`, license **Apache-2.0**, **22M params** (well under your 8B cap), 384-dim embeddings, BERT architecture (6 layers).
- Trained *specifically* for matching a clean reference dataset against noisy operational records — same shape as your problem (S1 = reference, S2/S3 = noisy).
- Its own training methodology is literally the loop described in §3: iterative LoRA fine-tuning rounds (22 of them), each round mining hard negatives from its own review queue and retraining — the model card documents this in detail and is a genuinely good reference/template to read before you build your own version.
- Reported results (on their own benchmark, take with appropriate caution since it's their own eval): precision 88.6%, combined recall 99.7%, and — the metric that maps most directly to your problem — reduced false positives in auto-match from 131 to 0 vs. the untrained base model, on 10k×10k synthetic entity pairs.
- Explicitly documents its own limitation: **cannot resolve acronym-to-full-name matches** (e.g. "TRMS" → "Taylor, Reeves and Mcdaniel SRL") — recommends a composite scoring approach (embedding + fuzzy string + keyword) for exactly this case. This matches what you'd expect and reinforces: **don't rely on embeddings alone, combine with string-edit features** — same conclusion as the ML section in ideas.md, now confirmed by someone else's actual production numbers.
- Note: this specific model's training domain was financial counterparty names/addresses, not generic business records — so treat it as a **methodology reference and possibly a base to further fine-tune on your own data**, not a drop-in solution you can zero-shot on this dataset and expect great numbers from.

### Their base-model comparison (also from the same source, useful data point)
They tested 4 base embedding models before fine-tuning any of them further:

| Model | Params | Best score-overlap after fine-tune | Combined recall |
|---|---|---|---|
| all-MiniLM-L6-v2 | 22M | (baseline only, not fine-tuned in their test) | — |
| BAAI/bge-small-en-v1.5 | 33M | 0.070 | 97.3% |
| BAAI/bge-base-en-v1.5 | 110M | 0.046 | ~98.5% |
| Snowflake/snowflake-arctic-embed-xs | 22M | **0.031** | **99.7%** |

Takeaway: bigger isn't automatically better for this kind of short-text matching — arctic-embed-xs won despite being the smallest, likely because its pretraining already included heavy hard-negative mining. Worth trying arctic-embed-xs as your base if you fine-tune your own bi-encoder, rather than defaulting to a bigger/more "famous" model.

### General entity-matching architecture reference
**Ditto** (`megagonlabs/ditto`, GitHub) — academic but real and well-cited (VLDB 2021 paper, up to 96.5% F1 on a 789K×412K real company-matching task):
- Casts entity matching as a **sequence-pair classification** problem: serialize each record pair into one text sequence with `COL`/`VAL` tokens marking fields, feed into a fine-tuned BERT/DistilBERT/RoBERTa.
- Three optimizations worth stealing conceptually even if you don't use their code directly: (1) **domain knowledge injection** — highlight important tokens (e.g. mark the matched-length substring) before feeding to the model, (2) **summarization** for long fields so the model doesn't waste capacity on irrelevant text, (3) **data augmentation** — specifically generating harder training examples, same spirit as hard-negative mining above.
- Caveat: it explicitly separates blocking (their paper says use "simple heuristics" for that) from matching (where Ditto focuses) — same two-stage shape as your pipeline. It's a matcher, not a blocker — pair it with your BK-tree/automaton blocking from ideas.md, don't expect it to replace that stage.
- Base models (BERT, DistilBERT, RoBERTa) are Apache-2.0 licensed, all comfortably under the 8B param cap.

---

## 5. Compliance note (important, don't skip)

Downloading open-source model weights (Ditto's base LMs, arctic-embed-xs, etc.) and running them **locally/offline** to score your own candidate pairs is fine — that's just using a pretrained model, same as using BERT for anything else, and it satisfies the "MIT/Apache, ≤8B" final-model requirement directly.

What's **not allowed** is calling any live external API/service/database *during inference* to look up or verify a business's identity (geocoding APIs, business registries, commercial ER APIs, etc.) — that's the explicit "external data lookup" prohibition in the problem statement. Fine-tuning locally on the provided training data, and running a downloaded model locally at inference time, does not touch that line. Just don't wire up anything that phones out to the internet as part of your scoring pipeline.

---

## Summary / what to actually do tomorrow given all this

1. Build baseline first (blocking + feature classifier) — gets you a submittable score.
2. Once baseline works, fine-tune a small bi-encoder (arctic-embed-xs or bge-small as base) on pairs from your own train_ground_truth — this is your highest-leverage upgrade, confirmed by real numbers from someone who did almost exactly this task.
3. Use the round-based hard-negative-mining loop (§3) as your fine-tuning methodology — and as the literal agent skill if you want to automate it.
4. Skip RL as a direct method; borrow the DIAL insight (don't reuse blocker negatives for matcher tuning) if you have time.
5. Combine embedding similarity with string-edit + phonetic features regardless of how good the embedding model gets — every source confirms pure-embedding approaches have a real, acknowledged failure mode (acronyms, abbreviation-heavy names) that only heuristic features catch.
