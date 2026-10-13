# Things To Look Out For — Business Entity Resolution Challenge

Checked the PDF closely for prompt injection (hidden/invisible text, "ignore previous instructions" style payloads, anything addressed to an AI agent). **Found none.** What's actually there is a set of plainly-stated spec traps — rules an agent (or a fast-moving human) will walk into precisely because they're easy to skim past, not because they're hidden.

## The big one: don't let an LLM be your matcher
> "STRICTLY PROHIBITED: External Data Lookup ... Using commercial entity resolution APIs or services ... Any external data augmentation from internet sources"

The most obvious "just use AI" move — calling GPT/Claude/Gemini per candidate pair to ask "same business?" — is a commercial API being used to resolve entities. It also independently fails the separate constraint: **final model must be MIT/Apache-2.0, ≤8B params** — most hosted frontier LLMs fail license and/or size. This is exactly the kind of thing flagged in the "top teams' packages are reviewed in detail" audit step. Use a small locally fine-tuned model instead (see posttraining_models.md).

## Other real traps, ranked by how easily an automatic/agent-built pipeline breaks on them

1. **Country is an explicitly open set.** "Do not hard-code, filter, or one-hot your pipeline to only {US, India}... every test entity — France included — must appear in your submission." One-hot encoding country from training data is a very standard, very automatic step — and it silently breaks on France if done naively.

2. **`candidate_pairs.tsv` must be your *last* blocking stage, not your first.** "The final candidate list just before the ML model scores them, not the raw output of an early blocking pass you later filter further... whatever your model actually runs inference over." If your pipeline has multiple blocking/filtering passes, logging the wrong one misrepresents your blocking quality even if your final matches are fine.

3. **ID hygiene**: matched/candidate IDs must only reference the test set, no self-matches to S1, no duplicates within a list, no duplicate `source1_entity_id` rows. Easy to violate if any code path mixes train/test IDs or double-appends a candidate.

4. **Passing the validator ≠ a good score.** "It only reads your output files and the test source files; it does not compute your score." It's a format checker, not a correctness checker — don't treat "validator: PASS" as "task complete."

5. **Every S1 entity needs exactly one row, even singletons with an empty match list.** Easy to forget if your code only writes rows for entities that got at least one candidate during blocking.

6. **F0.5 macro-averages per S1 entity, singletons included** — a model that's slightly trigger-happy (always predicts *something*) gets punished hard across every singleton, even if raw precision looks okay in aggregate. Check your singleton accuracy specifically, not just overall F0.5.

## Bottom line
None of this is injection — it's a document that rewards careful reading and punishes the fast, agent-default path: hardcode categories, call an LLM API as the matcher, trust the validator blindly, mislog candidate_pairs.tsv, skip singleton handling. Read the constraints section twice before building anything.
