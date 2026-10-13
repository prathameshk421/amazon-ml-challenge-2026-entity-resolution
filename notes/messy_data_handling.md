# Handling Messy Data — Literature Deep Dive for Entity Resolution

Went and actually checked the papers/systems that deal with exactly this kind of dirty-data matching problem, rather than guessing. Organized by what kind of messiness they tackle.

---

## 1. The foundational survey (start here if you want the full landscape)

**Elmagarmid, Ipeirotis, Verykios — "Duplicate Record Detection: A Survey"** (IEEE TKDE, 2007, ~785 citations — the most-cited paper in this exact space)

Covers, comprehensively:
- Why duplicates happen in the first place: transcription errors, incomplete info, lack of standard formats, or combinations of these — literally the same causes listed in your problem statement.
- Field-level similarity metrics (edit distance, token-based, phonetic) and when each works/fails.
- An extensive set of matching algorithms, plus efficiency/scalability techniques (this maps directly to your blocking stage).
- Concludes with open problems — worth skimming since some are still unsolved and might explain why certain pairs in your dataset are genuinely hard.

This is the one paper to skim first if you have 20 minutes tonight — it's the map of the whole territory.

---

## 2. Real-world comparative evaluation — what actually works vs doesn't

**Köpcke, Thor, Rahm — "Evaluation of Entity Resolution Approaches on Real-World Match Problems"** (VLDB 2010)

Directly relevant finding, worth internalizing: they tested both ML-based and non-ML approaches (plus a commercial ER tool) on real messy datasets and found **significant quality differences between approaches, and that some tasks — like matching product entities across sources — are NOT sufficiently solved by conventional attribute-similarity approaches alone.** That's an important, humbling data point: don't assume that stacking more similarity features will get you arbitrarily close to perfect. Some genuinely ambiguous cases will remain, and that's expected, not a bug in your pipeline.

They also released benchmark datasets (Abt-Buy, Amazon-Google Products, DBLP-ACM, DBLP-Scholar) that are the standard testbeds this whole field calibrates against — good context for what "good" F1/F0.5 numbers even look like in practice (these tasks typically see F1 in the 60-90% range depending on domain, not high-90s) — so calibrate your expectations for score accordingly, this is a genuinely hard task class, not one where near-perfect is the norm.

---

## 3. Name-specific noise — the transliteration/phonetic problem (very relevant given India is in your training data)

This is a well-studied, specific sub-problem, and it's worth taking seriously rather than treating as "just run Soundex":

- **Standard Soundex is English-biased and performs poorly on Indian names.** Found a specific paper on this: *"Need for Customized Soundex based Algorithm on Indian Names for Phonetic Matching"* (Jaisunder, Ahmed, Mishra — Global Journal of Enterprise Information System, 2016) — built out of India's own National Informatics Centre, motivated by exactly this problem in government digitization records. Their point: off-the-shelf Soundex doesn't capture Indian name phonetics well; a customized variant is needed.
- **Devanagari transliteration is genuinely ambiguous, not just noisy.** Found a directly relevant field note: a single Devanagari name can transliterate to multiple valid Latin spellings (example given: भेरु → "Bhairoo" or "Bheru") — these are the *same* name in the original script but look unrelated after transliteration. A source (IDinsight, a research/data org) states plainly that **off-the-shelf fuzzy matching (Stata's reclink, generic packages) performs poorly here and needs dataset-specific customization** — this is a direct validation of "don't hardcode generic rules, build from what you see in this dataset."
- **Editex algorithm** — came up in a related NLP paper on cross-lingual name matching (Nayan et al., IJCNLP 2008 workshop) — a hybrid that combines edit-distance with phonetic grouping (characters that sound alike get near-zero substitution cost). Worth trying as a middle ground between pure edit distance and pure phonetic codes — more nuanced than Soundex, cheaper than full transliteration modeling.
- Real production systems (e.g. Indian fintech/BFSI identity platforms) report the exact same noise categories your problem statement lists: name order swaps (surname-first vs given-name-first), initials vs full names, honorifics (Shri/Smt.), regional spelling variants (Shekhar vs Sekhar). Confirms your training data's noise patterns aren't synthetic edge cases — they mirror real deployed-system pain points.

**Practical takeaway**: for the India-heavy portion of your data, plain Levenshtein/Jaro-Winkler on transliterated names will systematically miss a category of true matches that look completely different but are phonetically/script-wise identical. A phonetic feature (even a rough customized Soundex-for-Indian-names, not the stock English one) is not optional flavor — it's addressing a documented, named failure mode.

---

## 4. Address-specific noise — the structural/parsing problem

**libpostal** (openvenues/libpostal, ~4,500 GitHub stars) — the most mature open answer to exactly your "address abbreviations, missing components, landmark references, component reordering" problem:
- Uses Conditional Random Fields trained on over a billion real-world addresses (OpenStreetMap + OpenAddresses) to parse and normalize addresses across 60+ languages/countries.
- Handles almost exactly the noise categories your problem statement lists: abbreviation expansion ("St" → "Street"), numeric expression parsing, transliteration, and severe format variation (their own examples span US, French, Russian, German, Spanish address conventions).
- Reports 98.9-99.45% parse accuracy on held-out data — genuinely strong, mature tool, not a toy.
- **Compliance flag, read carefully**: this is a pretrained model, same category as using BERT — but unlike BERT (trained for general language understanding), libpostal's entire value is *encoded real-world address knowledge* (it was trained specifically on address data from OpenStreetMap). Given your rules explicitly prohibit "external data augmentation from internet sources" and "geocoding APIs to normalize addresses," a model whose core skill *is* normalizing addresses using knowledge distilled from real address databases sits in a genuine gray zone — it's not calling an external API live, but its underlying training data is exactly what you're told not to bring in. **Worth asking Amazon's organizers directly rather than assuming either way** — the safer default if you can't check in time is to build your own address normalization rules/FST from patterns you observe in the training/test data itself, which is unambiguously safe and was your original plan anyway.

---

## 5. Deep learning approach to dirty product/business data specifically

**Ditto** (already covered in posttraining_models.md, repeating the relevant point here): the paper explicitly demonstrates going from noisy structured records to 96.5% F1 on a *real* large-scale company-matching task (789K × 412K records) — this is strong direct evidence that pre-trained-LM-based matching handles this category of noise well at scale, once fine-tuned. Their "data augmentation" optimization specifically manufactures harder training examples (character-level noise injection, token shuffling) to force the model to generalize past surface-level string differences — directly applicable to your abbreviation/typo/reordering noise categories.

---

## 6. Practical synthesis — what this means for your pipeline tomorrow

| Noise type from problem statement | What the literature says to actually do |
|---|---|
| Abbreviations, legal suffix differences (Corp/Corporation) | Rule-based normalization dictionary, built from EDA on your own data — Elmagarmid survey confirms this is standard first-pass cleaning, not something requiring ML |
| Typos | Edit distance / Levenshtein, well-established, low-risk |
| Transliteration variants (Indian names) | Plain edit distance under-performs here — literature specifically flags this as needing customized phonetic handling (Editex or a customized-for-Indian-names Soundex variant), not generic Soundex |
| Word-order transpositions | Token-set similarity (Jaccard on token sets), not sequence-based edit distance |
| Address abbreviations, missing components, landmarks | Component-wise comparison after parsing; libpostal is the mature tool but has a compliance gray zone — build your own lightweight parser/normalizer from observed patterns as the safe default |
| "Some tasks aren't solved by attribute similarity alone" (Köpcke et al.) | Don't expect near-100% — budget time for genuinely ambiguous cases, and make sure your precision-heavy F0.5 strategy defaults to "no match" on true toss-ups rather than guessing |

**One more honest note**: multiple sources here independently converge on the same message — generic, off-the-shelf fuzzy matching tools consistently underperform on this exact category of noise (Indian name transliteration, business name legal-suffix variation) *specifically because* they weren't built with these patterns in mind. That's not a reason to panic — it's validation that your planned approach (derive normalization/matching rules from actually looking at *this* dataset's specific noise, rather than trusting a generic library's defaults) is the correct instinct, and the literature backs it up directly.
