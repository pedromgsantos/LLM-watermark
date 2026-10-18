# Stress-testing SynthID-Text like a user of Anthropic's detection API

Practical part of the article **"Is Claude's text watermark actually reliable?"** (Learn the Machine).

Anthropic says Claude's text watermark is a version of SynthID-Text (Dathathri et al., *Nature* 2024). Its key and detection API are private, so this project rebuilds the setup from the outside:

- a small open model plays **Claude** (writes with a watermark),
- a separate detector plays **Anthropic's detection API** (holds the key, receives plain text only, returns a verdict),
- the experiment code plays **the user** (copies, cuts, edits, rewrites and mixes text, then submits it).

> Scope: this is SynthID-Text with our key on a 2B model. It is not Anthropic's system. Results show the shape of each effect, not Claude's real numbers. Keep it simple: one model, one temperature, ~200 texts, four tests.

---

## 1. Design principle: match the real API experience

The whole point is that every test should look like what a real person would do with Claude and the future detection API. These rules are mandatory:

1. **The detector only receives a string.** `detect(text: str)` is the only entry point. No token ids, no prompt, no metadata, no knowledge of how the text was made. Experiments must not import anything from the generator module.
2. **The detector re-tokenizes from the raw string**, exactly as a service receiving pasted text would. Any loss from decode/re-encode, whitespace or punctuation changes is part of the result, not a bug to hide.
3. **Texts are what a user would copy:** only the assistant's reply, decoded with special tokens removed and stripped. Never the chat template, never the prompt.
4. **All user actions operate on strings** (truncate by words, swap words, paraphrase, translate, paste into a document).
5. **The detector returns an API-like response:** a score, a p-value and a three-level verdict (Section 5).
6. **Lengths are reported in words**, because that is how a user thinks about a text, not tokens.

Suggested structure that enforces this:

```
src/
  claude_sim.py   # model + watermark config + chat template -> generate(prompt) -> str
  api_sim.py      # class WatermarkDetectorAPI: holds key + tokenizer; detect(text) -> dict
  user_sim.py     # truncate_words, synonym_edit, paraphrase, translate, embed_in_document (strings only)
  data.py         # prompts, human texts, Wikipedia articles
  stats.py        # detection rate, false-positive rate, bootstrap CIs
  plots.py        # article figure style
experiments/
  test0_validation.py
  test1_length.py
  test2_edits.py
  test3_rewrites.py
  test4_dilution.py
configs/config.yaml
results/  figures/  data/
```

## 2. Stack

- Python 3.10+, PyTorch. Target hardware: Apple M5 (MPS), batched generation. A Colab notebook is optional.
- `transformers` with SynthID-Text support (>= 4.46): `SynthIDTextWatermarkingConfig` and `SynthIDTextWatermarkLogitsProcessor` (its g-value and context-repetition-mask helpers are used by the detector).
- `datasets`, `nltk` (WordNet), `numpy`, `pandas`, `scipy`, `matplotlib`.

**Before coding, check the installed `transformers` docs and source for the exact class, argument and method names.** Names here come from v4.46 and may have changed. If a needed helper does not exist, implement the g-value computation from the paper and document it.

## 3. "Claude" (generator)

| Setting | Value | Why |
|---|---|---|
| Model | `google/gemma-2b-it` | Small instruct model, runs locally |
| Chat template | Always | Real assistant usage |
| Temperature | 1.0 | Claude API default |
| top-k / top-p | Off (pure sampling) | Claude API default has neither set. Make configurable |
| max_new_tokens | 512 | Enough for ~300-word answers |
| Seed context | Last 4 tokens + key (`ngram_len=5`, verify) | Paper default (H = 4) |
| Tournament layers | 30 keys | Paper default (m = 30) |
| Competitors per match | 2 (non-distortionary) | Matches Anthropic's "no quality impact" claim. Verify HF does this |
| Repeated context masking | On | Paper default |

Generate each prompt twice: watermark on, and watermark off (same settings). Cache all outputs in `data/*.jsonl`.

## 4. Data

| Set | Size | Source |
|---|---|---|
| Prompts | 200 | Everyday assistant requests: ~80 explanations (ELI5 questions), ~60 emails or messages, ~60 short opinion pieces. Emails and opinions come from a small hand-written list of topics in `data/prompts.yaml` |
| Human texts | 200 | ELI5 human answers, 150+ words |
| Wikipedia articles | 100 + 100 | English Wikipedia, cut to ~1,500 words at paragraph boundaries. 100 for dilution, 100 human-only for calibration |
| Translation sources | 50 | Human ELI5 answers to be translated in Test 3 |

## 5. "Anthropic's API" (detector)

`WatermarkDetectorAPI(key, tokenizer).detect(text) -> {"score", "z", "p_value", "verdict", "n_scored_tokens"}`

- Tokenize the string (no special tokens).
- Compute g-values for every token and layer from the key and the previous 4 tokens. The first 4 tokens have no full context and are skipped; tokens whose context repeats earlier in the text are masked (same rule as generation).
- `score` = mean of the unmasked g-values.
- Under no watermark, g-values are Bernoulli(0.5), so with N = scored tokens × 30 layers: `z = (score - 0.5) / sqrt(0.25 / N)`, `p_value = 1 - Phi(z)`.
- Verdict: `p < 0.01` → "likely watermarked"; `0.01 <= p < 0.1` → "uncertain"; otherwise "not detected". Detection rate in the article = share of texts with "likely watermarked".
- Test 0 checks that this theoretical threshold really gives ~1% false positives. If not, report the empirical threshold (99th percentile of human scores) and use it everywhere, saying so in `summary.md`.

## 6. Tests

All figures: detection rate (share "likely watermarked") with 90% bootstrap CI (1,000 resamples). Each test writes `results/testX.csv` and `figures/testX.png`.

### Test 0 – Validation (run first)
- False-positive rate on the 200 human texts and the 200 unwatermarked model outputs.
- Detection rate on the 200 untouched watermarked outputs.
- Copy-paste check: the same watermarked texts after "plain handling" (strip markdown, curly quotes, normalize whitespace and line breaks). Expected: almost no change, since the words are the same.
- **Gate:** false positives near 1% and clear detection on full-length watermarked texts. If detection is near chance, stop and debug before running anything else.

### Test 1 – How much text does it need?
- Truncate watermarked and unwatermarked outputs at 25, 50, 100, 200 and 300 words (word boundaries, on the string).
- Figure: detection rate vs length in words, with the false-positive rate as a reference line.

### Test 2 – What if you edit it?
- Replace 10%, 30%, 50% and 70% of the words of each watermarked output with WordNet synonyms (skip stopwords, keep capitalization and punctuation; if a word has no synonym, pick another word).
- Figure: detection rate vs share of words edited.

### Test 3 – What if another model rewrites it?
On 50 watermarked outputs:
1. **Paraphrase:** same model, watermark off, prompt "Rewrite this text in your own words, keeping the meaning."
2. **Round trip:** EN → PT → EN, watermark off.
3. **Claude translates:** 50 human texts translated EN → PT by the watermarked model. Negatives: the same texts translated with the watermark off (use these to check the false-positive rate in Portuguese).
- Figure: detection rate per condition, with untouched text as the first bar.

### Test 4 – Two paragraphs in a twenty-page document
- Build documents of ~1,500 words: a Wikipedia article with one contiguous watermarked passage inserted at a paragraph boundary, at 2%, 5%, 10%, 25% and 50% of the words (concatenate watermarked outputs when one is too short).
- Score each document two ways:
  - **Whole document:** one `detect()` call, which is what an API would do with a pasted document.
  - **Sliding window:** windows of 200 words, stride 50 words, keep the minimum p-value. The max over windows inflates false positives, so **calibrate this threshold on the 100 human-only articles** to 1% false positives.
- Figure: detection rate vs share of watermarked words, one line per method.

## 7. Outputs

- `figures/test1.png` … `test4.png` (Test 0 can be a small table instead of a figure).
- `results/summary.md`: one line per test with the headline number, its 90% CI and one plain-English sentence. Every number must come from a CSV produced by code.
- Figure style: matplotlib, white background, one accent colour for the watermarked condition and grey for baselines, title that states the finding, axis labels with units, 1200 px wide.

## 8. Run order and compute

1. `--quick` mode first (20 prompts, everything end to end). Measure throughput and report the real estimate for the full run.
2. Generate and cache all texts (~650 generations, ~300k tokens; expected to fit in well under an hour on an M5, to be confirmed by the `--quick` run).
3. Test 0 → gate → Tests 1 to 4 → `summary.md`.

Fix all random seeds. Model, sizes, lengths and thresholds live in `configs/config.yaml`.

## 9. Known limits (for the article, not to solve)

Claude's key, settings, detector, thresholds and tokenizer are private; Claude is far larger and more heavily fine-tuned, so it likely has less freedom per word; the paper's trained Bayesian detector would do better than this simple test, so results are a lower bound; 200 texts show trends, not precise numbers.

## References

- Dathathri et al., Scalable watermarking for identifying large language model outputs, *Nature* 634 (2024). https://doi.org/10.1038/s41586-024-08025-4
- google-deepmind/synthid-text: https://github.com/google-deepmind/synthid-text
- Hugging Face, Introducing SynthID Text: https://huggingface.co/blog/synthid-text
- Anthropic, How Claude's text watermark works: https://www.anthropic.com/news/claude-text-watermark
