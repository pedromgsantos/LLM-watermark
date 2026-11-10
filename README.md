# SynthID-Text Watermark Quick Test

This repository currently contains a single exploratory notebook, [watermark_quick_test.ipynb](watermark_quick_test.ipynb). It tests how a SynthID-Text-style watermark behaves under a small set of text edits and rewrites. It is an informal demonstration, not a benchmark or a reproduction of Claude's private watermarking system.

## Run the notebook

1. Open `watermark_quick_test.ipynb` in Jupyter or VS Code.
2. Run the cells from top to bottom. The first code cell installs `transformers>=4.46`, `accelerate`, `datasets`, and `matplotlib`.
3. The notebook selects CUDA, Apple MPS, or CPU automatically and downloads `Qwen/Qwen2.5-1.5B-Instruct` (the watermarked "Claude"), `HuggingFaceTB/SmolLM2-1.7B-Instruct` (the "other AI" used for rewrites) and the Wikipedia dataset when their cells run. This requires network access and enough disk space for the models and dataset.
4. The generation cell creates 30 watermarked and 30 unwatermarked answers and writes them to `generations.json` in the current working directory. With `REGENERATE = False` (the default in the notebook) it reads that file when it exists instead of generating again, so the Qwen generation is only done once. Set `REGENERATE = True` to generate new answers and overwrite the file.

The notebook needs PyTorch as well. If it is not already installed in the selected Python environment, install a PyTorch build suitable for that environment before running the notebook.

## What the notebook tests

- **Baseline:** compares watermarked model outputs, unwatermarked outputs, Wikipedia paragraphs, and optional writing supplied in `MY_TEXTS`.
- **Length:** measures detection on watermarked and unwatermarked outputs truncated to 25, 50, 100, 150, 200, or 300 words.
- **Edits:** replaces 0%, 10%, 30%, 50%, or 70% of words with randomly selected words from a Wikipedia-derived pool. This is a crude word-replacement test, not synonym substitution.
- **Manual edits:** prints three watermarked outputs for the user to edit, then compares each edited text with its original.
- **Model rewrites and translation:** a second model from another family and of similar size, `SmolLM2-1.7B-Instruct`, has no key and no watermark. It paraphrases 15 watermarked answers and runs an English-to-Portuguese-to-English round trip on them. Separately, the watermarked Qwen translates 15 Wikipedia paragraphs into Portuguese, with a control run of the same Qwen with the watermark off. SmolLM2 is mostly an English model, so the Portuguese leg of the round trip may be weak; the paraphrase alone already shows the effect. Rewrites are capped at 300 new tokens.
- **Document dilution:** places a watermarked passage in a 1,500-word Wikipedia-derived document at 2%, 5%, 10%, 25%, or 50%. It compares whole-document scoring with the highest score among 200-word windows at a 50-word stride. The window cutoff is the maximum score observed on 30 human-only documents; it is not a calibrated 1% false-positive threshold.

The notebook displays results and plots in the notebook.

## Watermark configuration

The notebook configures the Transformers SynthID-Text implementation with `ngram_len=5`, 30 integer keys, a sampling table of size `2**16`, sampling-table seed `0`, and context history size `1024`. An n-gram length of 5 means the four preceding tokens form the context for the next token. The 30 keys represent the tournament layers.

The key list is sampled deterministically with `random.Random(42)` from the small range 1 to 9,999. This makes the experiment repeatable, but it is not a secret production key. The sampling-table and context-history values are implementation parameters exposed by Transformers; they should not all be described as paper-specific defaults.

## Detector and the 2.33 threshold

The notebook re-tokenizes the text, obtains the 0/1 g-values, and ignores repeated contexts. Its detector computes

`z = (mean_g - 0.5) / sqrt(0.25 / N)`,

where `N` is the number of retained g-values across token positions and watermark layers. Under the unwatermarked null, the g-values are treated as independent Bernoulli(0.5) observations. The threshold `2.33` is the rounded 99th percentile of the standard normal distribution: `P(Z >= 2.33) ≈ 0.0099`. Thus it is a one-sided, normal-approximation p-value cutoff of about 1% under those assumptions. It does **not** mean there is a 1% probability that a flagged text is human-written.

The value `2.33` comes from the standard normal quantile, not from the SynthID-Text paper. This notebook uses a simple z-test; it does not implement the paper's trained Bayesian detector or the exact-binomial frequentist detector in Google's reference repository. Treat the 1% as a nominal threshold and inspect the false-positive rates the notebook actually observes on its unwatermarked and human comparison texts. Those results are not a calibration guarantee for every corpus or fixed key.

## Scope and limitations

- The model is Qwen 2.5 1.5B, not Claude. The watermark key, detector, tokenizer, and thresholds used by Claude are private and are not reproduced here.
- The sample is small: 30 prompts, a limited number of human paragraphs, and 30 documents for the window cutoff. Results indicate trends; they are not precise performance estimates.
- The watermark configuration is compatible with the SynthID-Text approach, but the experiment and detector are not an exact reproduction of the paper's full setup.
- Changing the installed Transformers version may change implementation details. Keep generation and detection on the same configuration and tokenizer.
- Generate and detect on the same device. The random table behind the key differs between devices (for example MPS and CPU), and a mismatch makes a watermarked text look unwatermarked. The notebook has a check for this that must print `True`.
- Memory: with both models loaded, the notebook is heavy for a 16 GB machine. The models are loaded in `bfloat16` to halve their size. If the kernel dies or the system starts swapping, restart the kernel, close other apps, and rely on `generations.json` instead of generating again.

## References

- Dathathri et al., ["Scalable watermarking for identifying large language model outputs"](https://doi.org/10.1038/s41586-024-08025-4), *Nature* 634 (2024).
- Google DeepMind, [SynthID Text reference implementation](https://github.com/google-deepmind/synthid-text), including its [tournament watermarking and detector description](https://github.com/google-deepmind/synthid-text/blob/main/docs/tournament.md).
- Hugging Face, [Introducing SynthID Text](https://huggingface.co/blog/synthid-text) and [Transformers v4.46 SynthID configuration documentation](https://huggingface.co/docs/transformers/v4.46.0/en/internal/generation_utils#transformers.SynthIDTextWatermarkingConfig).
- SciPy, [standard normal distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html), whose 99th percentile gives the rounded `2.33` cutoff.
- Anthropic, [How Claude's text watermark works](https://www.anthropic.com/news/claude-text-watermark).
