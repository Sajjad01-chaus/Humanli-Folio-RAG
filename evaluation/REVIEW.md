# Evaluation review

## Initial 30-case policy run

`policy-baseline-results.json` records the initial live GPT-OSS 120B run before migrating the same
bounded corrective/reflective flow into LangGraph and before adding one-revision recovery.

- 30 completed requests; zero request errors.
- Expected page recall at five passages: 1.0 for BM25 and hybrid retrieval on this tiny corpus.
- Expected-term diagnostic: 24/26 answerable questions (92.31%).
- All four unanswerable cases abstained.
- One answerable case abstained (case 09).
- Mean cited expected-page recall: 0.9615.
- Median end-to-end latency: 27.12 seconds on the local CPU/provider configuration used for the run.

These are authored regression cases. BM25 and hybrid both reached the page-recall ceiling; this run
does not establish a hybrid-retrieval advantage. Expected-term matching does not prove correctness.

## Review of the two failed term checks

**Case 16:** The model correctly compared five minutes in 2025 with *fifteen* minutes in 2024, citing
both policies. The diagnostic expected `15`, so this was a term-matching false negative. The raw
result is preserved rather than silently edited to improve the score.

**Case 09:** The model's reflection removed its sole legal-hold claim, producing an unnecessary
abstention despite retrieving the relevant page. That is a real failure. The final graph allows one
shorter revision and verifies it again; if still unsupported, it abstains. A behavioral test ensures
the loop terminates even if every revision fails. A targeted live regression run is recorded separately.

## Public reports

The public corpus contains two original NIST reports, 112 physical PDF pages in total. The six
authored cases test core functions, voluntariness, a definition, cross-document synthesis, a contextual
follow-up, and an out-of-corpus question. `public-results.json` records actual live outputs and
metrics when that run completes. Reference pages are not an exhaustive list of all valid passages.

An earlier concurrent test encountered Groq HTTP 429 errors. Evaluations are run sequentially for the
recorded reports; the application reports provider errors rather than generating a placeholder answer.

The initial six-case public run completed without errors. Expected-term checks passed 4/5 answerable
cases; the unavailable question abstained. Hybrid expected-page recall was 0.8 versus 0.6 for BM25,
but the compound question had zero exact-reference-page recall and returned a partial answer. That
exposed a real multi-part retrieval weakness. The subsequent graph adds focused subquestion searches
and preserves their evidence during correction. Targeted runs are separate artifacts, not replacements
for the initial failures.

`compound-final-results.json` records the focused question-form planner retest with GPT-OSS 20B:
the answer includes both requested facts and passes the expected-term diagnostic. Exact reference-page
recall is 0.5 because other purpose passages were cited rather than the single annotated purpose page.
This is one targeted regression after development, not a new six-case benchmark or a controlled
ablation. The original six-case public report preserves the retrieval failure.
An earlier development run also exposed an optional note containing the literal string `None`. Final
response parsing normalizes empty-note sentinels, with a regression test. The final targeted artifact
records the latest post-refactor run; the initial full-corpus reports remain unchanged.

## Final policy regression

`final-regression-results.json` contains eight targeted cases executed with **GPT-OSS 20B**, rather than
120B. All eight completed with no errors. Expected-term checks, cited expected-page recall, and the
included unanswerable abstention check passed. This includes the earlier legal-hold case. Because the
model and workflow changed, this is not a controlled causal comparison with the initial run. It is a
regression check of the final path. The earlier “fifteen” diagnostic mismatch did not recur in this run.

Groq's per-minute token quota was observed during testing. Final code adds per-process token-aware
pacing, a bounded context budget, and explicit provider failures. Separate processes still compete for
the same account quota. An interrupted 120B follow-up regression is not counted as a successful run.

## Validity limits

- The author created these questions after selecting the documents; this is not a blind benchmark.
- The same LLM generates and verifies claims; correlated errors remain possible.
- Source-page recall is neither passage precision nor citation entailment.
- Scanned-document OCR accuracy and complex-table reconstruction have not been quantitatively benchmarked.
- Unit/integration test counts describe software behavior, not factual answer accuracy.
- No state-of-the-art accuracy or general production-readiness claim is made.
