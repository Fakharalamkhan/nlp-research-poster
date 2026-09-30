# Poster numbers and where they come from

Paths are relative to `experiments/` in this repository.

| Poster location | Number(s) | Source |
|---|---|---|
| Methodology: Setup | 500 questions each; 100 dev / 400 test | `cgsc/data/splits.json` (n_sample 500, n_dev 100, n_test 400) |
| Methodology: Setup | 2× NVIDIA T4 | `results/environment.json`; `kaggle/kernel-metadata.json` (machine NvidiaTeslaT4) |
| Methodology: Signals | N = 10 samples | `cgsc/run.py` (seeds 42–51) |
| Results (a) GSM8K | W→R 0, R→W 1, f 0.000, b 0.003, κ ∞ | `results/tables/table1_transitions.csv` |
| Results (a) HotpotQA | W→R 6, R→W 21, f 0.036, b 0.089, κ 3.50 | `results/tables/table1_transitions.csv` |
| Results (a) caption | 400 test questions | `table1_transitions.csv` (N = 400) |
| Results (b) GSM8K AUROC | 0.64 / 0.95 / 0.64 / 0.95 (S_verb, S_sc10, S_verif, S_avg) | `results/tables/table2_auroc.csv`: 0.640, 0.954, 0.641, 0.953 (shown to 2 decimals) |
| Results (b) HotpotQA AUROC | 0.62 / 0.69 / 0.58 / 0.71 | `table2_auroc.csv`: 0.623, 0.693, 0.577, 0.708 |
| Results (b) | chance 0.5 | definition of AUROC for an uninformative score |
| Results (c) Never | 92.5 / 58.8 | `results/tables/table3_compact.csv` |
| Results (c) Always | 92.3 (p 1.00) / 55.0 (p 0.006) | `table3_compact.csv` |
| Results (c) Majority vote | 95.5 (p 0.004) / 60.0 (p 0.383) | `table3_compact.csv` |
| Results (c) IoE | 93.0 (p 0.500) / 57.3 (p 0.070) | `table3_compact.csv` |
| Results (c) Best dev-tuned gate | 92.5 (p 1.00) / 58.8 (p 1.00) | `table3_compact.csv` rows Gate verb / sc10 / verif / avg (all equal) |
| Results (c) Oracle | 92.5 (p 1.00) / 60.3 (p 0.031) | `table3_compact.csv` |
| Results (d) GSM8K | 14 fixes, 3 breaks, κ 0.21, f 0.47 (14 of 30) | majority answer of the 10 samples scored against gold on the test split (`cgsc/generate_table6.py`, operator R_mv) from `results/raw/qwen_gsm8k.jsonl`: 14 of 30 wrong fixed, 3 of 370 right broken |
| Results (d) HotpotQA | 18 fixes, 14 breaks, κ 0.78 | same computation on `results/raw/qwen_hotpotqa.jsonl` |
| Results (d) S_sc gate | 92.5 % (GSM8K), 59.5 % (HotpotQA) | `results/tables/table6_mv_revision.csv` (Test Acc, Gate S_sc10) |
| Results (d) key numbers | 59.56 % predicted, 59.50 % observed | `table6_mv_revision.csv` (Pred Acc, Obs Acc, HotpotQA, Gate S_sc10) |
| Results (e) H2 | AUROC S_sc 0.95 / 0.69 | `table2_auroc.csv` (S_sc10: 0.954, 0.693) |
| Results (e) H3 | ceiling 0 / 1.5 points | (1−a)f from `results/summary.json` (GSM8K f = 0; HotpotQA a = 0.5875, f = 0.03636 → 1.5 points) |
| Hypothesis / Idea | S(A0) < τ, Acc = a + (1−a) f t − a b u, ceiling (1−a) f | analytical model of gated revision (no measured numbers) |
| Results (f) Unexpected | self-critique changed only 2 of 500 GSM8K answers | `results/raw/qwen_gsm8k.jsonl`: numerically normalised `a0_extracted` vs `a1_extracted` |
| Results (f) Unexpected | P(True) near 0 even for correct GSM8K answers | `results/figures/figure2_histograms.pdf` (S_verif scores of correct GSM8K answers) |
| Results (g) Limitations | 400 test questions, dev split of 100 | `cgsc/data/splits.json` |
| Results (e) H2 | S_verb, S_verif ≤ 0.64 | `table2_auroc.csv` (max 0.641) |
| Methodology | vLLM [10] | inference engine used for all runs (`results/environment.json`: vllm 0.6.6.post1) |
