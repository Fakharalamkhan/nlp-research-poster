"""Built-in sanity checks for confidence-gated self-correction dry run."""

import json
from typing import List, Dict, Any, Tuple
import numpy as np


def compute_auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC without external dependencies using trapezoidal rule on ranks."""
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank scores with average for ties
    ranks = np.argsort(np.argsort(y_score)) + 1
    # Mann-Whitney U test statistic for positive class
    sum_ranks_pos = np.sum(ranks[y_true])
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def run_sanity_checks(records: List[Dict[str, Any]], model_name: str, dataset_name: str) -> Tuple[bool, str]:
    """Run all 7 sanity checks specified in Step 5 on dry-run records.
    Returns (all_passed, report_text)."""
    lines = []
    lines.append("=" * 80)
    lines.append(f" SANITY CHECK REPORT: Model={model_name}, Dataset={dataset_name} (N={len(records)})")
    lines.append("=" * 80)

    overall_pass = True

    # Check 1: LEAK check
    # Assert gold answer does NOT appear in S_verif prompt unless in question/context or extracted A0
    leak_fail = 0
    leak_examples = []
    for r in records:
        gold = str(r.get("gold", "")).strip().lower()
        verif_prompt = str(r.get("verif_prompt", "")).lower()
        q_ctx = (str(r.get("question", "")) + " " + str(r.get("context", ""))).lower()
        a0_ans = str(r.get("a0_extracted", "")).lower()

        # Only check if gold is meaningful (at least 2 chars)
        if len(gold) >= 2 and gold in verif_prompt:
            if gold not in q_ctx and gold not in a0_ans:
                leak_fail += 1

    leak_status = "PASS" if leak_fail == 0 else "FAIL"
    if leak_fail > 0:
        overall_pass = False
    lines.append(f"\n[1] LEAK Check: {leak_status}")
    lines.append(f"    Gold leaks detected in S_verif prompt: {leak_fail}/{len(records)}")
    lines.append("    Printing 2 sample S_verif prompts:")
    for i, r in enumerate(records[:2], 1):
        vp = str(r.get("verif_prompt", ""))
        # Truncate context if long
        if len(vp) > 400:
            vp_disp = vp[:300] + "\n... [truncated] ...\n" + vp[-100:]
        else:
            vp_disp = vp
        lines.append(f"    --- Example {i} ---")
        for pline in vp_disp.splitlines():
            lines.append(f"    | {pline}")

    # Check 2: S_verif computed from A0's extracted answer
    verif_match_fail = 0
    for r in records:
        a0_ext = str(r.get("a0_extracted", "")).strip()
        verif_prop = str(r.get("verif_proposed_answer", "")).strip()
        if a0_ext != verif_prop:
            verif_match_fail += 1
    vmatch_status = "PASS" if verif_match_fail == 0 else "FAIL"
    if verif_match_fail > 0:
        overall_pass = False
    lines.append(f"\n[2] S_verif Alignment with A0: {vmatch_status}")
    lines.append(f"    Mismatches between A0 extracted answer and verif proposed answer: {verif_match_fail}")

    # Check 3: ID Alignment
    id_list = [r["id"] for r in records if "id" in r]
    unique_ids = set(id_list)
    id_status = "PASS" if len(id_list) == len(records) and len(unique_ids) == len(records) else "FAIL"
    if id_status == "FAIL":
        overall_pass = False
    lines.append(f"\n[3] ID Alignment: {id_status}")
    lines.append(f"    Total records: {len(records)}, Unique IDs: {len(unique_ids)}")

    # Check 4: Random baseline AUROC vs Y0 within [0.35, 0.65]
    y0_list = [int(r.get("y0", 0)) for r in records]
    rng = np.random.default_rng(0)
    rand_scores = rng.random(len(records))
    rand_auroc = compute_auroc_binary(np.array(y0_list), rand_scores)
    rand_pass = 0.35 <= rand_auroc <= 0.65
    rand_status = "PASS" if rand_pass else "FAIL"
    if not rand_pass:
        overall_pass = False
    lines.append(f"\n[4] Random Baseline AUROC: {rand_status}")
    lines.append(f"    Random AUROC vs Y0: {rand_auroc:.4f} (Expected in [0.35, 0.65])")

    # Check 5: Print 3 examples: A0 answer, 10 sample answers, S_sc10
    lines.append(f"\n[5] Self-Consistency Agreement Examples (3 samples):")
    for i, r in enumerate(records[:3], 1):
        a0_ans = r.get("a0_extracted", "")
        sc_ans = r.get("sc_extracted_answers", [])
        sc10 = r.get("s_sc10", 0.0)
        lines.append(f"    --- Example {i} (ID: {r.get('id', '')}) ---")
        lines.append(f"    A0 answer: {a0_ans}")
        lines.append(f"    10 sample answers: {sc_ans}")
        lines.append(f"    S_sc10 agreement: {sc10:.2f}")

    # Check 6: Parse-failure counts
    verb_fails = sum(1 for r in records if r.get("verb_parse_fail", False))
    verif_fails = sum(1 for r in records if r.get("verif_parse_fail", False))
    lines.append(f"\n[6] Parse Failures:")
    lines.append(f"    S_verb parse failures: {verb_fails}/{len(records)}")
    lines.append(f"    S_verif parse failures: {verif_fails}/{len(records)}")

    # Check 7: Timing and projected GPU-hours
    time_per_q = [r.get("duration_seconds", 0.0) for r in records if "duration_seconds" in r]
    avg_sec = np.mean(time_per_q) if time_per_q else 0.0
    # Full run is 500 questions per setting
    projected_gpu_hours = (avg_sec * 500.0) / 3600.0
    lines.append(f"\n[7] Timing & Projected Compute:")
    lines.append(f"    Mean latency per question: {avg_sec:.2f} seconds")
    lines.append(f"    Projected full-run time (500 questions): {projected_gpu_hours:.2f} GPU-hours")
    time_pass = projected_gpu_hours <= 10.0
    time_status = "PASS" if time_pass else "WARN (exceeds 10 hours)"
    lines.append(f"    Status: {time_status}")

    # Check 8: Sample Diversity Check (S_sc)
    all_raw_identical_count = 0
    extracted_distinct_counts = []
    for r in records:
        raw_texts = r.get("sc_raw_texts")
        if not raw_texts:
            raw_texts = r.get("sc_extracted_answers", [])
        if len(set(raw_texts)) <= 1:
            all_raw_identical_count += 1
        
        ext_answers = r.get("sc_extracted_answers", [])
        extracted_distinct_counts.append(len(set(ext_answers)))

    frac_identical = all_raw_identical_count / max(1, len(records))
    sc_diversity_pass = frac_identical <= 0.50
    if not sc_diversity_pass:
        overall_pass = False

    from collections import Counter
    dist_map = Counter(extracted_distinct_counts)
    dist_str = ", ".join(f"{k} distinct: {dist_map[k]}" for k in sorted(dist_map.keys()))

    lines.append(f"\n[8] Sample Diversity Check (S_sc): {'PASS' if sc_diversity_pass else 'FAIL'}")
    lines.append(f"    Questions with all 10 raw sample texts identical: {all_raw_identical_count}/{len(records)} ({frac_identical*100:.1f}%, threshold <= 50.0%)")
    lines.append(f"    Distribution of distinct extracted answers per question: [{dist_str}]")

    lines.append("\n" + "=" * 80)
    lines.append(f" OVERALL VERDICT: {'ALL PASS' if overall_pass else 'FAIL'}")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    return overall_pass, report_text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f if l.strip()]
        passed, report = run_sanity_checks(data, "model", "dataset")
        print(report)
        sys.exit(0 if passed else 1)
