"""Generate Table 6: Majority Vote as a Revision Step."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from cgsc.scoring import score_gsm8k, score_hotpotqa, majority_vote, exact_match_score


DATASET_DISPLAY = {
    "gsm8k": "GSM8K",
    "hotpotqa": "HotpotQA"
}


def display_dataset(dataset_key: str) -> str:
    """Return the canonical display name of a dataset key for tables."""
    return DATASET_DISPLAY.get(dataset_key.lower(), dataset_key.upper())


def mcnemar_exact_p(y_a, y_b):
    discordant_a_right = np.sum((y_a == 1) & (y_b == 0))
    discordant_b_right = np.sum((y_a == 0) & (y_b == 1))
    n = discordant_a_right + discordant_b_right
    if n == 0:
        return 1.0
    return float(binomtest(discordant_a_right, n, 0.5, alternative="two-sided").pvalue)


def format_p_val(p):
    if p >= 0.999:
        return "1.00"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def select_threshold(scores, y0, y_rev):
    unique_scores = sorted(list(set(scores.tolist())) + [0.0, 1.000001])
    best_tau = 0.0
    best_acc = -1.0
    fewest_revisions = len(scores) + 1
    for tau in unique_scores:
        revised = scores < tau
        y_gated = np.where(revised, y_rev, y0)
        acc = float(np.mean(y_gated))
        n_rev = int(np.sum(revised))
        if acc > best_acc or (abs(acc - best_acc) < 1e-9 and n_rev < fewest_revisions):
            best_acc = acc
            best_tau = tau
            fewest_revisions = n_rev
    return float(best_tau), float(best_acc), int(fewest_revisions)


def main():
    splits_path = "cgsc/data/splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    csv_rows = []
    tex_rows = []

    for ds in ["gsm8k", "hotpotqa"]:
        raw_path = f"results/raw/qwen_{ds}.jsonl"
        recs = {}
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    recs[item["id"]] = item

        test_recs = [recs[i] for i in splits[ds]["test"] if i in recs]
        dev_recs = [recs[i] for i in splits[ds]["dev"] if i in recs]

        y0_test = np.array([int(r["y0"]) for r in test_recs])
        y0_dev = np.array([int(r["y0"]) for r in dev_recs])

        # R_mv operator: majority answer over the 10 samples
        y_mv_test = np.array(
            [
                score_gsm8k(majority_vote(r["sc_extracted_answers"][:10]), r["gold"])[0]
                if ds == "gsm8k"
                else score_hotpotqa(
                    majority_vote(r["sc_extracted_answers"][:10]), r["gold"]
                )[0]
                for r in test_recs
            ]
        )
        y_mv_dev = np.array(
            [
                score_gsm8k(majority_vote(r["sc_extracted_answers"][:10]), r["gold"])[0]
                if ds == "gsm8k"
                else score_hotpotqa(
                    majority_vote(r["sc_extracted_answers"][:10]), r["gold"]
                )[0]
                for r in dev_recs
            ]
        )

        y_maj_baseline = np.array([int(r["y_major"]) for r in test_recs])

        def get_sig(r_list):
            sc10, verb = [], []
            for r in r_list:
                a0 = r["a0_extracted"]
                sc_list = r.get("sc_extracted_answers", [])
                if ds == "gsm8k":
                    matches = [score_gsm8k(a, a0)[0] for a in sc_list]
                else:
                    matches = [exact_match_score(a, a0) for a in sc_list]
                sc10.append(sum(matches[:10]) / 10.0)
                verb.append(float(r["s_verb"]))
            sc10_arr = np.array(sc10, dtype=float)
            verb_arr = np.array(verb, dtype=float)
            avg_arr = 0.5 * (verb_arr + sc10_arr)
            return {"sc10": sc10_arr, "avg": avg_arr}

        sig_test = get_sig(test_recs)
        sig_dev = get_sig(dev_recs)

        w_tot = int(np.sum(y0_test == 0))
        r_tot = int(np.sum(y0_test == 1))
        a = float(np.mean(y0_test))

        wr_mv = int(np.sum((y0_test == 0) & (y_mv_test == 1)))
        rw_mv = int(np.sum((y0_test == 1) & (y_mv_test == 0)))
        f_mv = wr_mv / w_tot
        b_mv = rw_mv / r_tot
        kappa_mv = (a * b_mv) / ((1 - a) * f_mv)

        for sig_name in ["sc10", "avg"]:
            tau_star, _, _ = select_threshold(sig_dev[sig_name], y0_dev, y_mv_dev)
            rev_mask = sig_test[sig_name] < tau_star
            y_gated = np.where(rev_mask, y_mv_test, y0_test)

            acc = float(np.mean(y_gated))
            pct_rev = float(np.mean(rev_mask) * 100)
            wr = int(np.sum((y0_test == 0) & (y_gated == 1)))
            rw = int(np.sum((y0_test == 1) & (y_gated == 0)))

            p_nev = mcnemar_exact_p(y_gated, y0_test)
            p_mv_ungated = mcnemar_exact_p(y_gated, y_maj_baseline)

            h_tau, h_acc, _ = select_threshold(sig_test[sig_name], y0_test, y_mv_test)

            t_tpr = float(np.mean(sig_test[sig_name][y0_test == 0] < tau_star)) if w_tot > 0 else 0.0
            u_fpr = float(np.mean(sig_test[sig_name][y0_test == 1] < tau_star)) if r_tot > 0 else 0.0
            pred_acc = a + (1.0 - a) * f_mv * t_tpr - a * b_mv * u_fpr
            diff = acc - pred_acc

            gate_label = f"Gate $S_{{\\text{{{sig_name}}}}}$"
            csv_rows.append(
                {
                    "Model": r"\modelA{}",
                    "Dataset": display_dataset(ds),
                    "Gate": f"Gate S_{sig_name}",
                    "tau*": f"{tau_star:.3f}",
                    "Test Acc (%)": f"{acc*100:.1f}",
                    "% Rev": f"{pct_rev:.1f}",
                    "W->R": wr,
                    "R->W": rw,
                    "p vs Never": format_p_val(p_nev),
                    "p vs Ungated MV": format_p_val(p_mv_ungated),
                    "Hindsight tau": f"{h_tau:.3f}",
                    "Hindsight Acc (%)": f"{h_acc*100:.1f}",
                    "Pred Acc (%)": f"{pred_acc*100:.2f}",
                    "Obs Acc (%)": f"{acc*100:.2f}",
                    "Diff (%)": f"{diff*100:+.2f}",
                }
            )

            tex_rows.append(
                {
                    "Dataset": display_dataset(ds),
                    "Gate": gate_label,
                    "tau*": f"{tau_star:.3f}",
                    "Test Acc": f"{acc*100:.1f}\\%",
                    "Rev": f"{pct_rev:.1f}\\%",
                    "W->R": wr,
                    "R->W": rw,
                    "p_Never": format_p_val(p_nev),
                    "p_UngatedMV": format_p_val(p_mv_ungated),
                    "Hindsight tau": f"{h_tau:.3f}",
                    "Hindsight Acc": f"{h_acc*100:.1f}\\%",
                    "Pred Acc": f"{pred_acc*100:.2f}\\%",
                    "Obs Acc": f"{acc*100:.2f}\\%",
                    "Diff": f"{diff*100:+.2f}\\%",
                }
            )

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("paper/tables", exist_ok=True)

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv("results/tables/table6_mv_revision.csv", index=False)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(
        r"\caption{Evaluation of majority vote as a gated revision operator $R_{\text{mv}}$ on GSM8K and HotpotQA (\modelA{}, $N=400$ test questions). Thresholds $\tau^*$ are selected on the development split. $p$-values are computed with two-sided exact McNemar tests against Never revising ($A_0$) and against ungated Majority Vote. Predicted accuracy is given by the analytical model $a + (1-a) f_{\text{mv}} t - a b_{\text{mv}} u$.}"
    )
    lines.append(r"\label{tab:mv}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Gate & $\tau^*$ & Acc (\%) & Rev (\%) & $W\to R$ & $R\to W$ & $p_{\text{Never}}$ & $p_{\text{MV}}$ & $\tau_{\text{hind}}$ & $\text{Acc}_{\text{hind}}$ & Pred (\%) & Obs (\%) & Diff (\%) \\"
    )
    lines.append(r"\midrule")
    for r in tex_rows:
        lines.append(
            f"{r['Dataset']} & {r['Gate']} & {r['tau*']} & {r['Test Acc']} & {r['Rev']} & {r['W->R']} & {r['R->W']} & {r['p_Never']} & {r['p_UngatedMV']} & {r['Hindsight tau']} & {r['Hindsight Acc']} & {r['Pred Acc']} & {r['Obs Acc']} & {r['Diff']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    with open("paper/tables/table6_mv_revision.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Table 6 CSV & LaTeX generated successfully.")


if __name__ == "__main__":
    main()
