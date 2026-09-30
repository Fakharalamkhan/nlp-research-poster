"""Analysis and paper artifacts generator for confidence-gated self-correction."""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from cgsc.sanity import compute_auroc_binary
from cgsc.scoring import score_gsm8k, score_hotpotqa, exact_match_score, majority_vote

# Color-blind safe palette
CB_COLORS = {
    "Never": "#000000",
    "Always": "#7f7f7f",
    "Majority vote": "#9467bd",
    "IoE": "#8c564b",
    "Gate verb": "#1f77b4",
    "Gate sc10": "#ff7f0e",
    "Gate sc5": "#e377c2",
    "Gate verif": "#2ca02c",
    "Gate avg": "#d62728",
    "Oracle": "#17becf"
}

# Figures are drawn at the size they are printed in the paper, so font sizes are true point sizes (>= 9 pt)
FIG_WIDTH_IN = {"figure1": 6.5, "appendix": 5.65}  # \linewidth and 0.9\linewidth (5.85 in) of the 6.5 in text block, minus bbox slack
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

DATASET_DISPLAY = {
    "gsm8k": "GSM8K",
    "hotpotqa": "HotpotQA"
}


def fmt_pct(x: float, nd: int = 1, sign: bool = False) -> str:
    """Format a percentage with half-up rounding (not Python's round-half-to-even)."""
    d = Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{d:+f}" if sign else f"{d:f}"


def display_dataset(dataset_key: str) -> str:
    """Return the canonical display name of a dataset key for tables."""
    return DATASET_DISPLAY.get(dataset_key.lower(), dataset_key.upper())


def compute_auroc_with_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Compute AUROC and 95% bootstrap CI (skipping single-class resamples)."""
    base_auroc = compute_auroc_binary(y_true, y_score)
    n = len(y_true)
    if n == 0 or np.sum(y_true) == 0 or np.sum(y_true) == n:
        return base_auroc, base_auroc, base_auroc
    
    rng = np.random.default_rng(seed)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_sample = y_true[idx]
        if np.sum(yt_sample) == 0 or np.sum(yt_sample) == n:
            continue
        boot_aurocs.append(compute_auroc_binary(yt_sample, y_score[idx]))
    
    if len(boot_aurocs) < 50:
        return base_auroc, base_auroc, base_auroc
    
    ci_low = float(np.percentile(boot_aurocs, 2.5))
    ci_high = float(np.percentile(boot_aurocs, 97.5))
    return base_auroc, ci_low, ci_high


def mcnemar_exact_p(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Exact two-sided binomial test on discordant pairs. Returns 1.0 if no discordant pairs."""
    discordant_a_right = np.sum((y_a == 1) & (y_b == 0))
    discordant_b_right = np.sum((y_a == 0) & (y_b == 1))
    n = discordant_a_right + discordant_b_right
    if n == 0:
        return 1.0
    res = binomtest(discordant_a_right, n, 0.5, alternative="two-sided")
    return float(res.pvalue)


def format_p_val(p: float) -> str:
    """Format p-value cleanly for tables."""
    if p >= 0.999:
        return "1.00"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def select_threshold(scores: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> Tuple[float, float, int]:
    """Pick optimal tau among unique scores + {0.0, 1.000001}.
    Gate revises iff S < tau. Ties broken in favour of fewer revisions.
    Returns (best_tau, best_acc, best_n_rev)."""
    unique_scores = sorted(list(set(scores.tolist())) + [0.0, 1.000001])
    best_tau = 0.0
    best_acc = -1.0
    fewest_revisions = len(scores) + 1
    
    for tau in unique_scores:
        revised = scores < tau
        y_gated = np.where(revised, y1, y0)
        acc = float(np.mean(y_gated))
        n_rev = int(np.sum(revised))
        
        if acc > best_acc or (abs(acc - best_acc) < 1e-9 and n_rev < fewest_revisions):
            best_acc = acc
            best_tau = tau
            fewest_revisions = n_rev
            
    return float(best_tau), float(best_acc), int(fewest_revisions)


def generate_combined_figure1(completed_runs: List[Dict[str, Any]], output_dirs: List[str]):
    """Figure 1: Accuracy vs Compute Cost (2 panels for 1 model, 2x2 for 2 models)."""
    datasets = ["gsm8k", "hotpotqa"]
    models = sorted(list(set(r["model_key"] for r in completed_runs)))
    
    nrows = len(models)
    ncols = len(datasets)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN["figure1"], 2.7 * nrows), squeeze=False)
    
    marker_map = {
        "Never": ("s", 75),
        "Always": ("D", 75),
        "Majority vote": ("^", 85),
        "IoE": ("v", 75),
        "Gate verb": ("o", 75),
        "Gate sc10": ("P", 85),
        "Gate verif": ("X", 85),
        "Gate avg": ("*", 110),
        "Oracle": ("h", 95)
    }

    legend_labels = {
        "Gate verb": r"Gate $S_{\mathrm{verb}}$",
        "Gate sc10": r"Gate $S_{\mathrm{sc10}}$",
        "Gate verif": r"Gate $S_{\mathrm{verif}}$",
        "Gate avg": r"Gate $S_{\mathrm{avg}}$",
    }

    run_dict = {(r["model_key"], r["dataset_key"]): r for r in completed_runs}

    for row_idx, m_key in enumerate(models):
        for col_idx, d_key in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            run_data = run_dict.get((m_key, d_key))
            if not run_data:
                ax.text(0.5, 0.5, f"Pending {m_key} on {d_key}", ha="center", va="center")
                continue
            
            methods = run_data["methods"]
            x_vals, y_vals, names = [], [], []
            for m_name, m_info in methods.items():
                if "hindsight" in m_name.lower():
                    continue
                mean_tokens = float(np.mean(m_info["prompt_tokens"] + m_info["comp_tokens"]))
                acc_pct = float(np.mean(m_info["y"]) * 100)
                x_vals.append(mean_tokens)
                y_vals.append(acc_pct)
                names.append(m_name)
                
                color = CB_COLORS.get(m_name, "#333333")
                marker, size = marker_map.get(m_name, ("o", 70))
                ax.scatter(mean_tokens, acc_pct, color=color, marker=marker, s=size * 0.55, zorder=5,
                           edgecolors='black', linewidths=0.5, label=legend_labels.get(m_name, m_name))

            never_acc = float(np.mean(methods["Never"]["y"]) * 100)
            oracle_acc = float(np.mean(methods["Oracle"]["y"]) * 100)
            ax.axhline(never_acc, color=CB_COLORS["Never"], linestyle="--", alpha=0.45)
            ax.axhline(oracle_acc, color=CB_COLORS["Oracle"], linestyle=":", alpha=0.45)

            ax.set_xlabel("Mean Total Tokens per Question")
            ax.set_ylabel("Test Accuracy (%)")
            ax.set_title(f"{run_data['model_alias']} on {display_dataset(d_key)}", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)

    # One shared legend with distinct markers instead of point labels (avoids overlapping text)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.13, 1, 1))
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "figure1_accuracy_cost.pdf"), bbox_inches="tight")
        if os.path.basename(out_dir) != "images":  # the paper only includes the PDFs
            fig.savefig(os.path.join(out_dir, "figure1_accuracy_cost.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_combined_figure2(completed_runs: List[Dict[str, Any]], output_dirs: List[str]):
    """Figure 2: Confidence score distributions separated by Y0 (2 panels for 1 model, 2x2 for 2 models)."""
    datasets = ["gsm8k", "hotpotqa"]
    models = sorted(list(set(r["model_key"] for r in completed_runs)))
    
    nrows = len(models)
    ncols = len(datasets)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN["appendix"], 2.9 * nrows), squeeze=False)
    run_dict = {(r["model_key"], r["dataset_key"]): r for r in completed_runs}

    for row_idx, m_key in enumerate(models):
        for col_idx, d_key in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            run_data = run_dict.get((m_key, d_key))
            if not run_data:
                ax.text(0.5, 0.5, f"Pending {m_key} on {d_key}", ha="center", va="center")
                continue

            test_recs = run_data["test_recs"]
            signals = run_data["signals"]
            tau_stars = run_data["tau_stars"]
            aurocs = run_data["aurocs"]
            y0 = np.array([int(r["y0"]) for r in test_recs])

            sig_names = ["verb", "sc10", "verif", "avg"]
            sig_labels = [r"$S_{\mathrm{verb}}$", r"$S_{\mathrm{sc10}}$", r"$S_{\mathrm{verif}}$", r"$S_{\mathrm{avg}}$"]

            positions = np.arange(len(sig_names))
            width = 0.35

            correct_data = [signals[s][y0 == 1] for s in sig_names]
            wrong_data = [signals[s][y0 == 0] for s in sig_names]

            bp_c = ax.boxplot(correct_data, positions=positions - width/2, widths=width*0.8,
                              patch_artist=True, boxprops=dict(facecolor="#2ca02c", alpha=0.6),
                              medianprops=dict(color="black", linewidth=1.5), showfliers=False)
            bp_w = ax.boxplot(wrong_data, positions=positions + width/2, widths=width*0.8,
                              patch_artist=True, boxprops=dict(facecolor="#d62728", alpha=0.6),
                              medianprops=dict(color="black", linewidth=1.5), showfliers=False)

            # Plot tau* as diamond markers
            tau_points = [tau_stars.get(s, 0.0) for s in sig_names]
            ax.scatter(positions, tau_points, color="black", marker="D", s=20, zorder=6, label=r"Dev threshold $\tau^*$")

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{l}\n({aurocs.get(s, 0.5):.3f})" for l, s in zip(sig_labels, sig_names)])
            ax.set_xlabel("Signal (AUROC)")
            ax.set_ylabel("Confidence Score")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{run_data['model_alias']} on {display_dataset(d_key)}", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)

            if row_idx == 0 and col_idx == 0:
                legend_items = ([bp_c["boxes"][0], bp_w["boxes"][0], ax.collections[-1]],
                                [f"Correct ($Y_0=1$)", f"Incorrect ($Y_0=0$)", r"Dev threshold $\tau^*$"])

    # Shared legend below the panels so it does not cover the boxes
    fig.legend(*legend_items, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "figure2_histograms.pdf"), bbox_inches="tight")
        if os.path.basename(out_dir) != "images":  # the paper only includes the PDFs
            fig.savefig(os.path.join(out_dir, "figure2_histograms.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_combined_figure3(completed_runs: List[Dict[str, Any]], output_dirs: List[str]):
    """Figure 3: Threshold sensitivity curves (2 panels for 1 model, 2x2 for 2 models)."""
    datasets = ["gsm8k", "hotpotqa"]
    models = sorted(list(set(r["model_key"] for r in completed_runs)))
    
    nrows = len(models)
    ncols = len(datasets)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN["appendix"], 2.9 * nrows), squeeze=False)
    run_dict = {(r["model_key"], r["dataset_key"]): r for r in completed_runs}

    sig_configs = [
        ("verb", r"$S_{\mathrm{verb}}$", CB_COLORS["Gate verb"]),
        ("sc10", r"$S_{\mathrm{sc10}}$", CB_COLORS["Gate sc10"]),
        ("verif", r"$S_{\mathrm{verif}}$", CB_COLORS["Gate verif"]),
        ("avg", r"$S_{\mathrm{avg}}$", CB_COLORS["Gate avg"])
    ]

    for row_idx, m_key in enumerate(models):
        for col_idx, d_key in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            run_data = run_dict.get((m_key, d_key))
            if not run_data:
                ax.text(0.5, 0.5, f"Pending {m_key} on {d_key}", ha="center", va="center")
                continue

            test_recs = run_data["test_recs"]
            signals = run_data["signals"]
            tau_stars = run_data["tau_stars"]

            y0 = np.array([int(r["y0"]) for r in test_recs])
            y1 = np.array([int(r["y1"]) for r in test_recs])
            never_acc = float(np.mean(y0) * 100)
            always_acc = float(np.mean(y1) * 100)

            taus = np.linspace(0.0, 1.0, 101)

            for sig_name, sig_label, color in sig_configs:
                scores = signals[sig_name]
                accs = []
                for tau in taus:
                    rev = scores < tau
                    yg = np.where(rev, y1, y0)
                    accs.append(float(np.mean(yg) * 100))

                ax.plot(taus, accs, label=sig_label, color=color, linewidth=1.2)

                tau_s = tau_stars.get(sig_name, 0.0)
                acc_s = float(np.mean(np.where(scores < tau_s, y1, y0)) * 100)
                ax.scatter([tau_s], [acc_s], color=color, s=28, zorder=5, edgecolors='black', linewidths=0.5)

            ax.axhline(never_acc, color=CB_COLORS["Never"], linestyle="--", linewidth=1.2, label=f"Never ({fmt_pct(never_acc)}%)")
            ax.axhline(always_acc, color=CB_COLORS["Always"], linestyle=":", linewidth=1.2, label=f"Always ({fmt_pct(always_acc)}%)")

            ax.set_xlabel(r"Revision Threshold $\tau$")
            ax.set_ylabel("Test Accuracy (%)")
            ax.set_title(f"{run_data['model_alias']} on {display_dataset(d_key)}", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="center right", fontsize=9, frameon=True, borderpad=0.3, handlelength=1.5)

    plt.tight_layout()
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "figure3_tau_sensitivity.pdf"), bbox_inches="tight")
        if os.path.basename(out_dir) != "images":  # the paper only includes the PDFs
            fig.savefig(os.path.join(out_dir, "figure3_tau_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_all(raw_data_dir: str = "results/raw",
                results_dir: str = "results",
                paper_dir: str = "paper") -> Dict[str, Any]:
    """Run all analysis on completed test runs, generating all tables, figures, CSVs, and summary.json."""
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(paper_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(paper_dir, "images"), exist_ok=True)

    splits_path = "cgsc/data/splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(sf := f)

    settings = [
        ("qwen", "gsm8k", "Qwen2.5-7B-Instruct", "\\modelA{}"),
        ("qwen", "hotpotqa", "Qwen2.5-7B-Instruct", "\\modelA{}"),
        ("llama", "gsm8k", "Llama-3.1-8B-Instruct", "\\modelB{}"),
        ("llama", "hotpotqa", "Llama-3.1-8B-Instruct", "\\modelB{}"),
    ]

    summary = {}
    table1_rows = []
    table2_rows = []
    table5_rows = []
    t4_rows = []
    rw_cases_all = []

    compact_dict: Dict[str, Dict[str, Any]] = {}
    methods_order = [
        "Never",
        "Always",
        "Majority vote",
        "IoE",
        "Gate verb",
        "Gate verb (hindsight)*",
        "Gate sc10",
        "Gate sc10 (hindsight)*",
        "Gate verif",
        "Gate verif (hindsight)*",
        "Gate avg",
        "Gate avg (hindsight)*",
        "Oracle"
    ]
    for m in methods_order:
        compact_dict[m] = {}

    completed_runs = []

    for model_key, dataset_key, model_alias, latex_macro in settings:
        run_key = f"{model_key}_{dataset_key}"
        file_path = os.path.join(raw_data_dir, f"{run_key}.jsonl")
        if not os.path.exists(file_path):
            print(f"Notice: Results file {file_path} not found. Skipping setting {run_key}.")
            continue

        records_map = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    records_map[item["id"]] = item

        dev_ids = splits[dataset_key]["dev"]
        test_ids = splits[dataset_key]["test"]

        dev_recs = [records_map[i] for i in dev_ids if i in records_map]
        test_recs = [records_map[i] for i in test_ids if i in records_map]

        n_test = len(test_recs)
        if n_test == 0:
            continue

        # Test arrays
        y0_test = np.array([int(r["y0"]) for r in test_recs])
        y1_test = np.array([int(r["y1"]) for r in test_recs])
        y_ioe_test = np.array([int(r["y_ioe"]) for r in test_recs])
        y_maj_test = np.array([int(r["y_major"]) for r in test_recs])

        # Dev arrays
        y0_dev = np.array([int(r["y0"]) for r in dev_recs])
        y1_dev = np.array([int(r["y1"]) for r in dev_recs])

        # Explicitly compute sc5 and sc10 from extracted samples to guarantee exact formulation
        def compute_sc_scores(recs_list):
            sc10_arr, sc5_arr = [], []
            for r in recs_list:
                a0 = r["a0_extracted"]
                sc_list = r.get("sc_extracted_answers", [])
                if dataset_key == "gsm8k":
                    matches = [score_gsm8k(ans, a0)[0] for ans in sc_list]
                else:
                    matches = [exact_match_score(ans, a0) for ans in sc_list]
                sc10_arr.append(sum(matches[:10]) / max(1, len(matches[:10])))
                sc5_arr.append(sum(matches[:5]) / max(1, len(matches[:5])))
            return np.array(sc10_arr, dtype=float), np.array(sc5_arr, dtype=float)

        sc10_test, sc5_test = compute_sc_scores(test_recs)
        sc10_dev, sc5_dev = compute_sc_scores(dev_recs)

        signals = {
            "verb": np.array([float(r["s_verb"]) for r in test_recs]),
            "sc10": sc10_test,
            "sc5": sc5_test,
            "verif": np.array([float(r["s_verif"]) for r in test_recs]),
        }
        signals["avg"] = 0.5 * (signals["verb"] + signals["sc10"])

        dev_signals = {
            "verb": np.array([float(r["s_verb"]) for r in dev_recs]),
            "sc10": sc10_dev,
            "sc5": sc5_dev,
            "verif": np.array([float(r["s_verif"]) for r in dev_recs]),
        }
        dev_signals["avg"] = 0.5 * (dev_signals["verb"] + dev_signals["sc10"])

        # Table 1: Transitions (Always-revise on test)
        wr_count = int(np.sum((y0_test == 0) & (y1_test == 1)))
        rw_count = int(np.sum((y0_test == 1) & (y1_test == 0)))
        rr_count = int(np.sum((y0_test == 1) & (y1_test == 1)))
        ww_count = int(np.sum((y0_test == 0) & (y1_test == 0)))

        acc_a0 = float(np.mean(y0_test))
        acc_a1 = float(np.mean(y1_test))
        delta_always = acc_a1 - acc_a0

        w_total = np.sum(y0_test == 0)
        r_total = np.sum(y0_test == 1)

        f_rate = float(wr_count / w_total) if w_total > 0 else 0.0
        b_rate = float(rw_count / r_total) if r_total > 0 else 0.0
        a_param = acc_a0
        denom = (1.0 - a_param) * f_rate
        kappa = float((a_param * b_rate) / denom) if denom > 0 else float("inf")

        table1_rows.append({
            "Model": latex_macro,
            "Dataset": display_dataset(dataset_key),
            "N": n_test,
            "Acc_A0": fmt_pct(acc_a0*100),
            "W->R": f"{wr_count} ({fmt_pct(wr_count/n_test*100)}%)",
            "R->W": f"{rw_count} ({fmt_pct(rw_count/n_test*100)}%)",
            "R->R": f"{rr_count} ({fmt_pct(rr_count/n_test*100)}%)",
            "W->W": f"{ww_count} ({fmt_pct(ww_count/n_test*100)}%)",
            "Delta": fmt_pct(delta_always*100, sign=True),
            "f": f"{f_rate:.3f}",
            "b": f"{b_rate:.3f}",
            "a": f"{a_param:.3f}",
            "kappa": f"{kappa:.2f}"
        })

        # Table 2: AUROC for predicting Y0=1 on test
        t2_entry = {"Model": latex_macro, "Dataset": display_dataset(dataset_key)}
        aurocs = {}
        for sig_name in ["verb", "sc5", "sc10", "verif", "avg"]:
            auroc, ci_l, ci_h = compute_auroc_with_ci(y0_test == 1, signals[sig_name])
            aurocs[sig_name] = auroc
            t2_entry[sig_name] = f"{auroc:.3f} [{ci_l:.3f}, {ci_h:.3f}]"
        table2_rows.append(t2_entry)

        # Select Dev Thresholds
        tau_stars = {}
        for sig_name in ["verb", "sc10", "sc5", "verif", "avg"]:
            tau_stars[sig_name], _, _ = select_threshold(dev_signals[sig_name], y0_dev, y1_dev)

        # Hindsight Thresholds on Test Set (Oracle-tuned upper bound)
        hindsight_dict = {}
        for sig_name in ["verb", "sc10", "verif", "avg"]:
            t_tau, t_acc, t_nrev = select_threshold(signals[sig_name], y0_test, y1_test)
            hindsight_dict[sig_name] = {
                "tau": t_tau,
                "acc": t_acc,
                "pct_rev": (t_nrev / n_test) * 100.0,
                "n_rev": t_nrev
            }

        # Table 5: Analytical Model vs Observed Gated Accuracy
        for s_name in ["verb", "sc10", "verif", "avg"]:
            tau_star = tau_stars[s_name]
            scores_test = signals[s_name]
            rev_mask = scores_test < tau_star
            tpr_t = float(np.mean(scores_test[y0_test == 0] < tau_star)) if w_total > 0 else 0.0
            fpr_u = float(np.mean(scores_test[y0_test == 1] < tau_star)) if r_total > 0 else 0.0
            pred_acc = a_param + (1.0 - a_param) * f_rate * tpr_t - a_param * b_rate * fpr_u
            obs_acc = float(np.mean(np.where(rev_mask, y1_test, y0_test)))
            error = obs_acc - pred_acc
            table5_rows.append({
                "Model": latex_macro,
                "Dataset": display_dataset(dataset_key),
                "Gate": f"$S_{{\\text{{{s_name}}}}}$",
                "tau*": f"{tau_star:.3f}",
                "TPR (t)": f"{tpr_t:.3f}",
                "FPR (u)": f"{fpr_u:.3f}",
                "Pred Acc (%)": fmt_pct(pred_acc*100, 2),
                "Obs Acc (%)": fmt_pct(obs_acc*100, 2),
                "Diff (%)": fmt_pct(error*100, 2, sign=True)
            })

        # Methods evaluation
        methods = {}
        # 1. Never
        methods["Never"] = {
            "tau": "--",
            "y": y0_test,
            "revised": np.zeros(n_test, dtype=bool),
            "prompt_tokens": np.array([r.get("cost", {}).get("a0_prompt_tokens", 0) for r in test_recs]),
            "comp_tokens": np.array([r.get("cost", {}).get("a0_completion_tokens", 0) for r in test_recs])
        }
        # 2. Always
        methods["Always"] = {
            "tau": "--",
            "y": y1_test,
            "revised": np.ones(n_test, dtype=bool),
            "prompt_tokens": np.array([r.get("cost", {}).get("a0_prompt_tokens", 0) + r.get("cost", {}).get("rev_prompt_tokens", 0) for r in test_recs]),
            "comp_tokens": np.array([r.get("cost", {}).get("a0_completion_tokens", 0) + r.get("cost", {}).get("rev_completion_tokens", 0) for r in test_recs])
        }

        # 3. Majority vote (Rev = share of questions where answer differs from A0)
        if dataset_key == "gsm8k":
            maj_rev_mask = np.array([not score_gsm8k(r["majority_extracted"], r["a0_extracted"])[0] for r in test_recs], dtype=bool)
        else:
            maj_rev_mask = np.array([not exact_match_score(r["majority_extracted"], r["a0_extracted"]) for r in test_recs], dtype=bool)
        methods["Majority vote"] = {
            "tau": "--",
            "y": y_maj_test,
            "revised": maj_rev_mask,
            "prompt_tokens": np.array([r.get("cost", {}).get("a0_prompt_tokens", 0) + r.get("cost", {}).get("sc_prompt_tokens", 0) for r in test_recs]),
            "comp_tokens": np.array([r.get("cost", {}).get("a0_completion_tokens", 0) + r.get("cost", {}).get("sc_completion_tokens", 0) for r in test_recs])
        }

        # 4. IoE (Rev = share of questions where answer differs from A0)
        if dataset_key == "gsm8k":
            ioe_rev_mask = np.array([not score_gsm8k(r["ioe_extracted"], r["a0_extracted"])[0] for r in test_recs], dtype=bool)
        else:
            ioe_rev_mask = np.array([not exact_match_score(r["ioe_extracted"], r["a0_extracted"]) for r in test_recs], dtype=bool)
        methods["IoE"] = {
            "tau": "--",
            "y": y_ioe_test,
            "revised": ioe_rev_mask,
            "prompt_tokens": np.array([r.get("cost", {}).get("a0_prompt_tokens", 0) + r.get("cost", {}).get("ioe_prompt_tokens", 0) for r in test_recs]),
            "comp_tokens": np.array([r.get("cost", {}).get("a0_completion_tokens", 0) + r.get("cost", {}).get("ioe_completion_tokens", 0) for r in test_recs])
        }

        # Gates (Dev-tuned) and Hindsight Gates
        for sig_name in ["verb", "sc10", "verif", "avg"]:
            # Dev-tuned
            tau = tau_stars[sig_name]
            rev = signals[sig_name] < tau
            y_g = np.where(rev, y1_test, y0_test)
            
            p_toks, c_toks = [], []
            for r, is_rev in zip(test_recs, rev):
                c_dict = r.get("cost", {})
                p_tot = c_dict.get("a0_prompt_tokens", 0)
                c_tot = c_dict.get("a0_completion_tokens", 0)
                if sig_name == "verb":
                    p_tot += c_dict.get("verb_prompt_tokens", 0)
                    c_tot += c_dict.get("verb_completion_tokens", 0)
                elif sig_name == "sc10":
                    p_tot += c_dict.get("sc_prompt_tokens", 0)
                    c_tot += c_dict.get("sc_completion_tokens", 0)
                elif sig_name == "verif":
                    p_tot += c_dict.get("verif_prompt_tokens", 0)
                    c_tot += c_dict.get("verif_completion_tokens", 0)
                elif sig_name == "avg":
                    p_tot += c_dict.get("verb_prompt_tokens", 0) + c_dict.get("sc_prompt_tokens", 0)
                    c_tot += c_dict.get("verb_completion_tokens", 0) + c_dict.get("sc_completion_tokens", 0)
                if is_rev:
                    p_tot += c_dict.get("rev_prompt_tokens", 0)
                    c_tot += c_dict.get("rev_completion_tokens", 0)
                p_toks.append(p_tot)
                c_toks.append(c_tot)

            methods[f"Gate {sig_name}"] = {
                "tau": f"{tau:.3f}",
                "y": y_g,
                "revised": rev,
                "prompt_tokens": np.array(p_toks),
                "comp_tokens": np.array(c_toks)
            }

            # Hindsight (Oracle-tuned on test set)
            h_tau = hindsight_dict[sig_name]["tau"]
            h_rev = signals[sig_name] < h_tau
            h_y = np.where(h_rev, y1_test, y0_test)
            methods[f"Gate {sig_name} (hindsight)*"] = {
                "tau": f"{h_tau:.3f}",
                "y": h_y,
                "revised": h_rev,
                "prompt_tokens": np.array(p_toks),
                "comp_tokens": np.array(c_toks)
            }

        # Oracle gate
        oracle_rev = (y0_test == 0)
        y_oracle = np.where(oracle_rev, y1_test, y0_test)
        p_toks_orc = [r.get("cost", {}).get("a0_prompt_tokens", 0) + (r.get("cost", {}).get("rev_prompt_tokens", 0) if is_rev else 0) for r, is_rev in zip(test_recs, oracle_rev)]
        c_toks_orc = [r.get("cost", {}).get("a0_completion_tokens", 0) + (r.get("cost", {}).get("rev_completion_tokens", 0) if is_rev else 0) for r, is_rev in zip(test_recs, oracle_rev)]
        methods["Oracle"] = {
            "tau": "--",
            "y": y_oracle,
            "revised": oracle_rev,
            "prompt_tokens": np.array(p_toks_orc),
            "comp_tokens": np.array(c_toks_orc)
        }

        # Build Full Table 3 rows for this setting
        t3_rows = []
        for m_name in methods_order:
            m_data = methods[m_name]
            acc = float(np.mean(m_data["y"]))
            pct_rev = float(np.mean(m_data["revised"]) * 100)
            wr = int(np.sum((y0_test == 0) & (m_data["y"] == 1)))
            rw = int(np.sum((y0_test == 1) & (m_data["y"] == 0)))
            mp = float(np.mean(m_data["prompt_tokens"]))
            mc = float(np.mean(m_data["comp_tokens"]))
            p_nev = mcnemar_exact_p(m_data["y"], methods["Never"]["y"])
            p_alw = mcnemar_exact_p(m_data["y"], methods["Always"]["y"])

            t3_rows.append({
                "Method": m_name,
                "tau*": m_data["tau"],
                "Acc": fmt_pct(acc*100),
                "% revised": fmt_pct(pct_rev),
                "W->R": wr,
                "R->W": rw,
                "Mean Prompt": f"{mp:.0f}",
                "Mean Comp": f"{mc:.0f}",
                "p vs Never": format_p_val(p_nev) if m_name != "Never" else "--",
                "p vs Always": format_p_val(p_alw) if m_name != "Always" else "--"
            })

            # Record into compact dict
            compact_dict[m_name][f"{dataset_key}_acc"] = fmt_pct(acc*100)
            compact_dict[m_name][f"{dataset_key}_rev"] = fmt_pct(pct_rev)
            compact_dict[m_name][f"{dataset_key}_p_nev"] = format_p_val(p_nev) if m_name != "Never" else "--"

        t3_df = pd.DataFrame(t3_rows)
        t3_csv_path = os.path.join(results_dir, "tables", f"table3_gated_{run_key}.csv")
        t3_df.to_csv(t3_csv_path, index=False)

        # LaTeX version for Appendix D
        t3_tex_path = os.path.join(paper_dir, "tables", f"table3_gated_{run_key}.tex")
        with open(t3_tex_path, "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lccccccccc}\n\\toprule\n")
            f.write("Method & $\\tau^*$ & Acc (\\%) & Rev (\\%) & $W\\to R$ & $R\\to W$ & Prompt Tok & Comp Tok & $p_{\\text{Never}}$ & $p_{\\text{Always}}$ \\\\\n\\midrule\n")
            for r in t3_rows:
                m_raw = r["Method"]
                m_disp = m_raw.replace("Gate ", "Gate $S_{\\text{").replace("sc10", "sc10}}$").replace("verb", "verb}}$").replace("verif", "verif}}$").replace("avg", "avg}}$")
                m_disp = m_disp.replace("(hindsight)*", "(hindsight)$^\\dagger$")
                if "hindsight" in m_raw:
                    pass
                elif m_raw == "Gate verb":
                    f.write("\\midrule\n")
                elif m_raw == "Oracle":
                    f.write("\\midrule\n")
                f.write(f"{m_disp} & {r['tau*']} & {r['Acc']} & {r['% revised']} & {r['W->R']} & {r['R->W']} & {r['Mean Prompt']} & {r['Mean Comp']} & {r['p vs Never']} & {r['p vs Always']} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\multicolumn{10}{l}{\\footnotesize $^\\dagger$Oracle-tuned upper bound on test set (not a practical method).}\\\\\n")
            f.write("\\end{tabular}\n")

        # Save run data for multi-panel combined figures
        completed_runs.append({
            "model_key": model_key,
            "dataset_key": dataset_key,
            "model_alias": model_alias,
            "latex_macro": latex_macro,
            "test_recs": test_recs,
            "dev_recs": dev_recs,
            "methods": methods,
            "signals": signals,
            "dev_signals": dev_signals,
            "tau_stars": tau_stars,
            "aurocs": aurocs
        })

        # Store for summary
        summary[run_key] = {
            "a": a_param, "f": f_rate, "b": b_rate, "kappa": kappa,
            "aurocs": {k: float(compute_auroc_with_ci(y0_test == 1, signals[k])[0]) for k in ["verb", "sc5", "sc10", "verif", "avg"]},
            "accuracies": {m_name: float(np.mean(m_data["y"])) for m_name, m_data in methods.items()},
            "tau_stars": tau_stars,
            "hindsight": hindsight_dict
        }

        # Table 4: Format drift (HotpotQA only)
        if dataset_key == "hotpotqa":
            em_rw = int(np.sum((y0_test == 1) & (y1_test == 0)))
            y0_lenient = np.array([int(r.get("lenient_y0", r["y0"])) for r in test_recs])
            y1_lenient = np.array([int(r.get("lenient_y1", r["y1"])) for r in test_recs])
            lenient_rw = int(np.sum((y0_lenient == 1) & (y1_lenient == 0)))
            t4_rows.append({
                "Model": latex_macro,
                "Always-Revise R->W (EM)": em_rw,
                "Always-Revise R->W (Lenient)": lenient_rw,
                "Format Drift Cases": em_rw - lenient_rw
            })

        # Collect RW cases for rw_cases.csv
        for r in test_recs:
            if r["y0"] == 1 and r["y1"] == 0:
                rw_cases_all.append({
                    "id": r["id"],
                    "model": model_alias,
                    "dataset": dataset_key,
                    "question": r["question"],
                    "gold": r["gold"],
                    "a0_extracted": r.get("a0_extracted", ""),
                    "a1_extracted": r.get("a1_extracted", ""),
                    "a0_full_text": r.get("a0_text", ""),
                    "a1_full_text": r.get("a1_text", ""),
                    "lenient_y1": r.get("lenient_y1", False),
                    "category": ""
                })

    # Save Combined Multi-panel Figures
    fig_out_dirs = [os.path.join(paper_dir, "images"), os.path.join(results_dir, "figures")]
    generate_combined_figure1(completed_runs, fig_out_dirs)
    generate_combined_figure2(completed_runs, fig_out_dirs)
    generate_combined_figure3(completed_runs, fig_out_dirs)

    # Save Table 1
    t1_df = pd.DataFrame(table1_rows)
    t1_df.to_csv(os.path.join(results_dir, "tables", "table1_transitions.csv"), index=False)
    with open(os.path.join(paper_dir, "tables", "table1_transitions.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llcccccccccc}\n\\toprule\n")
        f.write("Model & Dataset & $N$ & Acc($A_0$) & $W\\to R$ & $R\\to W$ & $R\\to R$ & $W\\to W$ & $\\Delta$ & $f$ & $b$ & $\\kappa$ \\\\\n\\midrule\n")
        for r in table1_rows:
            wr_str = r['W->R'].replace("%", "\\%")
            rw_str = r['R->W'].replace("%", "\\%")
            rr_str = r['R->R'].replace("%", "\\%")
            ww_str = r['W->W'].replace("%", "\\%")
            kap_str = "$\\infty$" if str(r['kappa']).lower() in ["inf", "infinity"] else str(r['kappa'])
            f.write(f"{r['Model']} & {r['Dataset']} & {r['N']} & {r['Acc_A0']}\\% & {wr_str} & {rw_str} & {rr_str} & {ww_str} & {r['Delta']}\\% & {r['f']} & {r['b']} & {kap_str} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Save Table 2
    t2_df = pd.DataFrame(table2_rows)
    t2_df.to_csv(os.path.join(results_dir, "tables", "table2_auroc.csv"), index=False)
    with open(os.path.join(paper_dir, "tables", "table2_auroc.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
        f.write("Model & Dataset & $S_{\\text{verb}}$ & $S_{\\text{sc5}}$ & $S_{\\text{sc10}}$ & $S_{\\text{verif}}$ & $S_{\\text{avg}}$ \\\\\n\\midrule\n")
        for r in table2_rows:
            f.write(f"{r['Model']} & {r['Dataset']} & {r['verb']} & {r['sc5']} & {r['sc10']} & {r['verif']} & {r['avg']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Save Compact Table 3
    compact_rows = []
    for m in methods_order:
        c_row = {"Method": m}
        for r_info in completed_runs:
            d_key = r_info["dataset_key"]
            c_row[f"{display_dataset(d_key)} Acc (%)"] = compact_dict[m].get(f"{d_key}_acc", "--")
            c_row[f"{display_dataset(d_key)} Rev (%)"] = compact_dict[m].get(f"{d_key}_rev", "--")
            c_row[f"{display_dataset(d_key)} p vs Never"] = compact_dict[m].get(f"{d_key}_p_nev", "--")
        compact_rows.append(c_row)

    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(os.path.join(results_dir, "tables", "table3_compact.csv"), index=False)

    with open(os.path.join(paper_dir, "tables", "table3_compact.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("& \\multicolumn{3}{c}{\\textbf{GSM8K (\\modelA{})}} & \\multicolumn{3}{c}{\\textbf{HotpotQA (\\modelA{})}} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write("\\textbf{Method} & \\textbf{Acc (\\%)} & \\textbf{Rev (\\%)} & \\textbf{$p_{\\text{Never}}$} & \\textbf{Acc (\\%)} & \\textbf{Rev (\\%)} & \\textbf{$p_{\\text{Never}}$} \\\\\n\\midrule\n")
        for r in compact_rows:
            m_raw = r["Method"]
            m_disp = m_raw.replace("Gate ", "Gate $S_{\\text{").replace("sc10", "sc10}}$").replace("verb", "verb}}$").replace("verif", "verif}}$").replace("avg", "avg}}$")
            m_disp = m_disp.replace("(hindsight)*", "(hindsight)$^\\dagger$")
            if "hindsight" in m_raw:
                pass
            elif m_raw == "Gate verb":
                f.write("\\midrule\n")
            elif m_raw == "Oracle":
                f.write("\\midrule\n")
            gsm_acc = r.get("GSM8K Acc (%)", "--")
            gsm_rev = r.get("GSM8K Rev (%)", "--")
            gsm_p = r.get("GSM8K p vs Never", "--")
            hot_acc = r.get("HotpotQA Acc (%)", "--")
            hot_rev = r.get("HotpotQA Rev (%)", "--")
            hot_p = r.get("HotpotQA p vs Never", "--")
            f.write(f"{m_disp} & {gsm_acc} & {gsm_rev} & {gsm_p} & {hot_acc} & {hot_rev} & {hot_p} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\multicolumn{7}{l}{\\footnotesize $^\\dagger$Oracle-tuned upper bound on test set (not a practical method).}\\\\\n")
        f.write("\\end{tabular}\n")

    # Save Table 4: Format drift
    if t4_rows:
        t4_df = pd.DataFrame(t4_rows)
        t4_df.to_csv(os.path.join(results_dir, "tables", "table4_format_drift.csv"), index=False)
        with open(os.path.join(paper_dir, "tables", "table4_format_drift.tex"), "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lccc}\n\\toprule\n")
            f.write("Model & Always-Revise $R\\to W$ (EM) & Always-Revise $R\\to W$ (Lenient) & Format Drift Cases \\\\\n\\midrule\n")
            for r in t4_rows:
                f.write(f"{r['Model']} & {r['Always-Revise R->W (EM)']} & {r['Always-Revise R->W (Lenient)']} & {r['Format Drift Cases']} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # Save Table 5: Analytical Model vs Observed Accuracy
    if table5_rows:
        t5_df = pd.DataFrame(table5_rows)
        t5_df.to_csv(os.path.join(results_dir, "tables", "table5_analytical_model.csv"), index=False)
        with open(os.path.join(paper_dir, "tables", "table5_analytical_model.tex"), "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{llccccccc}\n\\toprule\n")
            f.write("Model & Dataset & Gate & $\\tau^*$ & TPR ($t$) & FPR ($u$) & Pred Acc (\\%) & Obs Acc (\\%) & Diff (\\%) \\\\\n\\midrule\n")
            for r in table5_rows:
                f.write(f"{r['Model']} & {r['Dataset']} & {r['Gate']} & {r['tau*']} & {r['TPR (t)']} & {r['FPR (u)']} & {r['Pred Acc (%)']} & {r['Obs Acc (%)']} & {r['Diff (%)']} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # Table 6: Majority Vote as a Revision Operator
    table6_csv_rows = []
    table6_tex_rows = []
    for r_info in completed_runs:
        d_key = r_info["dataset_key"]
        m_key = r_info["model_key"]
        l_macro = r_info["latex_macro"]
        t_recs = r_info["test_recs"]
        d_recs = r_info["dev_recs"]
        sigs = r_info["signals"]
        dev_sigs = r_info["dev_signals"]
        
        y0_t = np.array([int(r["y0"]) for r in t_recs])
        y0_d = np.array([int(r["y0"]) for r in d_recs])
        y_maj_base = np.array([int(r["y_major"]) for r in t_recs])
        y_mv_t = np.array([
            score_gsm8k(majority_vote(r["sc_extracted_answers"][:10]), r["gold"])[0]
            if d_key == "gsm8k"
            else score_hotpotqa(majority_vote(r["sc_extracted_answers"][:10]), r["gold"])[0]
            for r in t_recs
        ])
        y_mv_d = np.array([
            score_gsm8k(majority_vote(r["sc_extracted_answers"][:10]), r["gold"])[0]
            if d_key == "gsm8k"
            else score_hotpotqa(majority_vote(r["sc_extracted_answers"][:10]), r["gold"])[0]
            for r in d_recs
        ])

        w_tot_cur = int(np.sum(y0_t == 0))
        r_tot_cur = int(np.sum(y0_t == 1))
        a_cur = float(np.mean(y0_t))
        wr_mv_cur = int(np.sum((y0_t == 0) & (y_mv_t == 1)))
        rw_mv_cur = int(np.sum((y0_t == 1) & (y_mv_t == 0)))
        f_mv_cur = wr_mv_cur / w_tot_cur if w_tot_cur > 0 else 0.0
        b_mv_cur = rw_mv_cur / r_tot_cur if r_tot_cur > 0 else 0.0

        for sig_name in ["sc10", "avg"]:
            t_star, _, _ = select_threshold(dev_sigs[sig_name], y0_d, y_mv_d)
            rev_m = sigs[sig_name] < t_star
            y_gated_cur = np.where(rev_m, y_mv_t, y0_t)
            acc_cur = float(np.mean(y_gated_cur))
            pct_rev_cur = float(np.mean(rev_m) * 100)
            wr_cur = int(np.sum((y0_t == 0) & (y_gated_cur == 1)))
            rw_cur = int(np.sum((y0_t == 1) & (y_gated_cur == 0)))

            p_nev_cur = mcnemar_exact_p(y_gated_cur, y0_t)
            p_mv_ung_cur = mcnemar_exact_p(y_gated_cur, y_maj_base)
            h_tau_cur, h_acc_cur, _ = select_threshold(sigs[sig_name], y0_t, y_mv_t)

            t_tpr_cur = float(np.mean(sigs[sig_name][y0_t == 0] < t_star)) if w_tot_cur > 0 else 0.0
            u_fpr_cur = float(np.mean(sigs[sig_name][y0_t == 1] < t_star)) if r_tot_cur > 0 else 0.0
            pred_acc_cur = a_cur + (1.0 - a_cur) * f_mv_cur * t_tpr_cur - a_cur * b_mv_cur * u_fpr_cur
            diff_cur = acc_cur - pred_acc_cur

            table6_csv_rows.append({
                "Model": l_macro,
                "Dataset": display_dataset(d_key),
                "Gate": f"Gate S_{sig_name}",
                "tau*": f"{t_star:.3f}",
                "Test Acc (%)": fmt_pct(acc_cur*100),
                "% Rev": fmt_pct(pct_rev_cur),
                "W->R": wr_cur,
                "R->W": rw_cur,
                "p vs Never": format_p_val(p_nev_cur),
                "p vs Ungated MV": format_p_val(p_mv_ung_cur),
                "Hindsight tau": f"{h_tau_cur:.3f}",
                "Hindsight Acc (%)": fmt_pct(h_acc_cur*100),
                "Pred Acc (%)": fmt_pct(pred_acc_cur*100, 2),
                "Obs Acc (%)": fmt_pct(acc_cur*100, 2),
                "Diff (%)": fmt_pct(diff_cur*100, 2, sign=True)
            })

            table6_tex_rows.append({
                "Dataset": display_dataset(d_key),
                "Gate": f"Gate $S_{{\\text{{{sig_name}}}}}$",
                "tau*": f"{t_star:.3f}",
                "Test Acc": f"{fmt_pct(acc_cur*100)}\\%",
                "Rev": f"{fmt_pct(pct_rev_cur)}\\%",
                "W->R": wr_cur,
                "R->W": rw_cur,
                "p_Never": format_p_val(p_nev_cur),
                "p_UngatedMV": format_p_val(p_mv_ung_cur),
                "Hindsight tau": f"{h_tau_cur:.3f}",
                "Hindsight Acc": f"{fmt_pct(h_acc_cur*100)}\\%",
                "Pred Acc": f"{fmt_pct(pred_acc_cur*100, 2)}\\%",
                "Obs Acc": f"{fmt_pct(acc_cur*100, 2)}\\%",
                "Diff": f"{fmt_pct(diff_cur*100, 2, sign=True)}\\%"
            })

    if table6_csv_rows:
        t6_df = pd.DataFrame(table6_csv_rows)
        t6_df.to_csv(os.path.join(results_dir, "tables", "table6_mv_revision.csv"), index=False)
        # Compact version used in the paper (Table 5); the full set of columns is in the CSV
        def p_short(p_str: str) -> str:
            p = float(p_str)
            return "1.00" if p >= 0.999 else (f"{p:.2f}" if p >= 0.01 else p_str)

        with open(os.path.join(paper_dir, "tables", "table6_mv_revision.tex"), "w", encoding="utf-8") as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\caption{Majority vote as a gated revision operator $R_{\\text{mv}}$ on GSM8K and HotpotQA (\\modelA{}, test split). $\\tau^*$ tuned on development split. $p_{\\text{Never}}$ is two-sided exact McNemar test vs.\\ never revising. Predicted vs.\\ observed test accuracy from the analytical model.}\n")
            f.write("\\label{tab:mv}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n")
            f.write("\\begin{tabular}{llcccccccc}\n\\toprule\n")
            f.write("Dataset & Gate & $\\tau^*$ & Acc (\\%) & Rev (\\%) & $W\\to R$ & $R\\to W$ & $p_{\\text{Never}}$ & Pred (\\%) & Obs (\\%) \\\\\n\\midrule\n")
            for r in table6_csv_rows:
                gate = "Gate $S_{\\text{sc}}$" if r["Gate"].endswith("sc10") else "Gate $S_{\\text{avg}}$"
                f.write(f"{r['Dataset']} & {gate} & {r['tau*']} & {r['Test Acc (%)']} & {r['% Rev']} & {r['W->R']} & {r['R->W']} & {p_short(r['p vs Never'])} & {r['Pred Acc (%)']} & {r['Obs Acc (%)']} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Save rw_cases.csv
    if rw_cases_all:
        rw_df = pd.DataFrame(rw_cases_all)
        rw_df.to_csv(os.path.join(results_dir, "rw_cases.csv"), index=False)

    # Save summary.json
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Extended analysis complete. All tables, figures, CSVs, and summary.json generated successfully.")
    return summary


if __name__ == "__main__":
    analyze_all()
