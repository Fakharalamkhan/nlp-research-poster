"""CLI entry point for running CGSC pipeline."""

import argparse
import os
import sys
import time
import json
import re
import math
import zipfile
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from cgsc.scoring import (
    extract_final_answer,
    score_gsm8k,
    score_hotpotqa,
    majority_vote,
    exact_match_score,
)
from cgsc.sanity import run_sanity_checks


def clean_prompt(text: str) -> str:
    lines = text.splitlines()
    filtered = [l for l in lines if not l.strip().startswith("#")]
    return "\n".join(filtered).strip("\n")


def load_all_prompts(prompts_dir: str):
    prompts = {}
    files = {
        "a0_gsm8k": "a0_gsm8k.txt",
        "a0_hotpotqa": "a0_hotpotqa.txt",
        "verb": "verb.txt",
        "verif_gsm8k": "verif_gsm8k.txt",
        "verif_hotpotqa": "verif_hotpotqa.txt",
        "rev": "revision.txt",
        "ioe": "ioe.txt",
    }
    for k, fname in files.items():
        p = os.path.join(prompts_dir, fname)
        with open(p, "r", encoding="utf-8") as f:
            prompts[k] = clean_prompt(f.read())
    return prompts


def get_dataset_records(dataset_name: str):
    local_path = os.path.join("cgsc/data", f"{dataset_name}.jsonl")
    if os.path.exists(local_path):
        print(f"Loading {dataset_name} records from local file {local_path}...")
        records = {}
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    records[item["id"]] = {
                        "id": item["id"],
                        "question": item["question"].strip(),
                        "gold": str(item["gold"]).strip(),
                        "context": item.get("context", ""),
                    }
        print(f"Loaded {len(records)} records from {local_path}.")
        return records

    # Fallback to Hugging Face datasets if local file is missing
    print(f"Local {local_path} not found; falling back to Hugging Face datasets...")
    from datasets import load_dataset
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        records = {}
        for idx, item in enumerate(ds):
            gold = item["answer"].split("####")[-1].strip()
            records[f"gsm8k-test-{idx}"] = {
                "id": f"gsm8k-test-{idx}",
                "question": item["question"].strip(),
                "gold": gold,
                "context": "",
            }
        return records
    elif dataset_name == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        records = {}
        for idx, item in enumerate(ds):
            ctx_titles = item["context"]["title"]
            ctx_sents = item["context"]["sentences"]
            paras = []
            for t, s_list in zip(ctx_titles, ctx_sents):
                paras.append(f"{t}: {' '.join(s_list)}")
            context_str = "\n\n".join(paras)
            qid = str(item["id"]).strip()
            records[qid] = {
                "id": qid,
                "question": item["question"].strip(),
                "gold": item["answer"].strip(),
                "context": context_str,
            }
        return records
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def load_model(model_key: str):
    if model_key == "qwen":
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    elif model_key == "llama":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise ValueError(f"Unknown model key {model_key}")

    print(f"\nLoading tokenizer for {model_id}...")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    import torch

    compute_cap = (
        torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 8
    )
    model_dtype = "bfloat16" if compute_cap >= 8 else "float16"
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(
        f"GPUs: {num_gpus}, compute_cap: {compute_cap}, chosen dtype: {model_dtype}"
    )

    print(f"Loading vLLM model: {model_id}...")
    sys.stdout.flush()
    llm = LLM(
        model=model_id,
        tensor_parallel_size=num_gpus,
        max_model_len=8192,
        enable_prefix_caching=True,
        dtype=model_dtype,
        gpu_memory_utilization=0.88,
        trust_remote_code=True,
    )
    print("Model loaded successfully into vLLM!\n")
    sys.stdout.flush()
    return llm, tokenizer


def run_batch_pipeline(
    records_to_run, model_key, dataset_key, llm, tokenizer, prompts
):
    results = []
    start_total_time = time.time()
    n_items = len(records_to_run)
    print(
        f"Starting batch pipeline for {n_items} records ({model_key}/{dataset_key})..."
    )
    sys.stdout.flush()

    sp_greedy_768 = SamplingParams(temperature=0.0, max_tokens=768)
    sp_verb = SamplingParams(temperature=0.0, max_tokens=8)
    sp_verif = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    # 1. Generate A0
    print("Generating A0...")
    sys.stdout.flush()
    a0_prompts = []
    for r in records_to_run:
        if dataset_key == "gsm8k":
            user_msg = prompts["a0_gsm8k"].format(question=r["question"])
        else:
            user_msg = prompts["a0_hotpotqa"].format(
                context=r["context"], question=r["question"]
            )
        conv = [{"role": "user", "content": user_msg}]
        text_prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        a0_prompts.append((r, user_msg, text_prompt))

    a0_outputs = llm.generate([p[2] for p in a0_prompts], sp_greedy_768)

    verb_prompts = []
    sc_prompts = []
    verif_prompts = []
    rev_prompts = []
    ioe_prompts = []

    for (r, user_msg, text_prompt), out in zip(a0_prompts, a0_outputs):
        a0_text = out.outputs[0].text
        a0_extracted = extract_final_answer(a0_text)

        conv_verb = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": a0_text},
            {"role": "user", "content": prompts["verb"]},
        ]
        verb_prompts.append(
            tokenizer.apply_chat_template(
                conv_verb, tokenize=False, add_generation_prompt=True
            )
        )
        sc_prompts.append(text_prompt)

        if dataset_key == "gsm8k":
            verif_user_msg = prompts["verif_gsm8k"].format(
                question=r["question"], extracted_A0_answer=a0_extracted
            )
        else:
            verif_user_msg = prompts["verif_hotpotqa"].format(
                question=r["question"],
                context=r["context"],
                extracted_A0_answer=a0_extracted,
            )
        conv_verif = [{"role": "user", "content": verif_user_msg}]
        verif_base = tokenizer.apply_chat_template(
            conv_verif, tokenize=False, add_generation_prompt=True
        )
        verif_prompt_full = verif_base + "The proposed answer is: ("
        verif_prompts.append((verif_user_msg, a0_extracted, verif_prompt_full))

        conv_rev = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": a0_text},
            {"role": "user", "content": prompts["rev"]},
        ]
        rev_prompts.append(
            tokenizer.apply_chat_template(
                conv_rev, tokenize=False, add_generation_prompt=True
            )
        )

        conv_ioe = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": a0_text},
            {"role": "user", "content": prompts["ioe"]},
        ]
        ioe_prompts.append(
            tokenizer.apply_chat_template(
                conv_ioe, tokenize=False, add_generation_prompt=True
            )
        )

    print("Generating S_verb...")
    sys.stdout.flush()
    verb_outputs = llm.generate(verb_prompts, sp_verb)

    # Generate the 10 self-consistency samples as 10 separate requests with seeds 42, 43, ..., 51
    print("Generating S_sc (10 separate samples with seeds 42..51)...")
    sys.stdout.flush()
    sc_sample_outputs = []  # shape: (10, n_items)
    for j in range(10):
        sp_j = SamplingParams(
            n=1,
            temperature=0.7,
            top_p=0.95,
            seed=42 + j,
            max_tokens=768,
        )
        out_j = llm.generate(sc_prompts, sp_j)
        sc_sample_outputs.append(out_j)

    print("Generating S_verif (P(True))...")
    sys.stdout.flush()
    verif_outputs = llm.generate([vp[2] for vp in verif_prompts], sp_verif)

    print("Generating A1 (revision)...")
    sys.stdout.flush()
    rev_outputs = llm.generate(rev_prompts, sp_greedy_768)

    print("Generating IoE baseline...")
    sys.stdout.flush()
    ioe_outputs = llm.generate(ioe_prompts, sp_greedy_768)

    for idx in range(n_items):
        r = a0_prompts[idx][0]
        a0_out = a0_outputs[idx]
        a0_text = a0_out.outputs[0].text
        a0_extracted = extract_final_answer(a0_text)

        v_out = verb_outputs[idx]
        v_text = v_out.outputs[0].text
        m_int = re.search(r"\b(\d+)\b", v_text)
        if m_int:
            val = int(m_int.group(1))
            val = max(0, min(100, val))
            s_verb = val / 100.0
            verb_parse_fail = False
        else:
            s_verb = 1.0
            verb_parse_fail = True

        sc_raw_texts = [
            sc_sample_outputs[j][idx].outputs[0].text for j in range(10)
        ]
        sc_samples_extracted = [
            extract_final_answer(t) for t in sc_raw_texts
        ]
        if dataset_key == "gsm8k":
            matches = [
                score_gsm8k(ans, a0_extracted)[0] for ans in sc_samples_extracted
            ]
        else:
            matches = [
                exact_match_score(ans, a0_extracted)
                for ans in sc_samples_extracted
            ]
        s_sc10 = sum(matches) / 10.0
        s_sc5 = sum(matches[:5]) / 5.0

        vf_out = verif_outputs[idx]
        vf_user_msg, vf_proposed, _ = verif_prompts[idx]
        p_A = 0.0
        p_B = 0.0
        if vf_out.outputs[0].logprobs:
            top_dict = vf_out.outputs[0].logprobs[0]
            for tok_id, logprob_obj in top_dict.items():
                tok_str = logprob_obj.decoded_token.strip().upper()
                prob = math.exp(logprob_obj.logprob)
                if tok_str == "A":
                    p_A += prob
                elif tok_str == "B":
                    p_B += prob
        if (p_A + p_B) > 0:
            s_verif = p_A / (p_A + p_B)
            verif_parse_fail = False
        else:
            s_verif = 0.5
            verif_parse_fail = True

        a1_out = rev_outputs[idx]
        a1_text = a1_out.outputs[0].text
        a1_extracted = extract_final_answer(a1_text)

        ioe_out = ioe_outputs[idx]
        ioe_text = ioe_out.outputs[0].text
        ioe_extracted = extract_final_answer(ioe_text)

        maj_extracted = majority_vote([a0_extracted] + sc_samples_extracted)

        gold = r["gold"]
        if dataset_key == "gsm8k":
            y0, lenient_y0, f1_y0 = score_gsm8k(a0_extracted, gold)
            y1, lenient_y1, f1_y1 = score_gsm8k(a1_extracted, gold)
            y_ioe, lenient_ioe, f1_ioe = score_gsm8k(ioe_extracted, gold)
            y_major, lenient_major, f1_major = score_gsm8k(maj_extracted, gold)
        else:
            y0, lenient_y0, f1_y0 = score_hotpotqa(a0_extracted, gold)
            y1, lenient_y1, f1_y1 = score_hotpotqa(a1_extracted, gold)
            y_ioe, lenient_ioe, f1_ioe = score_hotpotqa(ioe_extracted, gold)
            y_major, lenient_major, f1_major = score_hotpotqa(
                maj_extracted, gold
            )

        sc_prompt_len = len(sc_sample_outputs[0][idx].prompt_token_ids)
        sc_comp_len = sum(
            len(sc_sample_outputs[j][idx].outputs[0].token_ids) for j in range(10)
        )

        cost_dict = {
            "a0_prompt_tokens": len(a0_out.prompt_token_ids),
            "a0_completion_tokens": len(a0_out.outputs[0].token_ids),
            "verb_prompt_tokens": len(v_out.prompt_token_ids),
            "verb_completion_tokens": len(v_out.outputs[0].token_ids),
            "sc_prompt_tokens": 10 * sc_prompt_len,
            "sc_completion_tokens": sc_comp_len,
            "verif_prompt_tokens": len(vf_out.prompt_token_ids),
            "verif_completion_tokens": len(vf_out.outputs[0].token_ids),
            "rev_prompt_tokens": len(a1_out.prompt_token_ids),
            "rev_completion_tokens": len(a1_out.outputs[0].token_ids),
            "ioe_prompt_tokens": len(ioe_out.prompt_token_ids),
            "ioe_completion_tokens": len(ioe_out.outputs[0].token_ids),
        }

        rec_res = {
            "id": r["id"],
            "model": model_key,
            "dataset": dataset_key,
            "question": r["question"],
            "context": r["context"],
            "gold": gold,
            "a0_text": a0_text,
            "a0_extracted": a0_extracted,
            "s_verb": s_verb,
            "verb_text": v_text,
            "verb_parse_fail": verb_parse_fail,
            "s_sc10": s_sc10,
            "s_sc5": s_sc5,
            "sc_raw_texts": sc_raw_texts,
            "sc_extracted_answers": sc_samples_extracted,
            "s_verif": s_verif,
            "verif_prompt": vf_user_msg,
            "verif_proposed_answer": vf_proposed,
            "verif_parse_fail": verif_parse_fail,
            "a1_text": a1_text,
            "a1_extracted": a1_extracted,
            "ioe_text": ioe_text,
            "ioe_extracted": ioe_extracted,
            "majority_extracted": maj_extracted,
            "y0": y0,
            "y1": y1,
            "y_ioe": y_ioe,
            "y_major": y_major,
            "lenient_y0": lenient_y0,
            "lenient_y1": lenient_y1,
            "f1_y0": f1_y0,
            "f1_y1": f1_y1,
            "cost": cost_dict,
            "duration_seconds": (time.time() - start_total_time) / n_items,
        }
        results.append(rec_res)

    return results


def resample_sc_batch(
    records_to_resample, model_key, dataset_key, llm, tokenizer, prompts
):
    """Resample ONLY the 10 SC samples with seeds 42..51, recomputing s_sc5, s_sc10, and majority vote."""
    results = []
    start_total_time = time.time()
    n_items = len(records_to_resample)
    print(
        f"Starting SC resample batch for {n_items} records ({model_key}/{dataset_key})..."
    )
    sys.stdout.flush()

    sc_prompts = []
    for r in records_to_resample:
        if dataset_key == "gsm8k":
            user_msg = prompts["a0_gsm8k"].format(question=r["question"])
        else:
            user_msg = prompts["a0_hotpotqa"].format(
                context=r.get("context", ""), question=r["question"]
            )
        conv = [{"role": "user", "content": user_msg}]
        text_prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        sc_prompts.append(text_prompt)

    print(
        f"Generating S_sc (10 separate samples with seeds 42..51 for {n_items} records)..."
    )
    sys.stdout.flush()
    sc_sample_outputs = []  # shape: (10, n_items)
    for j in range(10):
        sp_j = SamplingParams(
            n=1,
            temperature=0.7,
            top_p=0.95,
            seed=42 + j,
            max_tokens=768,
        )
        out_j = llm.generate(sc_prompts, sp_j)
        sc_sample_outputs.append(out_j)

    for idx, orig_r in enumerate(records_to_resample):
        # Merge by id: preserve all other fields untouched
        new_r = dict(orig_r)
        a0_extracted = new_r["a0_extracted"]
        gold = new_r["gold"]

        sc_raw_texts = [
            sc_sample_outputs[j][idx].outputs[0].text for j in range(10)
        ]
        sc_samples_extracted = [
            extract_final_answer(t) for t in sc_raw_texts
        ]
        if dataset_key == "gsm8k":
            matches = [
                score_gsm8k(ans, a0_extracted)[0] for ans in sc_samples_extracted
            ]
        else:
            matches = [
                exact_match_score(ans, a0_extracted)
                for ans in sc_samples_extracted
            ]
        s_sc10 = sum(matches) / 10.0
        s_sc5 = sum(matches[:5]) / 5.0

        maj_extracted = majority_vote([a0_extracted] + sc_samples_extracted)
        if dataset_key == "gsm8k":
            y_major, lenient_major, f1_major = score_gsm8k(maj_extracted, gold)
        else:
            y_major, lenient_major, f1_major = score_hotpotqa(
                maj_extracted, gold
            )

        sc_prompt_len = len(sc_sample_outputs[0][idx].prompt_token_ids)
        sc_comp_len = sum(
            len(sc_sample_outputs[j][idx].outputs[0].token_ids)
            for j in range(10)
        )

        cost_dict = dict(new_r.get("cost", {}))
        cost_dict["sc_prompt_tokens"] = 10 * sc_prompt_len
        cost_dict["sc_completion_tokens"] = sc_comp_len

        new_r["s_sc10"] = s_sc10
        new_r["s_sc5"] = s_sc5
        new_r["sc_raw_texts"] = sc_raw_texts
        new_r["sc_extracted_answers"] = sc_samples_extracted
        new_r["majority_extracted"] = maj_extracted
        new_r["y_major"] = y_major
        new_r["cost"] = cost_dict
        new_r["duration_seconds"] = (time.time() - start_total_time) / n_items

        results.append(new_r)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run CGSC pipeline for a model/dataset setting"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["qwen", "llama"]
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["gsm8k", "hotpotqa"]
    )
    parser.add_argument("--dry_count", type=int, default=20)
    parser.add_argument("--full_count", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working")
    parser.add_argument(
        "--resample_sc_only",
        action="store_true",
        default=False,
        help="Resample only the 10 SC samples and recompute s_sc5, s_sc10, majority vote.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="results/raw",
        help="Directory containing original raw JSONL files for resampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setting_tag = f"{args.model}_{args.dataset}"
    mode_str = "RESAMPLE SC ONLY" if args.resample_sc_only else "FULL PIPELINE"
    print(f"\n{'='*70}\nRUNNING SETTING: {setting_tag} ({mode_str})\n{'='*70}")
    sys.stdout.flush()

    # Load prompts
    prompts_dir = "cgsc/prompts"
    prompts = load_all_prompts(prompts_dir)

    # Load splits
    splits_path = "cgsc/data/splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Load Model
    llm, tokenizer = load_model(args.model)

    if args.resample_sc_only:
        # Load existing JSONL
        raw_file = os.path.join(args.raw_dir, f"{setting_tag}.jsonl")
        print(f"Loading existing raw records from {raw_file} for resampling...")
        sys.stdout.flush()
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Cannot resample: raw file {raw_file} does not exist!")

        existing_records = []
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_records.append(json.loads(line))
        print(f"Loaded {len(existing_records)} existing records from {raw_file}.")

        # For hotpotqa, ensure context is present (in case compressed raw stripped it)
        if args.dataset == "hotpotqa":
            dataset_records = None
            for r in existing_records:
                if not r.get("context"):
                    if dataset_records is None:
                        dataset_records = get_dataset_records(args.dataset)
                    if r["id"] in dataset_records:
                        r["context"] = dataset_records[r["id"]]["context"]
                if not r.get("verif_prompt"):
                    r["verif_prompt"] = prompts["verif_hotpotqa"].format(
                        question=r["question"],
                        context=r.get("context", ""),
                        extracted_A0_answer=r.get("a0_extracted", ""),
                    )

        # 1. DRY RUN
        dry_records = existing_records[: args.dry_count]
        print(
            f"\n--- Running RESAMPLE DRY RUN ({len(dry_records)} questions) for {setting_tag} ---"
        )
        sys.stdout.flush()
        dry_results = resample_sc_batch(
            dry_records, args.model, args.dataset, llm, tokenizer, prompts
        )

        # 2. SANITY CHECKS
        print(f"\n--- Running Sanity Checks for {setting_tag} ---")
        sys.stdout.flush()
        passed, report = run_sanity_checks(dry_results, args.model, args.dataset)
        print(report)
        sys.stdout.flush()

        # Save sanity report
        report_file = os.path.join(
            args.output_dir, f"sanity_report_{setting_tag}.txt"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Sanity report written to {report_file}")
        sys.stdout.flush()

        if not passed:
            err_msg = f"Sanity checks FAILED for {setting_tag}!"
            print(f"FATAL: {err_msg}")
            sys.stdout.flush()
            sys.exit(1)

        # 3. FULL RUN
        print(f"\n--- Sanity checks PASSED! Starting FULL RESAMPLE for {setting_tag} ---")
        sys.stdout.flush()
        remaining_records = existing_records[args.dry_count : args.full_count]
        if remaining_records:
            remaining_results = resample_sc_batch(
                remaining_records,
                args.model,
                args.dataset,
                llm,
                tokenizer,
                prompts,
            )
            final_results = dry_results + remaining_results
        else:
            final_results = dry_results

    else:
        # FULL PIPELINE RUN
        # Load dataset records
        print(f"Loading dataset records for {args.dataset}...")
        sys.stdout.flush()
        dataset_records = get_dataset_records(args.dataset)
        chosen_ids = splits[args.dataset]["dev"] + splits[args.dataset]["test"]
        ordered_records = [
            dataset_records[i] for i in chosen_ids if i in dataset_records
        ]
        print(f"Total available ordered records: {len(ordered_records)}")
        sys.stdout.flush()

        # 1. DRY RUN
        dry_records = ordered_records[: args.dry_count]
        print(
            f"\n--- Running DRY RUN ({len(dry_records)} questions) for {setting_tag} ---"
        )
        sys.stdout.flush()
        dry_results = run_batch_pipeline(
            dry_records, args.model, args.dataset, llm, tokenizer, prompts
        )

        # 2. SANITY CHECKS
        print(f"\n--- Running Sanity Checks for {setting_tag} ---")
        sys.stdout.flush()
        passed, report = run_sanity_checks(dry_results, args.model, args.dataset)
        print(report)
        sys.stdout.flush()

        # Save sanity report
        report_file = os.path.join(
            args.output_dir, f"sanity_report_{setting_tag}.txt"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Sanity report written to {report_file}")
        sys.stdout.flush()

        if not passed:
            err_msg = f"Sanity checks FAILED for {setting_tag}!"
            print(f"FATAL: {err_msg}")
            sys.stdout.flush()
            sys.exit(1)

        # 3. FULL RUN
        print(f"\n--- Sanity checks PASSED! Starting FULL RUN for {setting_tag} ---")
        sys.stdout.flush()
        remaining_records = ordered_records[args.dry_count : args.full_count]
        if remaining_records:
            remaining_results = run_batch_pipeline(
                remaining_records,
                args.model,
                args.dataset,
                llm,
                tokenizer,
                prompts,
            )
            final_results = dry_results + remaining_results
        else:
            final_results = dry_results

    # 4. Save JSONL
    jsonl_file = os.path.join(args.output_dir, f"{setting_tag}.jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in final_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(final_results)} records to {jsonl_file}")
    sys.stdout.flush()

    # 5. Create ZIP Archive immediately
    zip_file = os.path.join(args.output_dir, f"{setting_tag}_bundle.zip")
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(jsonl_file, arcname=f"{setting_tag}.jsonl")
        zf.write(report_file, arcname=f"sanity_report_{setting_tag}.txt")
    print(f"Created bundle archive at {zip_file}")
    sys.stdout.flush()

    print(f"\nSUCCESS: Pipeline finished cleanly for {setting_tag}!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
