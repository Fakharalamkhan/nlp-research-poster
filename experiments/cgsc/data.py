"""Data loading, formatting, sampling and splitting for GSM8K and HotpotQA."""

import os
import json
import random
import urllib.request
from typing import Dict, List, Any


def format_hotpotqa_context(context_entry: Any) -> str:
    """Format HotpotQA context paragraphs as:
    'Title: sentence sentence ...' per paragraph, separated by newlines.
    Give the model ALL 10 context paragraphs."""
    # HotpotQA context can be a dict {'title': [...], 'sentences': [[...], ...]}
    # or list of [title, [sentence1, sentence2, ...]]
    paragraphs = []
    if isinstance(context_entry, dict):
        titles = context_entry.get("title", [])
        sentences_list = context_entry.get("sentences", [])
        for title, sents in zip(titles, sentences_list):
            text = " ".join(sents)
            paragraphs.append(f"{title}: {text}")
    elif isinstance(context_entry, list):
        for item in context_entry:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sents = item[1]
                text = " ".join(sents) if isinstance(sents, list) else str(sents)
                paragraphs.append(f"{title}: {text}")
    return "\n\n".join(paragraphs)


def load_gsm8k_raw() -> List[Dict[str, Any]]:
    """Load GSM8K test set using HF datasets if available, else direct HTTP download."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        records = []
        for idx, item in enumerate(ds):
            gold = item["answer"].split("####")[-1].strip()
            records.append({
                "id": f"gsm8k-test-{idx}",
                "index": idx,
                "question": item["question"].strip(),
                "gold": gold,
                "full_answer": item["answer"].strip()
            })
        return records
    except Exception as e:
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            lines = [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]
        records = []
        for idx, item in enumerate(lines):
            gold = item["answer"].split("####")[-1].strip()
            records.append({
                "id": f"gsm8k-test-{idx}",
                "index": idx,
                "question": item["question"].strip(),
                "gold": gold,
                "full_answer": item["answer"].strip()
            })
        return records


def load_hotpotqa_raw() -> List[Dict[str, Any]]:
    """Load HotpotQA distractor validation set using HF datasets or parquet."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        records = []
        for idx, item in enumerate(ds):
            context_str = format_hotpotqa_context(item["context"])
            records.append({
                "id": str(item["id"]).strip(),
                "index": idx,
                "question": item["question"].strip(),
                "gold": item["answer"].strip(),
                "context": context_str
            })
        return records
    except Exception as e:
        # Fallback to downloading parquet via pyarrow or datasets-server
        import pyarrow.parquet as pq
        url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        local_pq = "hotpot_val.parquet"
        if not os.path.exists(local_pq):
            with urllib.request.urlopen(req) as resp, open(local_pq, "wb") as f:
                f.write(resp.read())
        table = pq.read_table(local_pq)
        pydict = table.to_pydict()
        records = []
        n = len(pydict["id"])
        for i in range(n):
            context_str = format_hotpotqa_context(pydict["context"][i])
            records.append({
                "id": str(pydict["id"][i]).strip(),
                "index": i,
                "question": pydict["question"][i].strip(),
                "gold": pydict["answer"][i].strip(),
                "context": context_str
            })
        return records


def prepare_data(data_dir: str = "cgsc/data", seed: int = 42, n_sample: int = 500, n_dev: int = 100) -> Dict[str, Any]:
    """Sample 500 questions per dataset with seed 42, split 20% dev / 80% test (seed 42),
    and save splits.json along with dataset jsonl files."""
    os.makedirs(data_dir, exist_ok=True)
    splits_file = os.path.join(data_dir, "splits.json")
    
    # 1. GSM8K
    print("Loading GSM8K...")
    gsm8k_all = load_gsm8k_raw()
    print(f"Total GSM8K test questions: {len(gsm8k_all)}")
    rng_sample_gsm = random.Random(seed)
    gsm8k_sampled_indices = rng_sample_gsm.sample(range(len(gsm8k_all)), n_sample)
    
    rng_split_gsm = random.Random(seed)
    gsm8k_dev_indices = rng_split_gsm.sample(gsm8k_sampled_indices, n_dev)
    gsm8k_dev_set = set(gsm8k_dev_indices)
    gsm8k_test_indices = [idx for idx in gsm8k_sampled_indices if idx not in gsm8k_dev_set]
    
    gsm8k_records = [gsm8k_all[idx] for idx in gsm8k_sampled_indices]
    gsm8k_splits = {
        "dev": [gsm8k_all[idx]["id"] for idx in gsm8k_dev_indices],
        "test": [gsm8k_all[idx]["id"] for idx in gsm8k_test_indices]
    }
    
    gsm8k_path = os.path.join(data_dir, "gsm8k.jsonl")
    with open(gsm8k_path, "w", encoding="utf-8") as f:
        for rec in gsm8k_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(gsm8k_records)} GSM8K records to {gsm8k_path}")

    # 2. HotpotQA
    print("Loading HotpotQA...")
    hotpot_all = load_hotpotqa_raw()
    print(f"Total HotpotQA val questions: {len(hotpot_all)}")
    rng_sample_hp = random.Random(seed)
    hotpot_sampled_indices = rng_sample_hp.sample(range(len(hotpot_all)), n_sample)
    
    rng_split_hp = random.Random(seed)
    hotpot_dev_indices = rng_split_hp.sample(hotpot_sampled_indices, n_dev)
    hotpot_dev_set = set(hotpot_dev_indices)
    hotpot_test_indices = [idx for idx in hotpot_sampled_indices if idx not in hotpot_dev_set]
    
    hotpot_records = [hotpot_all[idx] for idx in hotpot_sampled_indices]
    hotpot_splits = {
        "dev": [hotpot_all[idx]["id"] for idx in hotpot_dev_indices],
        "test": [hotpot_all[idx]["id"] for idx in hotpot_test_indices]
    }
    
    hotpot_path = os.path.join(data_dir, "hotpotqa.jsonl")
    with open(hotpot_path, "w", encoding="utf-8") as f:
        for rec in hotpot_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(hotpot_records)} HotpotQA records to {hotpot_path}")

    splits_data = {
        "seed": seed,
        "n_sample": n_sample,
        "n_dev": n_dev,
        "n_test": n_sample - n_dev,
        "gsm8k": gsm8k_splits,
        "hotpotqa": hotpot_splits
    }
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
    print(f"Saved splits to {splits_file}")
    return splits_data


if __name__ == "__main__":
    prepare_data()
