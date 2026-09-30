"""Scoring and answer extraction for GSM8K and HotpotQA."""

import re
import string
from collections import Counter
from typing import Tuple, Optional


def extract_final_answer(text: str) -> str:
    """Extract text after the LAST 'Final answer:' (case-insensitive).
    If absent, return the last non-empty line."""
    if not text:
        return ""
    matches = list(re.finditer(r"final\s+answer\s*:\s*", text, flags=re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        extracted = text[last_match.end():].strip()
        # Take the first line or strip subsequent blank lines
        lines = [line.strip() for line in extracted.splitlines() if line.strip()]
        return lines[0] if lines else extracted
    
    # If absent, use the last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalise_gsm8k(text: str) -> Optional[float]:
    """Take the last number in the extracted span; strip '$', ',', '%', spaces.
    Return as float if valid, else None."""
    if not text:
        return None
    cleaned = text.replace(",", "").replace("$", "").replace("%", " ")
    # Find all numeric patterns (integers, floats, negative numbers)
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def score_gsm8k(extracted_answer: str, gold_answer: str, tol: float = 1e-6) -> Tuple[bool, bool, float]:
    """Score GSM8K: returns (em, lenient, f1).
    Float-compare with tolerance 1e-6 (so 12, 12.0, $12 all equal 12).
    For GSM8K, lenient and f1 match em."""
    pred_num = normalise_gsm8k(extracted_answer)
    gold_num = normalise_gsm8k(gold_answer)
    if pred_num is None or gold_num is None:
        return False, False, 0.0
    match = abs(pred_num - gold_num) <= tol
    return match, match, 1.0 if match else 0.0


def normalise_hotpotqa(s: str) -> str:
    """Official SQuAD / HotpotQA answer normalisation:
    Lower-case, remove punctuation, remove articles (a, an, the), and collapse extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """EM on normalised strings."""
    return normalise_hotpotqa(prediction) == normalise_hotpotqa(ground_truth)


def lenient_match_score(prediction: str, ground_truth: str) -> bool:
    """Lenient match: normalised gold answer is contained in normalised extracted answer."""
    norm_gold = normalise_hotpotqa(ground_truth)
    norm_pred = normalise_hotpotqa(prediction)
    if not norm_gold:
        return False
    return norm_gold in norm_pred


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 on normalised strings."""
    norm_pred = normalise_hotpotqa(prediction)
    norm_gold = normalise_hotpotqa(ground_truth)
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_hotpotqa(extracted_answer: str, gold_answer: str) -> Tuple[bool, bool, float]:
    """Score HotpotQA: returns (em, lenient, f1)."""
    em = exact_match_score(extracted_answer, gold_answer)
    lenient = lenient_match_score(extracted_answer, gold_answer)
    f1 = f1_score(extracted_answer, gold_answer)
    return em, lenient, f1


def majority_vote(answers: list) -> str:
    """Find most frequent normalised extracted answer among A0 + samples.
    Ties broken in favour of A0's answer (which is index 0)."""
    if not answers:
        return ""
    norm_counts = Counter()
    norm_to_orig = {}
    for ans in reversed(answers):  # so earlier items overwrite in norm_to_orig
        norm = normalise_hotpotqa(ans)
        norm_to_orig[norm] = ans
    for ans in answers:
        norm = normalise_hotpotqa(ans)
        norm_counts[norm] += 1
    
    # Tie breaking: if A0 is tied for max, pick A0
    a0_norm = normalise_hotpotqa(answers[0])
    max_count = max(norm_counts.values())
    if norm_counts[a0_norm] == max_count:
        return answers[0]
    
    # Otherwise return the most frequent
    best_norm, _ = norm_counts.most_common(1)[0]
    return norm_to_orig[best_norm]
