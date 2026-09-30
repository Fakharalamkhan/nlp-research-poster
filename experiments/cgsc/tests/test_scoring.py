"""Unit tests for scoring.py as specified in Step 4."""

import unittest
from cgsc.scoring import (
    extract_final_answer,
    score_gsm8k,
    exact_match_score,
    lenient_match_score,
    score_hotpotqa,
    majority_vote,
)


class TestScoring(unittest.TestCase):
    def test_gsm8k_numbers(self):
        gold = "1200"
        for pred in ["1200", "1200.0", "$1200", "1,200", "The cost is $1,200."]:
            em, lenient, f1 = score_gsm8k(pred, gold)
            self.assertTrue(em, f"Failed for {pred}")

        gold_12 = "12"
        for pred in ["12", "12.0", "$12"]:
            em, lenient, f1 = score_gsm8k(pred, gold_12)
            self.assertTrue(em, f"Failed for {pred}")

    def test_hotpotqa_cases(self):
        # "American" vs "The nationality is American." (EM False, lenient True)
        gold = "American"
        pred = "The nationality is American."
        self.assertFalse(exact_match_score(pred, gold))
        self.assertTrue(lenient_match_score(pred, gold))

        # "the Beatles" vs "Beatles" (EM True)
        self.assertTrue(exact_match_score("the Beatles", "Beatles"))
        self.assertTrue(exact_match_score("Beatles", "the Beatles"))

    def test_extract_final_answer(self):
        text = "Some steps...\nFinal answer: 42\nExtra thoughts..."
        self.assertEqual(extract_final_answer(text), "42")

        text_multiple = "Final answer: 10\nWait, recalculating.\nFinal answer: 20"
        self.assertEqual(extract_final_answer(text_multiple), "20")

        text_no_tag = "Step 1\nStep 2\n42"
        self.assertEqual(extract_final_answer(text_no_tag), "42")

    def test_majority_vote(self):
        answers = ["10", "20", "10", "30"]
        self.assertEqual(majority_vote(answers), "10")

        # Tie breaking: A0 is tied with another answer -> A0 wins
        answers_tie = ["10", "20"]
        self.assertEqual(majority_vote(answers_tie), "10")


if __name__ == "__main__":
    unittest.main()
