"""Pure-data metric checks; no tensor library or model is imported."""

import unittest

from syntide.baseline.metrics import (
    fit_majority_answers,
    majority_predictions,
    parse_answer,
    score_predictions,
)


def row(template, answer, output, options=("yes", "no"), noisy=False, **extra):
    return {
        "template_id": template,
        "answer": answer,
        "answer_options": list(options),
        "output_text": output,
        "is_artifact": noisy,
        **extra,
    }


class MetricsTests(unittest.TestCase):
    def test_parser_deviation_is_explicit(self):
        self.assertEqual(parse_answer("Answer: no. ANSWER: YES!<eos>", ["yes", "no"]), "yes")
        self.assertIsNone(parse_answer("yes", ["yes", "no"]))
        self.assertEqual(parse_answer("yes", ["yes", "no"], mode="upstream"), "yes")
        self.assertIsNone(parse_answer("answer: yes", ["yes", "no"], mode="upstream"))
        self.assertEqual(parse_answer("Answer: ST/T change.", ["st/t change", "no"]), "stt change")
        self.assertIsNone(parse_answer("Answer: maybe", ["yes", "no"]))

    def test_template_aggregation_includes_absent_classes(self):
        records = [
            row(1, "yes", "Answer: yes"),
            row(1, "yes", "Answer: yes"),
            row(1, "no", "Answer: yes", noisy=True),
            row(2, "yes", "Answer: yes", options=("yes", "no", "maybe")),
        ]
        scores = score_predictions(records)["groups"]
        all_scores = scores["all"]
        self.assertEqual(all_scores["n"], 4)
        self.assertEqual(all_scores["accuracy"], 0.75)
        self.assertAlmostEqual(all_scores["template_macro_accuracy"], (2 / 3 + 1) / 2)
        self.assertAlmostEqual(all_scores["template_macro_f1"], (0.4 + 1 / 3) / 2)
        self.assertAlmostEqual(all_scores["weighted_template_macro_f1"], (3 * 0.4 + 1 / 3) / 4)
        self.assertEqual(scores["noisy"]["accuracy"], 0)
        self.assertEqual(scores["clean"]["n"], 3)

    def test_invalid_output_stays_in_denominator_and_adds_false_negative(self):
        scores = score_predictions([row(1, "yes", "unparsed"), row(1, "yes", "Answer: yes")])["groups"]
        self.assertEqual(scores["all"]["accuracy"], 0.5)
        self.assertEqual(scores["all"]["invalid_percent"], 50)
        self.assertAlmostEqual(scores["all"]["weighted_template_macro_f1"], 1 / 3)
        self.assertIsNone(scores["noisy"]["accuracy"])

    def test_bad_inputs_raise_instead_of_silently_dropping_rows(self):
        with self.assertRaises(ValueError):
            score_predictions([row(1, "unknown", "Answer: yes")])
        with self.assertRaises(ValueError):
            score_predictions([row(1, "yes", "Answer: yes"), row(1, "yes", "Answer: yes", ("yes", "maybe"))])
        with self.assertRaises(ValueError):
            parse_answer("Answer: ab", ["a-b", "ab"])
        with self.assertRaises(ValueError):
            score_predictions([])

    def test_majority_uses_training_only_and_lexical_ties(self):
        fitted = fit_majority_answers([row(1, "yes", "", strat_fold=1), row(1, "no", "", strat_fold=8)])
        self.assertEqual(fitted, {"1": "no"})
        predictions = majority_predictions([row(1, "yes", "", strat_fold=10)], fitted)
        self.assertEqual(predictions[0]["output_text"], "Answer: no")
        self.assertEqual(score_predictions(predictions)["groups"]["all"]["accuracy"], 0)
        with self.assertRaises(ValueError):
            fit_majority_answers([row(1, "yes", "", strat_fold=10)])


if __name__ == "__main__":
    unittest.main()
