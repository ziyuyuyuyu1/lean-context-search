"""Synthetic evaluation schema regressions; no neural models or benchmark runs."""
import copy
import unittest

from lean_context_search.cache import text_hash
from lean_context_search.metrics import evaluate, METRICS


class Evaluation(unittest.TestCase):
    def setUp(self):
        self.label = {'query_id': 'fixture', 'query_text': 'synthetic query',
                      'target_row': 1, 'target_name': 'Fixture.Target', 'fair': True}
        self.ranking = {'query_id': 'fixture', 'query_text_sha256': text_hash('synthetic query'),
                        'results': [{'row_id': 1, 'name': 'Fixture.Target'}]}

    def test_nonfair_subset_keeps_full_scores_and_undefined_fair_metrics(self):
        self.label['fair'] = False
        result = evaluate([self.ranking], [self.label])
        self.assertEqual(result['full']['queries'], 1)
        self.assertEqual(result['full']['ndcg@10'], 1.0)
        self.assertEqual(result['fair']['queries'], 0)
        self.assertTrue(all(result['fair'][key] is None for key in METRICS))

    def test_invalid_rows_cannot_receive_target_credit(self):
        for value in (True, False, -1, 1.0, '1'):
            with self.subTest(value=value):
                ranking = copy.deepcopy(self.ranking)
                ranking['results'][0]['row_id'] = value
                with self.assertRaisesRegex(ValueError, 'invalid ranked row'):
                    evaluate([ranking], [self.label])
                label = {**self.label, 'target_row': value}
                with self.assertRaisesRegex(ValueError, 'invalid target row'):
                    evaluate([self.ranking], [label])

    def test_empty_evaluation_and_nonboolean_membership_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            evaluate([], [])
        for value in ('false', 0, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'membership must be boolean'):
                evaluate([self.ranking], [{**self.label, 'fair': value}])

    def test_missing_corpus_target_is_zero_not_an_empty_cohort(self):
        result = evaluate([self.ranking], [{**self.label, 'target_row': None}])
        self.assertEqual(result['full']['targets_missing_from_corpus'], 1)
        self.assertTrue(all(result['full'][key] == 0.0 for key in METRICS))


if __name__ == '__main__':
    unittest.main()
