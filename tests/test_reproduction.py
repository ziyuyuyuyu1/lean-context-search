"""Reproduction guards and synthetic numerical checks; no real head training or inference."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPRO = Path(__file__).resolve().parents[1] / 'reproduction'
sys.path.insert(0, str(REPRO))
import benchmark
import build_caches
import common
import encode
import evaluate_development
import fetch_assets
import training_core
from lean_context_search.cache import AffineHead, Cache, normalize, write_cache


class Reproduction(unittest.TestCase):
    def test_bundled_data_and_protected_family_exclusion(self):
        manifest = common.verify_data()
        policy = common.read(REPRO / 'data/family_policy.json')
        records = common.rows(REPRO / 'data/train.jsonl')
        docs = Cache(REPRO / 'data/documents/manifest.json', 'documents')
        queries = Cache(REPRO / 'data/queries/manifest.json', 'queries')
        prepared = training_core.prepare_records(policy, records, docs, queries)
        self.assertEqual((len(prepared), len(docs.items)), (88, 696))
        self.assertEqual(len({policy['declaration_to_family'][r['source_names'][0]] for r in records}), 43)
        used = {n for r in records for k in ('positive_names', 'negative_names') for n in r[k]}
        self.assertEqual(set(docs.items), used)
        excluded = common.read(REPRO / 'data/benchmark_exclusions.json')
        self.assertFalse(used & (set(excluded['target_names']) | set(excluded['expanded_indexed_names'])))
        self.assertNotIn('confirmation', policy['family_split'].values())
        self.assertFalse(manifest['confirmation_content_included'])
        development = common.rows(REPRO / 'data/development.jsonl')
        self.assertEqual(len(development), 87)
        self.assertFalse({r['query_id'] for r in development} & {r['query_id'] for r in records})
        self.assertTrue(all(r['split'] == 'development' for r in development))
        for row in common.read(REPRO / 'data/document_rows.json'):
            self.assertEqual(docs.items[row['compact_row']], row['name'])
            self.assertEqual(hashlib.sha256(docs.vectors[row['compact_row']].tobytes()).hexdigest(), row['vector_sha256'])

    def test_protected_names_rejected_in_every_training_role(self):
        policy = common.read(REPRO / 'data/family_policy.json')
        original = common.rows(REPRO / 'data/train.jsonl')[0]
        name = next(n for n, f in policy['declaration_to_family'].items() if f in policy['protected_families'])
        for key in ('source_names', 'positive_names', 'negative_names', 'masked_alternative_names'):
            record = copy.deepcopy(original)
            record[key].append(name)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'nontraining'):
                common.check_training_record(policy, record)

    def test_analytic_gradient_matches_finite_difference_fixture(self):
        rng = np.random.default_rng(4)
        head = AffineHead(rng.normal(size=(3, 5)), rng.normal(size=3))
        q, docs = normalize(rng.normal(size=(1, 5))), normalize(rng.normal(size=(4, 5)))
        _, gw, gb = training_core.loss_gradient(head, q, docs, 2, .3, .07)
        epsilon = 1e-6
        for parameter, gradient in ((head.weights, gw), (head.bias, gb)):
            for index in np.ndindex(parameter.shape):
                old = parameter[index]
                parameter[index] = old + epsilon
                plus = training_core.loss_gradient(head, q, docs, 2, .3, .07)[0]
                parameter[index] = old - epsilon
                minus = training_core.loss_gradient(head, q, docs, 2, .3, .07)[0]
                parameter[index] = old
                self.assertAlmostEqual((plus - minus) / (2 * epsilon), gradient[index], places=6)

    def test_family_sampler_balances_styles_fixture(self):
        records = [{'source_names': [n], 'mixed_style': style} for n in ('A', 'B')
                   for style in ('statement', 'task') for _ in range(3 if n == 'A' else 1)]
        policy = {'declaration_to_family': {'A': 'fa', 'B': 'fb'}}
        first, count = training_core.groups_for_epoch(records, policy, np.random.default_rng(5))
        again, _ = training_core.groups_for_epoch(records, policy, np.random.default_rng(5))
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(first, again)
        selected = [(records[i]['source_names'][0], records[i]['mixed_style']) for i in first]
        self.assertEqual(set(selected), {('A', 'statement'), ('A', 'task'), ('B', 'statement'), ('B', 'task')})
        self.assertEqual(len(selected), 4)

    def test_pair_assembly_preserves_unmodified_rows_fixture(self):
        base = np.eye(3, dtype=np.float32)
        pairs = [{'row': 1, 'name': 'B'}]
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_caches.assemble(base, ['A', 'B', 'C'], pairs,
                lambda _: {'original': base[2], 'enriched': base[0]}, Path(tmp) / 'caches',
                'fixture-encoder', {'files': {}, 'shape': [3, 3], 'encoder_id': 'fixture-encoder'})
            for arm, index in (('original', 2), ('enriched', 0)):
                cache = Cache(paths[arm], 'documents')
                np.testing.assert_array_equal(cache.vectors[[0, 2]], base[[0, 2]])
                np.testing.assert_array_equal(cache.vectors[1], base[index])
                self.assertEqual(cache.items, ['A', 'B', 'C'])
            with self.assertRaisesRegex(ValueError, 'duplicate or unordered'):
                build_caches.assemble(base, ['A', 'B', 'C'], pairs * 2, None, Path(tmp) / 'bad', 'x', {})

    def test_query_text_and_split_bindings_fixture(self):
        contract = {'query_format': 'Instruction: {instruction}. Query: {text}', 'query_instruction': 'find'}
        record = {'query_id': 'fixture', 'query_text': 'a group'}
        self.assertEqual(encode.prepare_queries([record], 'train', contract)[0]['formatted'], 'Instruction: find. Query: a group')
        for change in ({'split': 'confirmation'}, {'query_text_sha256': '0' * 64}):
            with self.assertRaises(ValueError):
                encode.prepare_queries([{**record, **change}], 'train', contract)
        with self.assertRaises(ValueError):
            encode.prepare_queries([record], 'confirmation', contract)

    def test_development_binary_metrics_and_split_guard_fixture(self):
        label = {'query_id': 'fixture', 'query_text': 'fixture', 'split': 'development',
                 'judgment_status': 'accepted', 'judgment_provenance': 'synthetic', 'positive_names': ['A', 'B']}
        ranking = {'query_id': 'fixture', 'query_text_sha256': encode.text_hash('fixture'),
                   'results': [{'name': 'Other'}, {'name': 'A'}]}
        scores = evaluate_development.evaluate([label], [ranking], 'development', cutoffs=(2,))['metrics']
        self.assertEqual((scores['Recall@2'], scores['Hit@2']), (.5, 1.))
        self.assertAlmostEqual(scores['nDCG@2'], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        with self.assertRaises(ValueError):
            evaluate_development.evaluate([label], [ranking], 'confirmation', cutoffs=(2,))

    def test_encoder_overflow_rejected_before_encode_fixture(self):
        class Model:
            tokenizer = staticmethod(lambda *args, **kwargs: {'input_ids': [1, 2, 3]})
            tokenize = staticmethod(lambda *args: {'input_ids': np.array([[1, 2, 3]])})
            encode = staticmethod(lambda *args, **kwargs: self.fail('inference must not run'))
        with self.assertRaisesRegex(ValueError, 'overflows'):
            encode.encode_one(Model(), 'fixture', {'dimension': 3, 'embedding_limit_tokens': 2})

    def test_resume_rejects_wrong_query_or_nonfinite_vectors_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'one.npz'
            np.savez(path, vector=np.array([1., 0.], np.float32), query_id='fixture')
            with self.assertRaisesRegex(ValueError, 'input mismatch'):
                encode.shard_values(path, {'query_id': 'changed'}, ['vector'], 2)
            np.savez(path, vector=np.array([np.nan, 0.], np.float32), query_id='fixture')
            with self.assertRaisesRegex(ValueError, 'vector mismatch'):
                encode.shard_values(path, {'query_id': 'fixture'}, ['vector'], 2)

    def test_completed_resume_checks_assembled_cache_fixture(self):
        contract = {'dimension': 2, 'source_contract_sha256': 'fixture-encoder',
                    'query_format': '{text}', 'query_instruction': 'fixture'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.jsonl'
            common.write_rows(source, [{'query_id': 'fixture', 'query_text': 'fixture'}])
            output = root / 'encoding'
            (output / 'shards').mkdir(parents=True)
            expected = {'query_id': 'fixture', 'text_sha256': encode.text_hash('fixture')}
            np.savez(output / 'shards/00000000.npz', **expected, vector=np.array([1., 0.], np.float32))
            common.write(output / 'input_lock.json', {'kind': 'queries', 'input_sha256': common.sha(source),
                'rows': 1, 'encoder_contract_sha256': common.sha(REPRO.parent / 'configs/encoder_contract.json'),
                'split': 'benchmark', 'batch_size': 1})
            items = [{**expected, 'split': 'benchmark'}]
            write_cache(output / 'cache', np.array([[0., 1.]], np.float32), items, 'queries', 'fixture-encoder', 'fixture-representation')
            def read_fixture(path):
                if Path(path).name == 'encoder_contract.json':
                    return contract
                if Path(path).name == 'enriched_documents_manifest.json':
                    return {'representation_id': 'fixture-representation'}
                return common.read(path)
            args = argparse.Namespace(kind='queries', records=source, split='benchmark', output=output,
                resume=True, max_new=0, check_only=False, model=root / 'no-model', device='cuda:0')
            with patch.object(encode, 'read', side_effect=read_fixture), patch.object(encode, 'validate_encoder_contract'), \
                    patch.object(encode, 'load_encoder', side_effect=AssertionError('no inference')), \
                    self.assertRaisesRegex(ValueError, 'differs from its input shards'):
                encode.run(args)

    def test_benchmark_plan_has_independent_full_corpus_arms(self):
        args = argparse.Namespace(output=Path('out'), projection=None, allow_reencoded_cache=False,
            original_documents=Path('original'), enriched_documents=Path('enriched'), head=Path('head'),
            features=Path('scores'), passages=Path('texts'), enriched_passages=Path('overlay'))
        stages = dict(benchmark.plan(args))
        for key, documents, alpha in [('original_retrieval', 'original', '0'),
                                     ('enriched_retrieval', 'enriched', '0'), ('embedding_head', 'enriched', '0.30')]:
            command = stages[key]
            self.assertEqual(command[command.index('--top-k') + 1], '200')
            self.assertEqual(command[command.index('--documents') + 1], documents)
            self.assertEqual(command[command.index('--alpha') + 1], alpha)
        command = stages['qwen_reranked']
        self.assertIn('out/embedding_head.jsonl', command)
        self.assertEqual(command[command.index('--beta') + 1], '0')
        self.assertNotIn('--head', command)
        self.assertEqual(benchmark.ARMS['original_retrieval'][1], 'B_original_reencoded')

    def test_missing_or_duplicate_qwen_features_fail_fixture(self):
        pool = [{'query_id': 'fixture', 'query_text_sha256': 'query-hash', 'results': [{'row_id': 1}]}]
        feature = {'query_id': 'fixture', 'query_text_sha256': 'query-hash', 'row_id': 1}
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / 'features.jsonl', Path(tmp) / 'selected.jsonl'
            for values, message in (([], 'missing'), ([feature, feature], 'duplicate'),
                                    ([{**feature, 'query_text_sha256': 'wrong'}], 'mismatched')):
                path.write_text(''.join(json.dumps(v) + '\n' for v in values))
                with self.assertRaisesRegex(ValueError, message):
                    benchmark.filter_features(path, pool, output)
                self.assertFalse(output.exists())

    def test_download_integrity_and_existing_file_protection_fixture(self):
        data = b'synthetic fixture bytes'
        entry = {'path': 'model.bin', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                 'url': 'https://example.invalid/fixture'}
        asset = {'repository': 'synthetic', 'revision': 'fixture', 'files': [entry], 'total_bytes': len(data)}
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen', return_value=io.BytesIO(data)):
            self.assertEqual(fetch_assets.fetch(asset, tmp)['downloaded_files'], 1)
            with patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
                self.assertEqual(fetch_assets.fetch(asset, tmp, verify_only=True)['downloaded_files'], 0)
            (Path(tmp) / 'model.bin').write_bytes(b'preserve this')
            with self.assertRaisesRegex(ValueError, 'existing asset differs'):
                fetch_assets.fetch(asset, tmp)
            self.assertEqual((Path(tmp) / 'model.bin').read_bytes(), b'preserve this')


if __name__ == '__main__':
    unittest.main()
