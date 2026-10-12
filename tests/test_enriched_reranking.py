"""CPU regression fixtures for the measured enriched-margin release default.

Synthetic feature values and mocked model boundaries are not neural parity
measurements. The separate real-cache replay supplies measured rank evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lean_context_search import cli, qwen, runtime_assets
from lean_context_search.qwen_adapter import PREFIX, SUFFIX
from test_live_validation import copy_measured_metadata, changed_json, FakeModel, FakeTokenizer


ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class EnrichedReranking(unittest.TestCase):
    def setUp(self):
        self.contract = qwen.default_feature_contract()
        self.original = json.loads((ROOT/'configs/qwen_feature_contract.json').read_text())
        self.changed = self.contract['passage_inputs']['changed_row_ids']
        self.changed_row = self.changed[0]
        unchanged = [i for i in range(100) if i not in self.changed]
        self.rows = [unchanged[0], self.changed_row, unchanged[1]]
        texts = ['synthetic unchanged passage'] * self.contract['passage_inputs']['row_count']
        for row in self.changed:
            texts[row] = 'synthetic enriched passage'
        # This fixture stands in for the verified asset-loader boundary. The
        # real corpus and both transport forms are independently replayed.
        self.passages = qwen.VerifiedPassages(texts, qwen.ENRICHED_FEATURE_ID, self.changed)
        self.query = {'query_id': 'fixture', 'query_text': 'synthetic interface query'}
        self.pool = {'query_id': 'fixture', 'query_text_sha256': digest(self.query['query_text']),
                     'results': [{'row_id': row, 'name': 'Fixture.'+str(row)} for row in self.rows]}
        self.features = [
            {'query_id': 'fixture', 'query_text_sha256': self.pool['query_text_sha256'],
             'row_id': row, 'passage_sha256': digest(self.passages[row]),
             'feature_contract_id': qwen.ENRICHED_FEATURE_ID if row in self.changed else qwen.ORIGINAL_FEATURE_ID,
             'efficient_margin': [1.0, 2.0, 2.0][i]}
            for i, row in enumerate(self.rows)]

    def test_registered_contracts_and_default_cli(self):
        self.assertEqual(qwen.validate_feature_contract(None)['feature_contract_id'], qwen.ENRICHED_FEATURE_ID)
        self.assertEqual(qwen.validate_feature_contract(self.original), self.original)
        self.assertEqual((ROOT/'configs/qwen_enriched_feature_contract.json').read_bytes(),
                         (ROOT/'src/lean_context_search/qwen_enriched_feature_contract.json').read_bytes())
        fc = self.contract['feature_contract']
        self.assertEqual(hashlib.sha256(json.dumps(fc, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                         qwen.ENRICHED_FEATURE_ID)
        for action, additional in [('rerank-cached', ['--features','features.jsonl']),
                                   ('rerank', ['--query-file','queries.jsonl','--model','local-model'])]:
            args = cli.parser().parse_args([action,'--pools','pools.jsonl','--passages','passages.pkl',
                                           '--output','out.jsonl', *additional])
            self.assertEqual((args.mode, args.beta, args.head, args.feature_contract), ('margin',0.0,None,None))
        for key in ('prompt', 'passage_inputs', 'feature_contract'):
            altered = copy.deepcopy(self.contract)
            if key == 'prompt': altered[key]['instruction'] = 'changed'
            if key == 'passage_inputs': altered[key]['changed_row_ids'] = []
            if key == 'feature_contract': altered[key]['max_length'] = 1024
            with self.subTest(key=key), self.assertRaises(ValueError):
                qwen.validate_feature_contract(altered, require_full=True)

    def test_default_margin_mixed_origins_without_head_or_hidden(self):
        with patch.object(qwen, 'load_distribution_head', side_effect=AssertionError('head must not load')):
            result = qwen.score_saved_features([self.pool], self.features, passages=self.passages)
        row = result[0]
        expected = sorted(self.features, key=lambda item: (-item['efficient_margin'], item['row_id']))
        self.assertEqual([r['row_id'] for r in row['results']], [f['row_id'] for f in expected])
        self.assertEqual([r['score'] for r in row['results']], [f['efficient_margin'] for f in expected])
        self.assertFalse(row['reranker_head_applied'])
        self.assertEqual(row['feature_origin_counts'], {qwen.ORIGINAL_FEATURE_ID: 2, qwen.ENRICHED_FEATURE_ID: 1})
        self.assertEqual(row['feature_contract_id'], qwen.ENRICHED_FEATURE_ID)

    def test_cached_identity_and_origin_fail_closed(self):
        changes = [
            ('feature_contract_id', '0'*64),
            ('query_text_sha256', '0'*64),
            ('passage_sha256', '0'*64),
            ('efficient_margin', float('nan')),
            ('row_id', True),
        ]
        for key, value in changes:
            altered = copy.deepcopy(self.features)
            altered[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                qwen.score_saved_features([self.pool], altered, passages=self.passages)
        altered = copy.deepcopy(self.features)
        altered[1]['feature_contract_id'] = qwen.ORIGINAL_FEATURE_ID
        with self.assertRaisesRegex(ValueError, 'changed enriched passage'):
            qwen.score_saved_features([self.pool], altered, passages=self.passages)
        for passages in [list(self.passages), qwen.VerifiedPassages(self.passages, qwen.ORIGINAL_FEATURE_ID),
                         qwen.VerifiedPassages(self.passages[:3], qwen.ENRICHED_FEATURE_ID, self.changed)]:
            with self.assertRaisesRegex(ValueError, 'bound passages'):
                qwen.score_saved_features([self.pool], self.features, passages=passages)
        for features in [self.features[:2], self.features+self.features[:1]]:
            with self.assertRaises(ValueError):
                qwen.score_saved_features([self.pool], features, passages=self.passages)

    def test_explicit_historical_modes_and_head_policy(self):
        with self.assertRaisesRegex(ValueError, 'head is only accepted'):
            qwen.score_saved_features([self.pool], self.features, ROOT/'heads/qwen', passages=self.passages)
        with self.assertRaisesRegex(ValueError, 'beta=0'):
            qwen.score_saved_features([self.pool], self.features, passages=self.passages, beta=1.0)
        with self.assertRaisesRegex(ValueError, 'explicit positive beta'):
            qwen.score_saved_features([self.pool], self.features, mode='trained', passages=self.passages)
        for mode, beta in [('stock',0.0), ('trained',1.0)]:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'historical original contract'):
                qwen.score_saved_features([self.pool], self.features, mode=mode, beta=beta, passages=self.passages)
        original_features = copy.deepcopy(self.features)
        for item in original_features:
            item['feature_contract_id'] = qwen.ORIGINAL_FEATURE_ID
            item['stock_probability'] = .5
        result = qwen.score_saved_features([self.pool], original_features, contract=self.original,
                                          passages=list(self.passages), mode='stock')
        self.assertEqual([x['row_id'] for x in result[0]['results']], sorted(self.rows))

    def test_overlay_validates_actual_hashes_rows_and_texts(self):
        # Miniature transport fixture: only the contract validator is replaced;
        # passage and overlay SHA checks and reconstruction execute unchanged.
        original = ['old zero', 'old one', 'old two']
        enriched = ['old zero', 'new one', 'old two']
        entry = {'row':1,'changed':True,'original_text':original[1],'enriched_text':enriched[1],
                 'original_sha256':digest(original[1]),'candidate_sha256':digest(enriched[1])}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, overlay = root/'original.pkl', root/'enriched.pkl', root/'changed.jsonl'
            old.write_bytes(pickle.dumps(original)); new.write_bytes(pickle.dumps(enriched))
            overlay.write_text(json.dumps(entry)+'\n')
            contract = copy.deepcopy(self.contract)
            binding = contract['passage_inputs']
            binding.update(original_passages_sha256=qwen.sha(old), enriched_passages_sha256=qwen.sha(new),
                           changed_passages_sha256=qwen.sha(overlay), row_count=3, changed_row_ids=[1])
            with patch.object(qwen, 'validate_feature_contract', return_value=contract):
                direct = qwen.verify_passages(new, contract)
                merged = qwen.verify_passages(old, contract, overlay)
                self.assertEqual(tuple(direct), tuple(enriched)); self.assertEqual(direct, merged)
                self.assertEqual(merged.changed_rows, {1})
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    qwen.verify_passages(old, contract)
                original_bytes = overlay.read_bytes()
                overlay.write_bytes(original_bytes+b' ')
                with self.assertRaisesRegex(ValueError, 'overlay checksum'):
                    qwen.verify_passages(old, contract, overlay)
                for mutate in [lambda e: e.update(row=0), lambda e: e.update(original_text='stale'),
                               lambda e: e.update(candidate_sha256='0'*64)]:
                    bad = copy.deepcopy(entry); mutate(bad)
                    overlay.write_text(json.dumps(bad)+'\n')
                    binding['changed_passages_sha256'] = qwen.sha(overlay)
                    with self.assertRaises(ValueError): qwen.verify_passages(old, contract, overlay)
                overlay.write_text(json.dumps(entry)+'\n'+json.dumps(entry)+'\n')
                binding['changed_passages_sha256'] = qwen.sha(overlay)
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    qwen.verify_passages(old, contract, overlay)
                overlay.write_text(''); binding['changed_passages_sha256'] = qwen.sha(overlay)
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    qwen.verify_passages(old, contract, overlay)

    def test_long_input_batch_plan(self):
        lengths = [100, 1000, 1024, 1024, 4097, 8192, 2000, 2000, 3000]
        batches = qwen.plan_enriched_batches(lengths)
        self.assertEqual(batches, [(0,4),(4,5),(5,6),(6,8),(8,9)])
        self.assertEqual([i for a,b in batches for i in range(a,b)], list(range(len(lengths))))
        for start, stop in batches:
            self.assertLessEqual(stop-start, 4)
            self.assertTrue(stop-start == 1 or max(lengths[start:stop])*(stop-start) <= 4096)
        for values in [[8193],[0],[True]]:
            with self.assertRaises(ValueError): qwen.plan_enriched_batches(values)
        with self.assertRaises(ValueError): qwen.plan_enriched_batches([100], microbatch=5)

    def test_enriched_live_boundaries_and_decoder_route(self):
        constructed, tokenizers, decoder_calls = [], [], []
        torch = ModuleType('torch')
        torch.bfloat16 = 'mock BF16'
        torch.cuda = SimpleNamespace(is_available=lambda: True)
        torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))
        transformers = ModuleType('transformers')

        class Tokenizer(FakeTokenizer):
            def __call__(self, bodies, **kwargs):
                self_test.assertEqual(kwargs['max_length'], 8144)
                return {'input_ids': [[0]*52, [0]*4952, [0]*152][:len(bodies)]}

        self_test = self
        def make_tokenizer(*args, **kwargs):
            tokenizers.append(kwargs); return Tokenizer()
        def make_model(*args, **kwargs):
            constructed.append(kwargs); return FakeModel()  # not callable: full LM path would fail
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=make_tokenizer)
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=make_model)
        def prepare(tokenizer, queries, passages, instruction, device, max_length):
            return {'fixture_size': len(queries)}
        def extract(model, inputs, yes, no):
            decoder_calls.append(inputs['fixture_size'])
            return {'base_logits': np.zeros(inputs['fixture_size'], dtype=np.float32)}
        fc = self.contract['feature_contract']
        model_hashes = {'tokenizer.json':fc['tokenizer_sha256'], 'model.safetensors.index.json':fc['model_index_sha256'],
                        **fc['model_shards_sha256']}
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('sys.modules', {'torch':torch,'transformers':transformers}), \
             patch.object(qwen, 'verify_passages', return_value=self.passages), \
             patch.object(qwen, 'sha', side_effect=lambda p:model_hashes[Path(p).name]), \
             patch.object(runtime_assets, 'version', side_effect=lambda p:{'transformers':'4.57.6','tokenizers':'0.22.2'}[p]), \
             patch.object(qwen, 'load_distribution_head', side_effect=AssertionError('head loaded in default margin')), \
             patch.object(qwen, 'prepare_inputs', side_effect=prepare), \
             patch.object(qwen, 'extract_pair_features', side_effect=extract):
            root = Path(directory)
            copy_measured_metadata(root, 'qwen', self.contract)
            def run(**kwargs):
                return qwen.score_live([self.pool], [self.query], root/'passages.pkl', root, **kwargs)
            for filename, mutation in [('config.json', {'rms_norm_eps':.01}),
                                       ('tokenizer_config.json', {'pad_token':'changed'})]:
                old = changed_json(root/filename, mutation)
                with self.assertRaisesRegex(ValueError, filename): run()
                (root/filename).write_bytes(old)
            with patch.object(runtime_assets, 'version', return_value='incorrect'):
                with self.assertRaisesRegex(ValueError, 'version differs'): run()
            (root/'special_tokens_map.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unexpected runtime override'): run()
            (root/'special_tokens_map.json').unlink()
            with self.assertRaisesRegex(ValueError, 'microbatch'): run(microbatch=5)
            with self.assertRaisesRegex(ValueError, 'head is only accepted'): run(head_dir=ROOT/'heads/qwen')
            self.assertEqual((len(constructed),len(tokenizers)), (0,0))
            result = run()
            self.assertEqual(decoder_calls, [1,1,1])
            self.assertEqual((len(constructed),len(tokenizers)), (1,1))
            self.assertEqual(constructed[0]['attn_implementation'], 'sdpa')
            self.assertTrue(constructed[0]['local_files_only'])
            self.assertFalse(result[0]['reranker_head_applied'])
            changed = copy.deepcopy(self.contract)
            changed['feature_contract']['max_length'] = 4096
            with self.assertRaises(ValueError): run(contract=changed)
            self.assertEqual(len(constructed), 1)


if __name__ == '__main__':
    unittest.main()
