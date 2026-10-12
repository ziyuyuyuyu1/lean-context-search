"""Artificial CPU fixtures for the agreed fresh-query JSONL handoff.

Only the NumPy retrieval path executes. Encoder, corpus loader and reranker
model execution boundaries are mocked; this is not a neural parity benchmark.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lean_context_search import cli, encoder, qwen
from lean_context_search.cache import write_cache, text_hash
from lean_context_search.metrics import read_jsonl
from lean_context_search.qwen_adapter import PREFIX, SUFFIX

ROOT = Path(__file__).resolve().parents[1]


class FreshQueryHandoff(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        contract = json.loads((ROOT/'configs/encoder_contract.json').read_text())
        vectors = np.eye(237, dtype=np.float32)
        self.documents = write_cache(self.root/'documents', vectors,
            [f'Synthetic.{i}' for i in range(237)], 'documents',
            contract['source_contract_sha256'], 'synthetic-context')
        self.vector = np.arange(237, 0, -1, dtype=np.float32)
        self.query = 'synthetic fresh query with a typed interface'
        self.query_sha = text_hash(self.query)

    def search(self, extra=()):
        args = cli.parser().parse_args(['search', '--query', self.query,
            '--documents', str(self.documents), '--encoder', str(self.root/'no-model'),
            '--encoder-contract', str(ROOT/'configs/encoder_contract.json'),
            '--alpha', '0', *extra])
        stdout = io.StringIO()
        with patch.object(encoder, 'encode_local', return_value=self.vector), \
             contextlib.redirect_stdout(stdout):
            cli.command(args)
        return stdout.getvalue()

    def test_stdout_compatibility_and_default_two_hundred(self):
        row = json.loads(self.search())
        self.assertEqual(row['query'], self.query)
        self.assertEqual(row['query_text'], self.query)
        self.assertEqual(row['query_id'], 'query-'+self.query_sha)
        self.assertEqual(row['query_text_sha256'], self.query_sha)
        self.assertEqual(len(row['results']), 200)
        explicit = json.loads(self.search(['--top-k', '3']))
        self.assertEqual(len(explicit['results']), 3)

    def test_default_and_explicit_id_jsonl_export(self):
        for query_id in (None, 'user-fresh-query'):
            destination = self.root/('default.jsonl' if query_id is None else 'explicit.jsonl')
            extra = ['--output', str(destination)]
            if query_id is not None:
                extra += ['--query-id', query_id]
            self.search(extra)
            rows = read_jsonl(destination)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(destination.read_text().splitlines()), 1)
            row = rows[0]
            self.assertEqual(row['query_id'], query_id or 'query-'+self.query_sha)
            self.assertEqual(row['query_text'], self.query)
            self.assertEqual(row['query_text_sha256'], self.query_sha)
            self.assertEqual(len(row['results']), 200)
            qwen.validate_score_request(rows)

    def test_export_is_both_pool_and_query_file_for_default_rerank(self):
        pool_path = self.root/'fresh.jsonl'
        self.search(['--query-id', 'fixture-q', '--output', str(pool_path)])
        pools = read_jsonl(pool_path)
        contract = qwen.default_feature_contract()
        texts = [f'Synthetic.Passage.{i}' for i in range(237)]
        texts.extend(['Synthetic.Other'] * (contract['passage_inputs']['row_count'] - len(texts)))
        passages = qwen.VerifiedPassages(texts, qwen.ENRICHED_FEATURE_ID,
                                        contract['passage_inputs']['changed_row_ids'])
        torch = ModuleType('torch')
        torch.bfloat16 = 'synthetic BF16 marker'
        torch.cuda = SimpleNamespace(is_available=lambda: True)
        torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))

        class FakeTokenizer:
            padding_side = 'left'
            def convert_tokens_to_ids(self, token):
                return {'yes':9693, 'no':2152}[token]
            def encode(self, text, **kwargs):
                return [0]*{PREFIX:39, SUFFIX:9}[text]
            def __call__(self, bodies, **kwargs):
                return {'input_ids': [[0]*12 for _ in bodies]}

        class FakeModel:
            def to(self, device): return self
            def eval(self): return self
            def parameters(self): return []

        transformers = ModuleType('transformers')
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTokenizer())
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *a, **kw: FakeModel())
        fc = contract['feature_contract']
        hashes = {'tokenizer.json': fc['tokenizer_sha256'],
                  'model.safetensors.index.json': fc['model_index_sha256'],
                  **fc['model_shards_sha256']}
        observed = []
        def prepare(tokenizer, queries, passages, instruction, device, limit):
            self.assertTrue(all(query == self.query for query in queries))
            self.assertEqual(instruction, contract['prompt']['instruction'])
            self.assertEqual(limit, 8192)
            return {'synthetic_rows': [int(passage.rsplit('.', 1)[1]) for passage in passages]}
        def extract(model, inputs, yes, no):
            rows = inputs['synthetic_rows']
            observed.extend(rows)
            return {'base_logits': np.array(rows, dtype=np.float32)}

        output = self.root/'reranked.jsonl'
        args = cli.parser().parse_args(['rerank', '--pools', str(pool_path),
            '--query-file', str(pool_path), '--passages', str(self.root/'mocked-passages.pkl'),
            '--model', str(self.root), '--output', str(output)])
        with patch.dict('sys.modules', {'torch':torch, 'transformers':transformers}), \
             patch.object(qwen, 'verify_passages', return_value=passages), \
             patch.object(qwen, 'validate_runtime_assets', return_value={}), \
             patch.object(qwen, 'sha', side_effect=lambda path: hashes[Path(path).name]), \
             patch.object(qwen, 'prepare_inputs', side_effect=prepare), \
             patch.object(qwen, 'extract_pair_features', side_effect=extract), \
             patch.object(qwen, 'load_distribution_head', side_effect=AssertionError('margin must not load a reranker head')), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.command(args)
        candidate_ids = [row['row_id'] for row in pools[0]['results']]
        self.assertEqual(observed, candidate_ids)
        row = read_jsonl(output)[0]
        self.assertEqual(row['candidate_count'], 200)
        self.assertEqual(row['query_id'], 'fixture-q')
        self.assertEqual(row['query_text_sha256'], self.query_sha)
        self.assertEqual(row['mode'], 'qwen_margin')
        self.assertFalse(row['reranker_head_applied'])
        self.assertEqual(len(row['results']), 100)
        self.assertEqual([hit['row_id'] for hit in row['results']], sorted(candidate_ids, reverse=True)[:100])
        self.assertEqual([hit['score'] for hit in row['results']], sorted(candidate_ids, reverse=True)[:100])


if __name__ == '__main__':
    unittest.main(verbosity=2)
