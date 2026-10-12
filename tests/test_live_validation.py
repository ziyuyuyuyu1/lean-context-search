"""CPU-only regression checks; model and asset reads are mocked, never inferred."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from lean_context_search import encoder, qwen, runtime_assets
from lean_context_search.qwen_adapter import PREFIX, SUFFIX, prepare_inputs


ROOT = Path(__file__).resolve().parents[1]
MODEL_METADATA_ROOT = Path(os.environ.get(
    'LEAN_CONTEXT_TEST_MODELS', '/root/LeanSearch-v2/data/cache/models'))


def copy_measured_metadata(destination, profile, contract):
    """Copy original small/tokenizer metadata, never model weights, for CPU probes."""
    model_name = 'Qwen3-Embedding-8B' if profile == 'embedding' else 'Qwen3-Reranker-8B'
    source = MODEL_METADATA_ROOT / model_name
    manifest = runtime_assets._manifest_profile(profile)
    missing = [name for name in manifest['required_sha256'] if not (source / name).is_file()]
    if missing:
        raise unittest.SkipTest(f'local measured metadata unavailable: {model_name}: {missing}')
    for name in manifest['required_sha256']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    if profile == 'embedding':
        # The large pinned shards are never copied. Only the old hash reader is
        # mocked; its existence preflight still sees files in the fixture.
        for name in contract['model_file_hashes']:
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'large historical asset mocked in CPU test')


def changed_json(path, changes):
    original = path.read_bytes()
    data = json.loads(original)
    data.update(changes)
    path.write_text(json.dumps(data))
    return original


class ReachedInputBoundary(RuntimeError):
    pass


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []


class FakeTokenizer:
    padding_side = 'left'

    def convert_tokens_to_ids(self, token):
        return {'yes': 9693, 'no': 2152}[token]

    def encode(self, text, add_special_tokens=False):
        return [0] * {PREFIX: 39, SUFFIX: 9}[text]


class LiveValidation(unittest.TestCase):
    def setUp(self):
        self.qwen_contract = json.loads((ROOT/'configs/qwen_feature_contract.json').read_text())
        self.encoder_contract = json.loads((ROOT/'configs/encoder_contract.json').read_text())
        self.query = {'query_id': 'fixture', 'query_text': 'fixture query'}
        self.pool = {'query_id': 'fixture',
                     'query_text_sha256': hashlib.sha256(b'fixture query').hexdigest(),
                     'results': [{'row_id': i} for i in range(3)]}

    def test_qwen_five_counterexamples_and_canonical_input(self):
        canonical = self.qwen_contract
        fc = canonical['feature_contract']
        digest = hashlib.sha256(json.dumps(fc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, canonical['feature_contract_id'])
        self.assertEqual(hashlib.sha256(canonical['prompt']['instruction'].encode()).hexdigest(),
                         fc['instruction_sha256'])
        hashes = {'tokenizer.json': fc['tokenizer_sha256'],
                  'model.safetensors.index.json': fc['model_index_sha256'],
                  **fc['model_shards_sha256']}
        constructed = []
        tokenizers_constructed = []
        captured = []
        torch = ModuleType('torch')
        torch.bfloat16 = 'mock BF16'
        torch.cuda = SimpleNamespace(is_available=lambda: True)
        torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))
        transformers = ModuleType('transformers')
        def construct_tokenizer(*args, **kwargs):
            tokenizers_constructed.append((args, kwargs))
            return FakeTokenizer()

        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=construct_tokenizer)

        def construct(*args, **kwargs):
            constructed.append((args, kwargs))
            return FakeModel()

        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=construct)

        def capture(tokenizer, queries, passages, instruction, device, max_length):
            captured.append((list(queries), list(passages), instruction, device, max_length))
            raise ReachedInputBoundary

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('sys.modules', {'torch': torch, 'transformers': transformers}), \
             patch.object(qwen, 'sha', side_effect=lambda p: hashes[Path(p).name]), \
             patch.object(qwen, 'verify_passages', return_value=['a', 'b', 'c']), \
             patch.object(runtime_assets, 'version', side_effect=lambda name: {
                 'transformers': '4.57.6', 'tokenizers': '0.22.2'}[name]), \
             patch.object(qwen, 'prepare_inputs', side_effect=capture):
            root = Path(directory)
            copy_measured_metadata(root, 'qwen', canonical)

            def run(contract=canonical, pool=None, **kwargs):
                kwargs.setdefault('mode', 'trained')
                kwargs.setdefault('beta', 1.0)
                return qwen.score_live([pool or self.pool], [self.query], root/'passages.pkl',
                                       root, ROOT/'heads/qwen', contract, **kwargs)

            changed = copy.deepcopy(canonical)
            changed['prompt']['instruction'] = 'Changed instruction'
            with self.assertRaisesRegex(ValueError, 'prompt differ'):
                run(changed)
            changed = copy.deepcopy(canonical)
            changed['feature_contract']['max_length'] = 4096
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                run(changed)
            changed['feature_contract_id'] = hashlib.sha256(json.dumps(
                changed['feature_contract'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                run(changed)
            with self.assertRaisesRegex(ValueError, 'return_k'):
                run(return_k=-1)
            with self.assertRaisesRegex(ValueError, 'microbatch'):
                run(microbatch=0)
            with self.assertRaisesRegex(ValueError, 'beta'):
                run(beta=float('nan'))
            changed_pool = copy.deepcopy(self.pool)
            changed_pool['results'].append({'row_id': 0})
            with self.assertRaisesRegex(ValueError, 'candidate row ID'):
                run(pool=changed_pool)
            changed_pool = copy.deepcopy(self.pool)
            changed_pool['results'][0]['row_id'] = True
            with self.assertRaisesRegex(ValueError, 'candidate row ID'):
                run(pool=changed_pool)
            with self.assertRaisesRegex(ValueError, 'live query records'):
                qwen.score_live([self.pool], [], root/'passages.pkl', root,
                                ROOT/'heads/qwen', canonical, mode='trained', beta=1.0)
            self.assertEqual(len(constructed), 0)
            self.assertEqual(len(tokenizers_constructed), 0)

            for filename, changes in [
                ('config.json', {'rms_norm_eps': 0.01}),
                ('tokenizer_config.json', {'pad_token': '<|changed|>'})]:
                asset = root / filename
                original = changed_json(asset, changes)
                try:
                    with self.assertRaisesRegex(ValueError, filename):
                        run()
                    self.assertEqual(len(constructed), 0)
                    self.assertEqual(len(tokenizers_constructed), 0)
                finally:
                    asset.write_bytes(original)
            required = root / 'config.json'
            original = required.read_bytes()
            required.unlink()
            try:
                with self.assertRaisesRegex(ValueError, 'config.json'):
                    run()
            finally:
                required.write_bytes(original)
            optional = root / 'special_tokens_map.json'
            optional.write_text('{"pad_token":"<|changed|>"}')
            try:
                with self.assertRaisesRegex(ValueError, 'special_tokens_map.json'):
                    run()
            finally:
                optional.unlink()
            self.assertEqual(len(constructed), 0)
            self.assertEqual(len(tokenizers_constructed), 0)
            with self.assertRaises(ReachedInputBoundary):
                run()
            self.assertEqual(len(constructed), 1)
            self.assertEqual(len(tokenizers_constructed), 1)
            self.assertEqual(captured, [(['fixture query']*3, ['a', 'b', 'c'],
                                         canonical['prompt']['instruction'], 'cuda:0', 8192)])

    def test_encoder_changed_instruction_and_format_rejected_before_load(self):
        constructed = []
        captured = []

        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs):
                constructed.append((args, kwargs))

            def encode(self, prompts, **kwargs):
                captured.extend(prompts)
                raise ReachedInputBoundary

        st = ModuleType('sentence_transformers')
        st.SentenceTransformer = FakeSentenceTransformer
        torch = ModuleType('torch')
        torch.bfloat16 = 'mock BF16'
        torch.cuda = SimpleNamespace(is_available=lambda: True)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('sys.modules', {'torch': torch, 'sentence_transformers': st}), \
             patch.object(runtime_assets, 'version', side_effect=lambda name: {
                 'sentence-transformers': '5.2.2', 'transformers': '4.57.6',
                 'tokenizers': '0.22.2'}[name]), \
             patch.object(encoder, '_sha', side_effect=lambda p: self.encoder_contract['model_file_hashes'][Path(p).name]):
            root = Path(directory)
            copy_measured_metadata(root, 'embedding', self.encoder_contract)
            contract_path = root/'contract.json'

            def run(contract, allow_new_corpus=False):
                contract_path.write_text(json.dumps(contract))
                return encoder.encode_local('fixture query', root, contract_path,
                                            allow_new_corpus=allow_new_corpus)

            changed = copy.deepcopy(self.encoder_contract)
            changed['query_instruction'] = 'Changed embedding instruction'
            with self.assertRaisesRegex(ValueError, 'query_instruction'):
                run(changed)
            changed = copy.deepcopy(self.encoder_contract)
            changed['query_format'] = 'Query: {text}'
            with self.assertRaisesRegex(ValueError, 'query_format'):
                run(changed)
            self.assertEqual(len(constructed), 0)

            for filename, changes in [
                ('1_Pooling/config.json', {
                    'pooling_mode_lasttoken': False, 'pooling_mode_mean_tokens': True}),
                ('config_sentence_transformers.json', {'default_prompt_name': 'query'})]:
                asset = root / filename
                original = changed_json(asset, changes)
                try:
                    with self.assertRaisesRegex(ValueError, filename):
                        run(self.encoder_contract)
                    self.assertEqual(len(constructed), 0)
                finally:
                    asset.write_bytes(original)
            required = root / '1_Pooling/config.json'
            original = required.read_bytes()
            required.unlink()
            try:
                with self.assertRaisesRegex(ValueError, '1_Pooling/config.json'):
                    run(self.encoder_contract)
            finally:
                required.write_bytes(original)
            optional = root / 'sentence_bert_config.json'
            optional.write_text('{"max_seq_length":1}')
            try:
                with self.assertRaisesRegex(ValueError, 'sentence_bert_config.json'):
                    run(self.encoder_contract)
            finally:
                optional.unlink()
            self.assertEqual(len(constructed), 0)
            with self.assertRaises(ReachedInputBoundary):
                run(self.encoder_contract)
            self.assertEqual(len(constructed), 1)
            self.assertEqual(captured, [self.encoder_contract['query_format'].format(
                instruction=self.encoder_contract['query_instruction'], text='fixture query')])
            changed = copy.deepcopy(self.encoder_contract)
            changed['corpus_source_commit'] = 'new-corpus-fixture'
            with self.assertRaisesRegex(ValueError, 'corpus_source_commit'):
                run(changed)
            with self.assertRaises(ReachedInputBoundary):
                run(changed, allow_new_corpus=True)
            constructed_before = len(constructed)
            asset = root / '1_Pooling/config.json'
            original = changed_json(asset, {'pooling_mode_lasttoken': False,
                                            'pooling_mode_mean_tokens': True})
            try:
                with self.assertRaisesRegex(ValueError, '1_Pooling/config.json'):
                    run(changed, allow_new_corpus=True)
                self.assertEqual(len(constructed), constructed_before)
            finally:
                asset.write_bytes(original)

    def test_saved_feature_entry_rejects_stale_full_contract(self):
        changed = copy.deepcopy(self.qwen_contract)
        changed['prompt']['instruction'] = 'Changed instruction'
        with self.assertRaisesRegex(ValueError, 'prompt differ'):
            qwen.score_saved_features([self.pool], [], ROOT/'heads/qwen', changed,
                                      ['a', 'b', 'c'], mode='trained', beta=1.0)
        with self.assertRaisesRegex(ValueError, 'return_k'):
            qwen.score_saved_features([self.pool], [], ROOT/'heads/qwen',
                                      {'feature_contract_id': self.qwen_contract['feature_contract_id']},
                                      ['a', 'b', 'c'], mode='trained', beta=1.0, return_k=-1)

    def test_runtime_manifest_rechecks_bytes_and_binds_historical_ids(self):
        versions = {'sentence-transformers': '5.2.2', 'transformers': '4.57.6',
                    'tokenizers': '0.22.2'}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runtime_assets, 'version', side_effect=lambda name: versions[name]):
            root = Path(directory)
            embedding = root/'embedding'
            embedding.mkdir()
            copy_measured_metadata(embedding, 'embedding', self.encoder_contract)
            receipt = runtime_assets.validate_runtime_assets(
                embedding, 'embedding', self.encoder_contract)
            self.assertEqual(receipt['metadata_files_checked'], 5)
            self.assertEqual(receipt['absent_overrides_checked'], 16)
            asset = embedding/'generation_config.json'
            original = asset.read_bytes()
            asset.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'generation_config.json'):
                runtime_assets.validate_runtime_assets(embedding, 'embedding', self.encoder_contract)
            asset.write_bytes(original)
            runtime_assets.validate_runtime_assets(embedding, 'embedding', self.encoder_contract)

            qwen_dir = root/'qwen'
            qwen_dir.mkdir()
            copy_measured_metadata(qwen_dir, 'qwen', self.qwen_contract)
            receipt = runtime_assets.validate_runtime_assets(qwen_dir, 'qwen', self.qwen_contract)
            self.assertEqual(receipt['metadata_files_checked'], 6)
            self.assertEqual(receipt['absent_overrides_checked'], 8)
            changed = copy.deepcopy(self.qwen_contract)
            changed['feature_contract']['max_length'] = 4096
            changed['feature_contract_id'] = hashlib.sha256(json.dumps(
                changed['feature_contract'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'manifest/scientific contract binding mismatch'):
                runtime_assets.validate_runtime_assets(qwen_dir, 'qwen', changed)
            changed = copy.deepcopy(self.encoder_contract)
            changed['source_contract_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'manifest/scientific contract binding mismatch'):
                runtime_assets.validate_runtime_assets(embedding, 'embedding', changed)

    def test_runtime_manifest_rejects_unsafe_paths_and_wrong_loader(self):
        for path in ['../config.json', '/tmp/config.json', 'a//b', 'a\\b', '']:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'unsafe runtime asset path'):
                runtime_assets._safe_relative(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_measured_metadata(root, 'qwen', self.qwen_contract)
            with patch.object(runtime_assets, 'version', return_value='0.0.0'):
                with self.assertRaisesRegex(ValueError, 'measured loader version differs'):
                    runtime_assets.validate_runtime_assets(root, 'qwen', self.qwen_contract)

    def test_canonical_adapter_constructs_the_recorded_input(self):
        recorded = {}

        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def to(self, device):
                recorded['device'] = device
                return self

        class RecordingTokenizer(FakeTokenizer):
            def __call__(self, pairs, **kwargs):
                recorded['pairs'] = pairs
                recorded['tokenize_settings'] = kwargs
                return {'input_ids': [[7]]}

            def pad(self, encoded, **kwargs):
                recorded['encoded'] = encoded['input_ids']
                recorded['pad_settings'] = kwargs
                return {'input_ids': FakeTensor(encoded['input_ids']),
                        'attention_mask': FakeTensor([[1]*len(encoded['input_ids'][0])])}

        contract = self.qwen_contract
        result = prepare_inputs(RecordingTokenizer(), ['fixture query'], ['original passage'],
                                contract['prompt']['instruction'], 'cuda:0',
                                contract['feature_contract']['max_length'])
        self.assertEqual(recorded['tokenize_settings'],
                         {'padding': False, 'truncation': 'longest_first',
                          'return_attention_mask': False, 'add_special_tokens': False,
                          'max_length': contract['feature_contract']['body_max_length']})
        self.assertEqual(recorded['pad_settings'],
                         {'padding': True, 'return_tensors': 'pt', 'max_length': 8192})
        self.assertEqual(recorded['encoded'][0], [0]*39 + [7] + [0]*9)
        self.assertIn('<Instruct>: '+contract['prompt']['instruction'], recorded['pairs'][0])
        self.assertIn('<Query>: fixture query\n<Document>: original passage', recorded['pairs'][0])
        self.assertEqual(recorded['device'], 'cuda:0')
        self.assertEqual(set(result), {'input_ids', 'attention_mask'})


if __name__ == '__main__':
    unittest.main()
