"""Public contract checks for the installed NumPy paths and frozen assets."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lean_context_search.cache import Cache, ExactRetriever, build_projection, text_hash, write_cache
from lean_context_search.heads import load_distribution_head, qwen_scores
from lean_context_search.metrics import replay_saved
from lean_context_search.qwen import score_saved_features
from lean_context_search.retrieval import rerank_vector


ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def test_explicit_new_corpus_and_fixed_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            encoder=json.loads((ROOT/'heads/e0/config.json').read_text())['encoder_contract_id']
            docs=np.zeros((4,4096),dtype=np.float32)
            docs[0,0]=1;docs[1,1]=1;docs[2,0]=.5;docs[2,1]=.5;docs[3]=docs[0]
            q=np.zeros((1,4096),dtype=np.float32);q[0,0]=1
            dm=write_cache(root/'docs',docs,['Fixture.A','Fixture.B','Fixture.Mixed','Fixture.Tied'],
                           'documents',encoder,'fixture-representation','complete-context-v1')
            qm=write_cache(root/'queries',q,[{'query_id':'fixture-q','text_sha256':text_hash('A'),
                                             'split':'benchmark'}],
                           'queries',encoder,'fixture-query')
            cache=Cache(dm,'documents')
            with self.assertRaisesRegex(ValueError,'explicit new-corpus'):
                build_projection(cache,ROOT/'heads/e0',root/'bad-projection')
            build_projection(cache,ROOT/'heads/e0',root/'projection',allow_new_corpus=True)
            retriever=ExactRetriever(dm,ROOT/'heads/e0',root/'projection',alpha=.3,allow_new_corpus=True)
            retriever.documents.compatible_queries(Cache(qm,'queries'))
            hits=retriever.search_vector(q[0],4)
            self.assertEqual([x['row_id'] for x in hits],
                             [x['row_id'] for x in rerank_vector(retriever,q[0],[3,2,1,0])])
            self.assertEqual([x['row_id'] for x in rerank_vector(retriever,q[0],[1,2])],
                             sorted([1,2],key=lambda row:next(i for i,x in enumerate(hits) if x['row_id']==row)))
            self.assertEqual([x['row_id'] for x in rerank_vector(retriever,q[0],[3,0])],[0,3])
            with self.assertRaises(ValueError): rerank_vector(retriever,q[0],[0,0])
            with self.assertRaises(ValueError): rerank_vector(retriever,q[0],[4])
            with self.assertRaises(ValueError): rerank_vector(retriever,np.zeros(4096),[0])

    def test_pair_feature_contract_and_passage_binding(self):
        delta,cfg=load_distribution_head(ROOT/'heads/qwen','joint_pair_linear_residual')
        hidden=np.ones((2,4096),dtype=np.float32)
        hidden[1,0]=2
        margin=[-1.,2.]
        values=qwen_scores(hidden,margin,delta)
        passages=['original A','original B']
        qhash=text_hash('query')
        features=[{'query_id':'q','row_id':i,'query_text_sha256':qhash,
                   'passage_sha256':text_hash(passages[i]),
                   'feature_contract_id':cfg['feature_contract_id'],
                   'efficient_margin':margin[i],'stock_probability':.5,
                   'hidden':hidden[i].tolist()} for i in (0,1)]
        pools=[{'query_id':'q','query_text_sha256':qhash,
                'results':[{'row_id':i,'name':f'Fixture.{i}'} for i in (0,1)]}]
        contract={'feature_contract_id':cfg['feature_contract_id']}
        got=score_saved_features(pools,features,ROOT/'heads/qwen',contract,passages,mode='trained',beta=1.0)
        self.assertEqual([x['row_id'] for x in got[0]['results']],
                         sorted([0,1],key=lambda i:(-values[i],i)))
        with self.assertRaisesRegex(ValueError,'passage'):
            score_saved_features(pools,features,ROOT/'heads/qwen',contract,['changed','original B'],mode='trained',beta=1.0)
        with self.assertRaisesRegex(ValueError,'contract'):
            score_saved_features(pools,features,ROOT/'heads/qwen',{'feature_contract_id':'wrong'},passages,mode='trained',beta=1.0)

    def test_saved_public_aggregate_replay(self):
        result=replay_saved(ROOT/'results/public_per_query.jsonl',ROOT/'results/public_metrics.json')
        self.assertEqual((result['queries'],result['fair_queries']),(946,810))
        self.assertLessEqual(result['max_absolute_error'],1e-12)

    def test_heads_are_exact_original_bytes(self):
        expected={'e0':'d2cbc75546284bb6f9297c21aea0be536494e7bc07baf9e64cc8ec5a975a6700',
                  'qwen':'2134a403f2ef2ac26be0c9a69cb222bbc12a2bef3afae6c0e99c7dff2bcdc1d7'}
        for name,digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT/'heads'/name/'head.npz').read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
