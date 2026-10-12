"""Portable command line for context, exact retrieval, saved replay and optional Qwen."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import __version__
from .cache import Cache, ExactRetriever, AffineHead, build_projection, write_cache
from .retrieval import rerank_vector, read_pools
from .heads import load_distribution_head
from .metrics import evaluate, replay_saved, read_jsonl


def write_lines(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def demo(_):
    # This is an explicitly synthetic software fixture, never a benchmark result.
    with TemporaryDirectory() as temp:
        root = Path(temp)
        docs = np.array([[1,0,0,0],[0,1,0,0],[.5,.5,0,0]],dtype=np.float32)
        query = np.array([1,.2,0,0],dtype=np.float32)
        dm = write_cache(root/'documents',docs,['JoinMeet','Unrelated','Mixed'],
                         'documents','fixture-encoder','fixture-context')
        retriever = ExactRetriever(dm)
        hits = retriever.search_vector(query,3)
        print(json.dumps({'kind':'synthetic_offline_fixture','query':'join and meet',
                          'top_names':[x['name'] for x in hits],'scores':[x['score'] for x in hits],
                          'network_calls':0,'model_calls':0},indent=2))


def command(args):
    if args.action in {'retrieve','reorder','search'} and args.top_k <= 0:
        raise ValueError('top_k must be a positive integer')
    if args.action in {'retrieve','reorder','search','rerank','rerank-cached'} and args.output:
        output=Path(args.output)
        if output.exists() or output.is_symlink():
            raise ValueError(f'output already exists: {output}; choose a new path')
        if not output.parent.is_dir():
            raise ValueError(f'output directory does not exist: {output.parent}')
    if args.action == 'demo':
        return demo(args)
    if args.action == 'prepare-context':
        from .context import prepare
        return prepare(args)
    if args.action == 'render-context':
        from .context import render
        return render(args)
    if args.action == 'project':
        return build_projection(Cache(args.documents,'documents'),args.head,args.output,args.chunk_size,args.allow_new_corpus)
    if args.action == 'verify-head':
        kind='shared_affine_cosine_head' if args.kind == 'embedding' else 'joint_pair_linear_residual'
        _,config=load_distribution_head(args.head,kind)
        print(json.dumps({'status':'passed','kind':args.kind,'weights_sha256':config['weights_sha256']}))
        return
    if args.action == 'replay':
        report=replay_saved(args.evidence,args.canonical)
        if args.output: Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return
    if args.action == 'evaluate':
        report=evaluate(read_jsonl(args.rankings),read_jsonl(args.labels))
        Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'full':report['full'],'fair':report['fair']},indent=2))
        return
    if args.action == 'rerank-cached':
        from .qwen import score_saved_features, validate_score_request, default_feature_contract
        contract=json.loads(Path(args.feature_contract).read_text()) if args.feature_contract else default_feature_contract()
        from .qwen import verify_passages
        pools=read_jsonl(args.pools)
        validate_score_request(pools,contract,args.mode,args.beta,args.return_k,require_full=True)
        passages=verify_passages(args.passages,contract,args.enriched_passages)
        rows=score_saved_features(pools,read_jsonl(args.features),
                                  args.head,contract,passages,args.mode,args.beta,args.return_k)
        write_lines(args.output,rows)
        print(json.dumps({'queries':len(rows),'mode':args.mode,'output':str(args.output)}))
        return
    if args.action == 'rerank':
        from .qwen import score_live, default_feature_contract
        contract=json.loads(Path(args.feature_contract).read_text()) if args.feature_contract else default_feature_contract()
        rows=score_live(read_jsonl(args.pools),read_jsonl(args.query_file),args.passages,
                        args.model,args.head,contract,args.mode,args.beta,args.microbatch,
                        args.return_k,args.device,args.enriched_passages)
        write_lines(args.output,rows)
        print(json.dumps({'queries':len(rows),'mode':args.mode,'output':str(args.output)}))
        return
    if args.action == 'search':
        from .encoder import encode_local, validate_encoder_contract
        if args.query_id == '':
            raise ValueError('query ID must be nonempty')
        contract=json.loads(Path(args.encoder_contract).read_text())
        source_digest=validate_encoder_contract(contract,args.allow_new_corpus)
    retriever = ExactRetriever(args.documents,args.head,args.projection,args.alpha,args.chunk_size,args.allow_new_corpus)
    if args.action == 'search':
        if source_digest != retriever.documents.meta['encoder_id']:
            raise ValueError('translated encoder contract ID differs from document cache encoder_id')
        vector=encode_local(args.query,args.encoder,args.encoder_contract,args.device,args.allow_new_corpus)
        digest=hashlib.sha256(args.query.encode()).hexdigest()
        result={'query_id':args.query_id or f'query-{digest}',
                'query':args.query,'query_text':args.query,'query_text_sha256':digest,
                'results':retriever.search_vector(vector,args.top_k)}
        if args.output:
            write_lines(args.output,[result])
            print(json.dumps({'queries':1,'action':'search','output':str(args.output)}))
        else:
            print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    queries=Cache(args.queries,'queries')
    retriever.documents.compatible_queries(queries)
    if args.action == 'retrieve':
        rows=[{'query_id':item['query_id'],'query_text_sha256':item['text_sha256'],
               'results':retriever.search_vector(queries.vectors[i],args.top_k)}
              for i,item in enumerate(queries.items)]
    elif args.action == 'reorder':
        pools=read_pools(args.pools,queries,retriever.documents)
        rows=[]
        for i,item in enumerate(queries.items):
            pool=pools[item['query_id']]
            rows.append({'query_id':item['query_id'],'query_text_sha256':item['text_sha256'],
                         'candidate_count':len(pool['results']),'mode':'fixed_pool_e0',
                         'results':rerank_vector(retriever,queries.vectors[i],
                                      [x['row_id'] for x in pool['results']],args.top_k)})
    else:
        raise ValueError('unknown action')
    write_lines(args.output,rows)
    print(json.dumps({'queries':len(rows),'action':args.action,'output':str(args.output)}))


def parser():
    p=argparse.ArgumentParser(prog='lean-context-search',description=__doc__)
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='action',required=True)
    d=sub.add_parser('demo',help='tiny synthetic CPU fixture; no downloads')
    d.add_argument('--offline',action='store_true',help='explicit fixture mode')
    pc=sub.add_parser('prepare-context',help='prepare pinned Lean typed-interface extraction')
    pc.add_argument('--db-dir',required=True);pc.add_argument('--output-dir',required=True)
    pc.add_argument('--imports',nargs='+',default=['Mathlib']);pc.add_argument('--snapshot',required=True)
    rc=sub.add_parser('render-context',help='render verified typed contexts as retrieval passages')
    rc.add_argument('--prepared-dir',required=True);rc.add_argument('--output-dir',required=True)
    rc.add_argument('--tokenizer-path',required=True);rc.add_argument('--max-tokens',type=int,required=True)
    rc.add_argument('--allow-unresolved-corpus-names',action='store_true')
    pr=sub.add_parser('project',help='precompute document embedding-head projections')
    pr.add_argument('--documents',required=True);pr.add_argument('--head',required=True)
    pr.add_argument('--output',required=True);pr.add_argument('--chunk-size',type=int,default=2048)
    pr.add_argument('--allow-new-corpus',action='store_true',help='explicit compatible new-corpus mode')
    for name in ['retrieve','reorder','search']:
        sp=sub.add_parser(name,help={'retrieve':'exact full-corpus cached retrieval',
                                     'reorder':'rescore a fixed candidate pool',
                                     'search':'fresh query with a separately supplied local encoder'}[name])
        sp.add_argument('--documents',required=True);sp.add_argument('--head');sp.add_argument('--projection')
        sp.add_argument('--alpha',type=float,default=.3,
                        help='selected embedding-head weight (default 0.30); use 0 for base embedding similarity')
        sp.add_argument('--top-k',type=int,default=100 if name=='reorder' else 200)
        sp.add_argument('--chunk-size',type=int,default=2048)
        sp.add_argument('--allow-new-corpus',action='store_true',help='explicit compatible new-corpus mode')
        if name=='search':
            sp.add_argument('--query',required=True);sp.add_argument('--encoder',required=True)
            sp.add_argument('--encoder-contract',required=True);sp.add_argument('--device',default='cuda:0')
            sp.add_argument('--query-id',help='optional query ID; default: query- followed by its text SHA256')
            sp.add_argument('--output',help='write a JSONL pool usable as both --pools and --query-file for rerank')
        else:
            sp.add_argument('--queries',required=True);sp.add_argument('--output',required=True)
        if name=='reorder': sp.add_argument('--pools',required=True)
    for name in ['rerank-cached','rerank']:
        sp=sub.add_parser(name,help='Qwen pair scoring over a fixed pool; never expands it')
        sp.add_argument('--pools',required=True)
        sp.add_argument('--feature-contract',help='default: installed measured enriched-passage contract; use the original contract for historical stock/trained modes')
        sp.add_argument('--mode',choices=['stock','margin','trained'],default='margin',help='default: efficient enriched Qwen margin, no added head')
        sp.add_argument('--head',help='historical trained mode only')
        sp.add_argument('--beta',type=float,default=0.0,help='0 for margin/stock; historical trained mode requires explicit positive beta (1.0)')
        sp.add_argument('--enriched-passages',help='registered changed_passages.jsonl overlay; --passages then supplies the trusted original texts.pkl')
        sp.add_argument('--return-k',type=int,default=100);sp.add_argument('--output',required=True)
        if name=='rerank-cached':
            sp.add_argument('--features',required=True);sp.add_argument('--passages',required=True)
        else:
            sp.add_argument('--passages',required=True);sp.add_argument('--query-file',required=True)
            sp.add_argument('--model',required=True);sp.add_argument('--microbatch',type=int,default=4)
            sp.add_argument('--device',default='cuda:0')
    ev=sub.add_parser('evaluate',help='evaluate public designated targets from saved rankings')
    ev.add_argument('--rankings',required=True);ev.add_argument('--labels',required=True);ev.add_argument('--output',required=True)
    re=sub.add_parser('replay',help='replay canonical public aggregates from saved ranks')
    re.add_argument('--evidence',required=True);re.add_argument('--canonical',required=True);re.add_argument('--output')
    vh=sub.add_parser('verify-head',help='verify a distributed NPZ head')
    vh.add_argument('--head',required=True);vh.add_argument('--kind',choices=['embedding','qwen'],required=True)
    return p


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    try:
        command(args)
    except (ValueError,KeyError,OSError) as exc:
        p.exit(2,f'lean-context-search: {exc}\n')


if __name__=='__main__':
    main()
