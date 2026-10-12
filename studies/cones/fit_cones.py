"""Bounded structure-only free-point and frozen-input-head fits."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from lean_context_search.cache import AffineHead,Cache,normalize,sha256
from radius_head import RadiusHead
from cone_loss import cone_terms,signed_loss,old_signed_loss,RHO_MIN,RHO_MAX


class FreePoints(torch.nn.Module):
    def __init__(self,directions,device):
        super().__init__()
        self.raw_direction=torch.nn.Parameter(torch.tensor(directions,dtype=torch.float32,device=device))
        initial=math.log((1-RHO_MIN)/(RHO_MAX-1))
        self.radius_logit=torch.nn.Parameter(torch.full((len(directions),),initial,dtype=torch.float32,device=device))
    def forward(self,unused=None):
        norms=torch.linalg.vector_norm(self.raw_direction,dim=-1)
        if bool((norms<=1e-7).any()):raise ArithmeticError('Free direction zero')
        u=self.raw_direction/norms[:,None]
        r=RHO_MIN+(RHO_MAX-RHO_MIN)*torch.sigmoid(self.radius_logit)
        return u,r,norms


def pair_data(nodes,pairs,device):
    names={x['name']:i for i,x in enumerate(nodes)}
    return (torch.tensor([names[p['source']] for p in pairs],device=device),
            torch.tensor([names[p['result']] for p in pairs],device=device),
            torch.tensor([p['label']=='positive' for p in pairs],dtype=torch.bool,device=device))


def group_metrics(terms,pairs):
    rows=[]
    for p,g,d in zip(pairs,terms['normalized_gap'].detach().cpu().tolist(),terms['clearance'].detach().cpu().tolist()):
        rows.append((p['label'],p['path_length'],float(g),float(d)))
    def one(label,kind):
        selected=[x for x in rows if x[0]==label and (kind=='all' or ((x[1]==1)==(kind=='direct')))]
        return {'n':len(selected),'contained':sum(x[2]<=0 for x in selected),
                'contained_fraction':sum(x[2]<=0 for x in selected)/len(selected) if selected else None,
                'mean_normalized_gap':float(np.mean([x[2] for x in selected])) if selected else None}
    return {label:{kind:one(label,kind) for kind in ('all','direct','composed')}
            for label in ('positive','negative')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='D0',choices=('D0','D1'))
    p.add_argument('--kind',choices=('free','head'),required=True)
    p.add_argument('--geometry',choices=('euclidean','hyperbolic'),required=True)
    p.add_argument('--loss',choices=('normalized','old'),default='normalized')
    p.add_argument('--seed',type=int,default=20261001)
    p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--max-seconds',type=float,default=180)
    p.add_argument('--documents',type=Path,required=True,help='Compatible frozen C cache manifest')
    p.add_argument('--head',type=Path,required=True,help='Pinned E0 head directory')
    p.add_argument('--nodes',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New output directory; never overwrites a fit')
    args=p.parse_args()
    if args.updates<1 or args.max_seconds<=0:
        p.error('Require positive update and time caps')
    nodes_path=args.nodes;train_path=args.train;diagnostic_path=args.diagnostic
    label=f'{args.dataset}_{args.kind}_{"E" if args.geometry=="euclidean" else "H"}_{args.loss}_seed{args.seed}'
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('GPU expected for bounded follow-up fit')
    started=time.monotonic()
    nodes=[json.loads(x) for x in nodes_path.read_text().splitlines()]
    train=[json.loads(x) for x in train_path.read_text().splitlines()]
    diag=[json.loads(x) for x in diagnostic_path.read_text().splitlines()]
    if not nodes or not train or not any(x['label']=='positive' for x in train) or not any(x['label']=='negative' for x in train):
        raise ValueError('Balanced eligible structural data required')
    docs=Cache(args.documents,'documents')
    if any(docs.items[x['row_id']]!=x['name'] for x in nodes):raise ValueError('C row mapping drift')
    x=torch.tensor(normalize(np.array(docs.vectors[[x['row_id'] for x in nodes]],copy=True)),dtype=torch.float32,device=device)
    old,oldmeta=AffineHead.load(args.head)
    if oldmeta.get('document_cache_sha256',oldmeta.get('original_C_manifest_sha256'))!=docs.digest:
        raise ValueError('E0/C cache binding mismatch')
    model=FreePoints(old.project(x.detach().cpu().numpy()),device) if args.kind=='free' else RadiusHead(old.weights,old.bias,device=device)
    old_parameters={name:t.detach().clone() for name,t in model.named_parameters()}
    train_s,train_r,train_pos=pair_data(nodes,train,device)
    diag_s,diag_r,diag_pos=pair_data(nodes,diag,device)
    lr=.01 if args.kind=='free' else .001
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    cfg={'schema':'cone-structure-fit-v1','dataset':args.dataset,'kind':args.kind,'geometry':args.geometry,
         'loss':args.loss,'seed':args.seed,'updates_cap':args.updates,'seconds_cap':args.max_seconds,
         'learning_rate':lr,'weight_decay':.0001,'gradient_clip':1.0,'full_balanced_train_pairs':len(train),
         'nodes':len(nodes),'diagnostic_pairs':len(diag),'K':.1,'radius_bounds':[.4,2.],
         'normalized_margin':.1 if args.loss=='normalized' else None,
         'old_raw_margin':.02 if args.loss=='old' else None,
         'C_manifest_sha256':docs.digest,'E0_manifest_sha256':sha256(args.head/('manifest.json' if (args.head/'manifest.json').is_file() else 'config.json')),
         'node_file_sha256':sha256(nodes_path),'train_file_sha256':sha256(train_path),
         'diagnostic_file_sha256':sha256(diagnostic_path),'source_code_sha256':sha256(__file__),
         'confirmation_content_read':False}
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    history=[];checkpoints=[]
    def evaluate(update):
        with torch.no_grad():
            u,r,norms=model(None if args.kind=='free' else x)
            tr=cone_terms(u[train_s],u[train_r],r[train_s],r[train_r],args.geometry)
            dg=cone_terms(u[diag_s],u[diag_r],r[diag_s],r[diag_r],args.geometry)
            pg=tr['raw_gap'] if args.loss=='old' else tr['normalized_gap']
            value=old_signed_loss(pg[train_pos],pg[~train_pos]) if args.loss=='old' else signed_loss(pg[train_pos],pg[~train_pos])
            meta={'update':update,'elapsed_seconds':time.monotonic()-started,'training_loss':float(value),
                  'train':group_metrics(tr,train),'diagnostic':group_metrics(dg,diag),
                  'radius_min':float(r.min()),'radius_median':float(r.median()),'radius_max':float(r.max()),
                  'radius_near_lower':int((r<=.401).sum()),'radius_near_upper':int((r>=1.999).sum()),
                  'minimum_raw_direction_norm':float(norms.min()),
                  'minimum_tangent_length':min(float(tr['tangent_length'].min()),float(dg['tangent_length'].min())),
                  'cosine_clamps':tr['cosine_clamp_count']+dg['cosine_clamp_count'],
                  'parameter_delta_l2':{name:float(torch.linalg.vector_norm(t-old_parameters[name])) for name,t in model.named_parameters()}}
            checkpoint=output/f'checkpoint_{update:04d}'
            checkpoint.mkdir(exist_ok=False)
            np.savez(checkpoint/'parameters.npz',**{name:t.detach().cpu().numpy() for name,t in model.named_parameters()})
            (checkpoint/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
            checkpoints.append(meta)
            print(json.dumps({'checkpoint':label,'update':update,'loss':meta['training_loss'],
                              'positive_containment':meta['train']['positive']['all']['contained_fraction'],
                              'negative_false_containment':meta['train']['negative']['all']['contained_fraction'],
                              'elapsed_seconds':meta['elapsed_seconds']}),flush=True)
    evaluate(0)
    last=0
    for update in range(1,args.updates+1):
        if time.monotonic()-started>=args.max_seconds:break
        optim.zero_grad(set_to_none=True)
        u,r,_=model(None if args.kind=='free' else x)
        terms=cone_terms(u[train_s],u[train_r],r[train_s],r[train_r],args.geometry)
        gaps=terms['raw_gap'] if args.loss=='old' else terms['normalized_gap']
        loss=old_signed_loss(gaps[train_pos],gaps[~train_pos]) if args.loss=='old' else signed_loss(gaps[train_pos],gaps[~train_pos])
        if not bool(torch.isfinite(loss)):raise ArithmeticError('Nonfinite loss')
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True))
        if not math.isfinite(grad):raise ArithmeticError('Nonfinite gradient')
        optim.step()
        if any(not bool(torch.isfinite(t).all()) for t in model.parameters()):raise ArithmeticError('Nonfinite parameter')
        history.append({'update':update,'loss':float(loss.detach()),'gradient_preclip_l2':grad,
                        'gradient_postclip_l2':min(grad,1.0),
                        'minimum_tangent_length':float(terms['tangent_length'].min().detach()),
                        'cosine_clamps':terms['cosine_clamp_count']})
        last=update
        if update in (100,300,1000):evaluate(update)
    (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (output/'run_manifest.json').write_text(json.dumps({**cfg,'completed_updates':last,
        'elapsed_seconds':time.monotonic()-started,'checkpoints':[v['update'] for v in checkpoints],
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'max_gradient_preclip_l2':max((h['gradient_preclip_l2'] for h in history),default=None),
        'nonfinite_updates':0,'numerical_unknowns':0},indent=2)+'\n')

if __name__=='__main__':main()
