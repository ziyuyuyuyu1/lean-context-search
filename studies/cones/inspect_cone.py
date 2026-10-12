"""Inspect a real saved directed pair using full 128D normalized cone geometry."""
import argparse
import json
from pathlib import Path

from cone_geometry import state_gap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coordinates',type=Path,required=True)
    p.add_argument('--dataset',choices=['D0','D1'],required=True)
    p.add_argument('--update',type=int,choices=[0,100,300,1000],required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--result',required=True)
    a=p.parse_args()
    packet=json.loads(a.coordinates.read_text())
    d=packet['datasets'][a.dataset]
    names={node['name']:i for i,node in enumerate(d['nodes'])}
    if a.source not in names or a.result not in names or a.source==a.result:
        p.error('source/result must be distinct known nodes in this packet')
    state=next(s for s in d['states'] if s['update']==a.update)
    gap=float(state_gap(state,packet['K'])[names[a.source],names[a.result]])
    witness=next((r for r in d['train_pairs']+d['diagnostic_pairs']
                  if r['source']==a.source and r['result']==a.result),None)
    output={'dataset':a.dataset,'update':a.update,'source':a.source,'result':a.result,
            'normalized_gap':gap,'inside':gap<=0,
            'evidence_label':witness['label'] if witness else 'unknown_or_catalogue_unlisted',
            'certificate_scope':witness.get('relation_scope') if witness else d['original_final_audit']['certificate_scope'],
            'witness_path':witness.get('witness_path') if witness else None,
            'checkpoint_sha256':state['checkpoint_parameter_sha256']}
    print(json.dumps(output,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
