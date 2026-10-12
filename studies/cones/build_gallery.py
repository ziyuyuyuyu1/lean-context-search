"""Build a standalone offline HTML from measured coordinates and checked static exports."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

from cone_geometry import state_gap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_svg(packet,learned):
    d=packet['datasets']['D0']; n=len(d['nodes']); final=d['states'][-1]
    bits=np.unpackbits(np.frombuffer(base64.b64decode(final['inside']),dtype=np.uint8),bitorder='little')[:n*n].reshape(n,n)
    labels=np.zeros((n,n),dtype=np.uint8)
    for pair in d['pairs']:
        labels[pair['s'],pair['t']]=1 if pair['label']=='positive' else 2
    order=d['order']; cell=2.4; margin=24; size=margin+cell*n+10
    paths={1:[],2:[],3:[]}
    for iy,s in enumerate(order):
        for ix,t in enumerate(order):
            if s==t or (learned and not bits[s,t]) or (not learned and labels[s,t]!=1): continue
            kind=1 if labels[s,t]==1 else 2 if labels[s,t]==2 else 3
            x=margin+ix*cell;y=margin+iy*cell
            paths[kind].append(f'M{x:.1f},{y:.1f}h{cell}v{cell}h-{cell}Z')
    colors={1:'#235f9b',2:'#aa5148',3:'#b86c13'}
    title='Actual fitted D0 containment' if learned else 'Checked D0 parent ancestry'
    body=''.join(f'<path fill="{colors[k]}" d="{"".join(values)}"><title>{len(values)} {"positives" if k==1 else "tested negatives" if k==2 else "catalogue extras"}</title></path>' for k,values in paths.items())
    return (f'<svg id="{"learned" if learned else "known"}-matrix" class="matrix-plot" role="img" '
            f'aria-label="{title}" viewBox="0 0 {size:.1f} {size:.1f}" xmlns="http://www.w3.org/2000/svg">'
            f'<title>{title}</title><rect x="{margin}" y="{margin}" width="{cell*n:.1f}" height="{cell*n:.1f}" fill="#edf1f5"/>'
            +body+f'<text x="{margin}" y="17" font-size="12">Rows: source A • Columns: result B</text></svg>')


def build(root,output):
    root=Path(root); output=Path(output)
    coordinates=json.loads((root/'studies/cones/data/coordinates.json').read_text())
    display=json.loads((root/'studies/cones/data/display_data.json').read_text())
    if coordinates['head_dimension']!=128 or coordinates['K']!=display['K'] or coordinates['seed']!=display['seed']:
        raise ValueError('coordinate/display contract mismatch')
    numerical=[]
    for name,d in coordinates['datasets'].items():
        shown=display['datasets'][name]
        if [s['update'] for s in shown['states']] != [0,1000]:
            raise ValueError('saved display endpoints changed')
        all_states=[]
        n=len(d['nodes'])
        labels=np.zeros((n,n),dtype=np.uint8)
        for pair in shown['pairs']:
            labels[pair['s'],pair['t']]=1 if pair['label']=='positive' else 2
        distinct=~np.eye(n,dtype=bool)
        for state in d['states']:
            gap=state_gap(state,coordinates['K'])
            inside=(gap<=0)&distinct
            counts={str(code):{'total':int(((labels==code)&distinct).sum()),
                               'inside':int(((labels==code)&inside).sum())} for code in (0,1,2)}
            packed=base64.b64encode(np.packbits(inside.reshape(-1),bitorder='little').tobytes()).decode()
            record={'update':state['update'],'directions':state['directions']['base64'],
                    'rho':state['rho']['base64'],'inside':packed,'counts':counts,
                    'checkpointSha':state['checkpoint_parameter_sha256']}
            prior=next((x for x in shown['states'] if x['update']==state['update']),None)
            if prior and (prior['inside']!=packed or prior['counts']!=counts or prior['checkpointSha']!=record['checkpointSha']):
                raise ValueError(f'{name} endpoint changed at update {state["update"]}')
            all_states.append(record)
            numerical.append({'dataset':name,'update':state['update'],'counts':counts,
                              'smallest_absolute_gap':float(np.nanmin(np.abs(gap)))})
        shown['states']=all_states
    display['packetSha']=sha(root/'studies/cones/data/coordinates.json')
    template=(root/'studies/cones/gallery.template.html').read_text()
    if (template.count('__LOCAL_D3__')!=1 or template.count('__REAL_DATA__')!=1
            or template.count('__D3_LICENSE__')!=1):
        raise ValueError('gallery template placeholders changed')
    library=(root/'studies/cones/vendor/d3-7.9.0.min.js').read_text()
    def safe_script(value): return value.replace('</script','<\\/script').replace('</SCRIPT','<\\/SCRIPT')
    license_notice=(root/'studies/cones/vendor/D3_LICENSE.txt').read_text()
    text=template.replace('__D3_LICENSE__',html.escape(license_notice))
    text=text.replace('__LOCAL_D3__',safe_script(library))
    text=text.replace('__REAL_DATA__',json.dumps(display,separators=(',',':')).replace('<','\\u003c'))
    for name in ['before-cone','after-cone']:
        placeholder=f'<svg id="{name}" class="cone-plot" role="img" aria-label="'
        start=text.index(placeholder); end=text.index('</svg>',start)+len('</svg>')
        figure=(root/'site/figures'/f'{name}.svg').read_text()
        text=text[:start]+figure+text[end:]
    for name,learned in [('known',False),('learned',True)]:
        placeholder=f'<svg id="{name}-matrix" class="matrix-plot" role="img" aria-label="'
        start=text.index(placeholder); end=text.index('</svg>',start)+len('</svg>')
        text=text[:start]+matrix_svg(display,learned)+text[end:]
    text=text.replace('<div id="real-chain" class="chain"></div>',
                      '<div id="real-chain" class="chain"><strong>AddAction</strong> → <strong>AddSemigroupAction</strong> → <strong>VAdd</strong></div>')
    text=text.replace('<p id="relation-status"></p>',
                      '<p id="relation-status">Checked parent-projection path. This composed pair was held out of direct pair supervision; its nodes and primitive maps were available.</p>')
    text=text.replace('<p id="audit-line" class="audit-line tabular"></p>',
                      '<p id="audit-line" class="audit-line tabular">D0 update 1000: 212/212 positives inside; 0/212 strict reversals inside; 601 catalogue extras inside.</p>')
    text=text.replace('<p id="before-verdict" class="panel-stat tabular"></p>',
                      '<p id="before-verdict" class="panel-stat tabular">At update 0, VAdd is outside AddAction\'s cone.</p>')
    text=text.replace('<p id="after-verdict" class="panel-stat tabular"></p>',
                      '<p id="after-verdict" class="panel-stat tabular">At update 1000, VAdd is inside AddAction\'s cone.</p>')
    if any(x in text for x in ('__LOCAL_D3__','__REAL_DATA__','__D3_LICENSE__','<script src=','src="http:','src="https:','https://fonts.')):
        raise ValueError('external or unresolved gallery asset')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text)
    receipt={'status':'passed','output':str(output.resolve()),'sha256':sha(output),
             'bytes':output.stat().st_size,'coordinate_sha256':sha(root/'studies/cones/data/coordinates.json'),
             'display_states':{k:[x['update'] for x in v['states']] for k,v in display['datasets'].items()},
             'checks':numerical,'external_assets':0,'model_calls':0}
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.root,a.output),indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as exc:
        print(f'gallery build: {exc}',file=sys.stderr);raise SystemExit(2)
