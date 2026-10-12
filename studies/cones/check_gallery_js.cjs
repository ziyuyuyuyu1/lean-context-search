/* Replay the page's own pairMath on every saved directed pair without a browser. */
const fs=require('fs');
const assert=require('assert/strict');
const path=require('path');
const input=process.argv[2] || path.resolve(__dirname,'../../site/index.html');
const html=fs.readFileSync(input,'utf8');
const data=JSON.parse(html.match(/<script id="real-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const controller=html.match(/<script id="site-controller">([\s\S]*?)<\/script>/)[1];
const source=controller.match(/  function pairMath\(st,s,t\) \{[\s\S]*?\n  \}/)[0];
const pairMath=Function('data',source+'\nreturn pairMath;')(data);
assert.equal(data.dimension,128);
assert.ok(controller.includes('data-checkpoint') && controller.includes('data-extent') && controller.includes('swap-pair'));
assert.ok(html.includes('212 / 212') && html.includes('601') && html.includes('<noscript>'));
assert.ok(html.includes('data-checkpoint="0"'));
assert.ok(html.includes('data-extent="common"'));
for(const id of ['summary-positive','summary-negative','source-picker','result-picker','summary-cards','export-svg'])
  assert.ok(html.includes('id="'+id+'"'),id+' control absent');
assert.ok(controller.includes('$("summary-positive").textContent') && controller.includes('$("summary-negative").textContent'));
for(const id of ['before-cone','after-cone','known-matrix','learned-matrix'])
  assert.ok(new RegExp('<svg id="'+id+'"[\\s\\S]*?<\\/svg>').test(html),id+' static SVG absent');
assert.ok(!/<(?:script|img|link)[^>]+(?:src|href)="https?:/i.test(html));
assert.ok(!/fetch\s*\(/.test(controller));
function decode(encoded){const buffer=Buffer.from(encoded,'base64');const out=new Float64Array(buffer.length/4);for(let i=0;i<out.length;i++)out[i]=buffer.readFloatLE(i*4);return out;}
let checks=0,mismatch=0,closest=Infinity;
const detail={};
for(const [dataset,d] of Object.entries(data.datasets)){
  const n=d.nodes.length;detail[dataset]=[];
  const labels=new Uint8Array(n*n);
  for(const pair of d.pairs)labels[pair.s*n+pair.t]=pair.label==='positive'?1:2;
  for(const st of d.states){
    const u=decode(st.directions),rho=decode(st.rho),bits=Buffer.from(st.inside,'base64');
    for(let i=0;i<n;i++){let norm=0;for(let k=0;k<128;k++)norm+=u[i*128+k]**2;norm=Math.sqrt(norm);for(let k=0;k<128;k++)u[i*128+k]/=norm;}
    let stateMismatch=0;
    const categories={0:{inside:0,total:0},1:{inside:0,total:0},2:{inside:0,total:0}};
    for(let s=0;s<n;s++)for(let t=0;t<n;t++){
      if(s===t)continue;
      const gap=pairMath({u,rho},s,t).gap;
      const inside=(bits[(s*n+t)>>3]>>((s*n+t)&7))&1;
      const code=labels[s*n+t];
      categories[code].total++;categories[code].inside+=inside;
      if(!Number.isFinite(gap) || (gap<=0)!==Boolean(inside)){mismatch++;stateMismatch++;}
      closest=Math.min(closest,Math.abs(gap));checks++;
    }
    assert.deepEqual(categories,st.counts,dataset+' update '+st.update+' count mismatch');
    detail[dataset].push({update:st.update,pairs:n*(n-1),mismatch:stateMismatch});
  }
}
assert.equal(mismatch,0);
console.log(JSON.stringify({status:'passed',file:input,all_pair_checks:checks,
  decision_mismatches:mismatch,closest_absolute_gap:closest,states:detail,
  static_svgs:4,external_runtime_assets:0}));
