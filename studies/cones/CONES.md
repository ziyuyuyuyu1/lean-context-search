# Directed relationship experiment: scoped evidence and limits

The retrieval pipeline uses an ordinary embedding head and frozen Qwen reranking without an added reranker head. The
experimental cone track asks whether directed supplier relationships can be
represented by geometric containment. The [offline explorer](../../site/index.html)
shows actual saved coordinates, not an artist's projection.

The [scientific overview](../../site/figures/hyperbolic-overview.svg) compares the
same measured `AddAction → AddSemigroupAction → VAdd` path before and after
fitting. Panels a/b share one x/y extent; panel c is a labelled 11.25× detail
of the final state. At update 0, `AddAction → VAdd` has angle 120.64° against
a 9.80° aperture and lies outside; at update 1000, its 8.90° angle fits the
16.39° aperture. The [fidelity matrices](../../site/figures/relation-fidelity.svg)
show the complete D0 directed-pair predictions beside the checked parent
catalogue. Amber cells are absent from that catalogue, not proved semantic
nonimplications. SVG figures are included; raster exports can be generated when needed.

For D0, `A → B` means a value of interface A supplies B through recorded,
parameterized Lean parent projections and their composition. The D0 packet has
181 canonical indexed nodes and 143 checked direct maps. Training used 201
positive arrows and 201 strict reversed ancestry negatives; 11 composed
positives and their 11 reversals were held from direct pair supervision.
The 32,156 other ordered distinct pairs lack a path in this *declared-parent
catalogue*. They are catalogue incomparables, not certified general semantic
nonimplications. A held composition uses already seen nodes and is not a
fresh-interface out-of-distribution test.

D1 instead concerns context-indexed Prop-valued predicates on the **same**
mathematical object or morphism. Its 11 eligible nodes span three explicit
contexts. Eight evidence-consuming arrows and eight checked countermodels
were training labels; two composed positives and two countermodels were held
diagnostics. For example, `CategoryTheory.IsIso f → IsSplitMono f → Mono f`
preserves the category, endpoints and the same morphism `f`. `IsSplitMono f
→ Epi f` has a Lean-checked `Unit ⟶ Bool` countermodel. `Mono f →
IsSplitMono f` is *unknown* in this packet, not negative. The other 90 D1
distinct pairs remain unknown absent a certificate. These data do not model
arbitrary joint-premise proof composition.

For unit directions `u` and radii `ρ` in curvature −1, the experiment fixes
`K=0.1`, `ρ∈[0.4,2]`, dimension 128, and aperture
`ψ_A=asin(2K/sinh(ρ_A))`. The stable full-dimensional source tangent
angle is `ξ(A,B)`. The normalized signed gap is
`g=(cos ψ_A−cos ξ)/(1−cos ψ_A)`; `g≤0` means B lies inside A's cone, with
self/coincident pairs excluded by the original contract. The fitted loss is
`0.5 mean(relu(g_positive+0.1)) + 0.5 mean(relu(0.1−g_negative))`.
This aperture-normalized loss corrects an earlier raw-margin feasibility
problem: at hyperbolic source radius 1, the old 0.02 cosine margin exceeded
the maximum positive clearance 0.014588. That diagnostic alone does not
explain the old pilot's retrieval failure; later structure-only fits reached
zero hinge after radii moved.

At update 1000, the displayed shared hyperbolic D0 head contains all
**212/212** positives, no **0/212** strict reversals, and **601/32,156**
catalogue extras. Catalogue precision is `212/(212+601)=26.08%`; the fit is
not exact ancestry reconstruction. The D1 head contains **10/10** positives,
**0/10** checked negatives, and **1/90** unknown pairs. Flat shared heads
also fit the 212 D0 and 10 D1 positives while rejecting checked negatives;
the flat D0 head adds 837 catalogue extras. A free flat fit obtains 210/212
positives and 60 extras. Thus finite fitting is possible in both geometries
and is not a hyperbolic-specific success. The normalized structure-only head
fits used seed 20261001, AdamW, 1000 updates, and saved 0/100/300/1000
checkpoints; [actual configs](configs/) and [public scoped labels](data/)
record inputs. The coordinate packet preserves parameter SHA256 values and
certificate scope.

The matched frozen-E0 retrieval branches did not establish a gain: across
three seeds, development nDCG@10 averaged 0.69105 for E0, 0.68797 for
hyperbolic relevance+cone, and 0.68797 for its flat counterpart. A shuffled
structural-label branch fit only 2/212 true D0 positives yet matched the
true hyperbolic branch on all 87 accepted-answer nDCG@10 values for one
seed. This separates numerical correctness and structural fitting from
relevance improvement. A separate exact typed-graph bridge reached 1.0 on
a constrained 109-candidate/72-requirement known-catalogue supplier task;
learned hyperbolic **and flat** bridges could also reach 1.0 with the same
posterior and node inventory. That is not 100% general LeanSearch accuracy
or fresh-interface transfer. Later new-endpoint packets remained small and
mixed; no hyperbolic retrieval/OOD advantage is established.

To rebuild the measured SVG/PNG figures and standalone explorer from release
assets alone, install the optional visualization extra and run from the
repository root:

```bash
python -m pip install -e '.[visualization]'
python studies/cones/build_structural_figures.py --root . --output-dir site/figures
python studies/cones/build_gallery.py --root . --output site/index.html
node studies/cones/check_gallery_js.cjs site/index.html
python studies/cones/inspect_cone.py --coordinates studies/cones/data/coordinates.json \
  --dataset D0 --update 1000 --source AddAction --result VAdd
python studies/cones/inspect_cone.py --coordinates studies/cones/data/coordinates.json \
  --dataset D1 --update 1000 --source CategoryTheory.IsSplitMono --result CategoryTheory.Epi
```

Download `site/index.html` alone and open it as a local file to explore without
network access or neighboring assets. The Download this view control saves a
new standalone HTML with the selected packet, pair, checkpoint and extent; the
right cone can be exported as a labelled SVG. If a repository host shows HTML
source, download/open the file or serve it as a static page. 

The explorer's folded Poincaré view places source A at `(tanh(ρ_A/2),0)` and
each result i at `(tanh(ρ_i/2)·dot(u_A,u_i),
tanh(ρ_i/2)·sqrt(1−dot(u_A,u_i)^2))`. This preserves each source/result
radius, origin angle, hyperbolic distance and cone decision, but it does not
preserve all result/result distances at once. All-pair matrix membership is
computed from the full 128D states, not the display plane. The chosen
checkpoint, source/result and common/independent zoom can be saved with the
standalone HTML. Static figures remain visible with JavaScript disabled.

`fit_cones.py` adapts the original finite-fit trainer into an **optional**
explicit-input CLI. A full rerun needs the separately held 310,579-row C
vector cache with ordered row IDs and the pinned E0 head. For example:

```bash
python studies/cones/fit_cones.py --dataset D0 --kind head --geometry hyperbolic \
  --loss normalized --seed 20261001 --updates 1000 --max-seconds 180 \
  --documents /assets/C/manifest.json --head heads/e0 \
  --nodes studies/cones/data/D0_nodes.jsonl \
  --train studies/cones/data/D0_train_pairs.jsonl \
  --diagnostic studies/cones/data/D0_diagnostic_pairs.jsonl \
  --output /tmp/new-D0-fit
```

The bundled measured coordinates and labels suffice to replay the all-pair decisions and gallery
without the nonredistributable C cache. The cone theory reference is Ganea,
Bécigneul and Hofmann, [*Hyperbolic Entailment Cones for Learning
Hierarchical Embeddings*](https://proceedings.mlr.press/v80/ganea18a.html),
ICML 2018.
