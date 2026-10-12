"""Export measured cone figures from the saved 128D states; no model fitting.

The overview SVG and PNG come from the same Matplotlib figure. The two small
SVGs are genuine measured, common-extent fallbacks for the offline HTML.
"""
import argparse
import hashlib
import io
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from cone_geometry import state_gap, unpack


NAVY = '#203248'
BLUE = '#235f9b'
TEAL = '#317b6c'
AMBER = '#a76812'
RED = '#a94343'
MUTED = '#617183'
GRID = '#dce4e8'
SOFT = '#e9f3ef'
PAPER = '#ffffff'
OTHER = '#afbec8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_arrays(state):
    u = unpack(state['directions'])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, unpack(state['rho'])


def folded_points(state, source):
    u, rho = state_arrays(state)
    radius = np.tanh(rho / 2)
    cos = np.clip(u @ u[source], -1, 1)
    x = radius * cos
    y = radius * np.sqrt(np.maximum(0, 1 - cos * cos))
    x[source], y[source] = radius[source], 0.0
    return x, y, radius[source]


def pair_readout(state, source, result, k):
    u, rho = state_arrays(state)
    dot = float(np.clip(u[source] @ u[result], -1, 1))
    one_minus = float(np.sum((u[source] - u[result]) ** 2) / 2)
    axial = math.sinh(rho[result] - rho[source]) - math.cosh(rho[source]) * math.sinh(rho[result]) * one_minus
    transverse = math.sinh(rho[result]) * float(np.linalg.norm(u[result] - dot * u[source]))
    return {
        'angle_degrees': math.degrees(math.atan2(transverse, axial)),
        'aperture_degrees': math.degrees(math.asin(2 * k / math.sinh(rho[source]))),
        'normalized_gap': float(state_gap(state, k)[source, result]),
    }


def geodesic_cone(source_radius, aperture):
    """Same Poincare-ball geodesic boundary construction as gallery JS."""
    def side(sign):
        points = []
        for i in range(321):
            q = min(.9999999, 1 - (1 - i / 320) ** 3)
            zx, zy = q * math.cos(aperture), sign * q * math.sin(aperture)
            den = 1 + 2 * source_radius * zx + source_radius * source_radius * q * q
            points.append((((1 + 2 * source_radius * zx + q * q) * source_radius
                            + (1 - source_radius * source_radius) * zx) / den,
                           (1 - source_radius * source_radius) * zy / den))
        return np.asarray(points)
    upper, lower = side(1), side(-1)
    angle = math.atan2(upper[-1, 1], upper[-1, 0])
    arc = np.asarray([(math.cos(angle * (1 - 2 * i / 80)),
                       math.sin(angle * (1 - 2 * i / 80))) for i in range(1, 81)])
    return upper, lower, np.concatenate((upper, arc, lower[::-1]), axis=0)


def label_counts(coordinates, display, metrics):
    record = {}
    for packet, d in coordinates['datasets'].items():
        shown = display['datasets'][packet]
        n = len(d['nodes'])
        labels = np.zeros((n, n), dtype=np.uint8)
        for pair in shown['pairs']:
            labels[pair['s'], pair['t']] = 1 if pair['label'] == 'positive' else 2
        final = d['states'][-1]
        gap = state_gap(final, coordinates['K'])
        inside = gap <= 0
        np.fill_diagonal(inside, False)
        counts = {str(code): {'inside': int(np.sum(inside & (labels == code))),
                              'total': int(np.sum((labels == code) & ~np.eye(n, dtype=bool)))}
                  for code in (0, 1, 2)}
        expected = metrics[packet]
        assert counts['1']['inside'] == expected['final_positive_inside']
        assert counts['2']['inside'] == expected['final_negative_inside']
        assert counts['0']['inside'] == expected['final_catalogue_extra_inside' if packet == 'D0' else 'final_unknown_inside']
        assert counts['0']['total'] == expected['catalogue_other_pairs' if packet == 'D0' else 'unknown_pairs']
        held_key = 'held_composed_positive' if packet == 'D0' else 'held_positive'
        assert counts['1']['total'] == expected['train_positive'] + expected[held_key]
        record[packet] = {'labels': labels, 'inside': inside, 'counts': counts,
                          'node_count': n, 'update': final['update']}
    return record


def format_axes(ax, xlim, ylim, short=False, show_xlabel=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#fbfcfc')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=13 if not short else 8, colors=MUTED, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(4 if not short else 3))
    ax.yaxis.set_major_locator(MaxNLocator(4 if not short else 3))
    ax.grid(color=GRID, linewidth=.55, alpha=.65, zorder=0)
    if show_xlabel:
        ax.set_xlabel('Along A · ball units', fontsize=12 if not short else 8, color=MUTED, labelpad=3)
    ax.set_ylabel('Off A', fontsize=12 if not short else 8, color=MUTED, labelpad=3)


def draw_fold(ax, state, source, via, result, k, xlim, ylim, label_mode, inset=None, short=False, show_xlabel=True):
    x, y, radius = folded_points(state, source)
    readout = pair_readout(state, source, result, k)
    aperture = math.radians(readout['aperture_degrees'])
    upper, lower, polygon = geodesic_cone(radius, aperture)
    format_axes(ax, xlim, ylim, short, show_xlabel)
    ax.fill(polygon[:, 0], polygon[:, 1], color=SOFT, edgecolor='none', zorder=1)
    for boundary in (upper, lower):
        ax.plot(boundary[:, 0], boundary[:, 1], color=TEAL, linewidth=1.5, zorder=2)
    ax.axhline(0, color=GRID, linewidth=.9, zorder=2)
    visible = ((x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1]))
    background = visible.copy()
    background[[source, via, result]] = False
    ax.scatter(x[background], y[background], s=8 if not short else 5,
               color=OTHER, alpha=.52, linewidths=0, zorder=3)
    ax.scatter([x[source]], [y[source]], marker='D', s=78 if not short else 65,
               facecolor=NAVY, edgecolor=PAPER, linewidth=1, zorder=6)
    ax.scatter([x[via]], [y[via]], marker='o', s=62 if not short else 50,
               facecolor=BLUE, edgecolor=PAPER, linewidth=1, zorder=6)
    ax.scatter([x[result]], [y[result]], marker='o', s=135 if not short else 100,
               facecolor=PAPER, edgecolor=AMBER if readout['normalized_gap'] > 0 else TEAL,
               linewidth=2.4, zorder=7)
    offsets = {
        'before': [(-30, 17), (11, 15), (-30, 10)],
        'after': [(-43, 30), (0, 49), (34, 16)],
        'detail': [(-39, 24), (-7, 48), (22, 19)],
    }[label_mode]
    for index, text, offset in zip((source, via, result), ('A', 'via', 'B'), offsets):
        ax.annotate(text, (x[index], y[index]), xytext=offset, textcoords='offset points',
                    color=NAVY, fontsize=14 if not short else 9, fontweight='bold',
                    arrowprops={'arrowstyle': '-', 'color': MUTED, 'linewidth': .7},
                    ha='center', va='center', zorder=9)
    if inset is not None:
        ix, iy = inset
        ax.add_patch(Rectangle((ix[0], iy[0]), ix[1]-ix[0], iy[1]-iy[0],
                               fill=False, edgecolor=AMBER, linewidth=1.6,
                               linestyle=(0, (4, 3)), zorder=10))
        ax.text(ix[1]+.018, iy[1]+.028, 'zoom → c', fontsize=12,
                color=AMBER, va='bottom', zorder=10)
    return {'update': state['update'], **readout,
            'decision': 'inside' if readout['normalized_gap'] <= 0 else 'outside',
            'visible_nodes': int(visible.sum()), 'clipped_nodes': int(len(x) - visible.sum()),
            'x_domain': list(xlim), 'y_domain': list(ylim)}


def save_pair(fig, folder, basename, dpi=200):
    png = folder / (basename + '.png')
    svg = folder / (basename + '.svg')
    metadata = {'Creator': 'Lean Context Search saved-state figure builder', 'Date': None}
    fig.savefig(svg, format='svg', metadata=metadata, facecolor=PAPER)
    fig.savefig(png, format='png', dpi=dpi, metadata={'Software': 'Lean Context Search saved-state figure builder'}, facecolor=PAPER)
    return {kind: {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}
            for kind, path in (('svg', svg), ('png', png))}


def overview(coordinates, display, counted, folder, dpi):
    d = coordinates['datasets']['D0']
    names = [row['name'] for row in display['datasets']['D0']['nodes']]
    source, via, result = (names.index(x) for x in ('AddAction', 'AddSemigroupAction', 'VAdd'))
    before, after = d['states'][0], d['states'][-1]
    n = len(names)
    counts = counted['D0']['counts']
    positive, negative, other = (counts[str(code)] for code in (1, 2, 0))
    common_x = (.004, .724)
    common_y = (-.030, .330)
    detail_x = (.298, .362)
    detail_y = (-.006, .026)
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'lean-context-visual-20261007',
                         'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.unicode_minus': False})
    fig = plt.figure(figsize=(14, 10.5), dpi=100, facecolor=PAPER)
    fig.text(.055, .955, 'Learning directed relations with hyperbolic cones', fontsize=27, fontweight='bold', color=NAVY)
    fig.text(.055, .918, 'AddAction  →  AddSemigroupAction  →  VAdd', fontsize=17, color=BLUE)
    fig.text(.055, .883, 'Held pair on seen nodes · x along A, y off A (ball units)', fontsize=16, color=MUTED)
    positions = [(.067, .57, .385, .245), (.555, .57, .385, .245),
                 (.067, .17, .385, .245)]
    axes = [fig.add_axes(p) for p in positions]
    for x, title in ((.055, 'a   Before fitting  ·  update 0'), (.543, 'b   After fitting  ·  update 1000')):
        fig.text(x, .835, title, fontsize=19, fontweight='bold', color=NAVY)
    fig.text(.055, .445, 'c   Final detail · same points', fontsize=19, fontweight='bold', color=NAVY)
    fig.text(.543, .445, 'd   Whole D0 catalogue', fontsize=19, fontweight='bold', color=NAVY)
    b = draw_fold(axes[0], before, source, via, result, coordinates['K'], common_x, common_y, 'before', show_xlabel=False)
    a = draw_fold(axes[1], after, source, via, result, coordinates['K'], common_x, common_y, 'after', inset=(detail_x, detail_y), show_xlabel=False)
    c = draw_fold(axes[2], after, source, via, result, coordinates['K'], detail_x, detail_y, 'detail', show_xlabel=False)
    for x, record in ((.055, b), (.543, a)):
        comparison = '≤' if record['decision'] == 'inside' else '>'
        accent = TEAL if record['decision'] == 'inside' else AMBER
        fig.text(x, .52, f"B angle {record['angle_degrees']:.1f}° {comparison} cone {record['aperture_degrees']:.1f}° · {record['decision'].upper()}",
                 fontsize=14.5, fontweight='bold', color=accent)
        fig.text(x, .492, f"{record['visible_nodes']} of {n} nodes visible ({record['clipped_nodes']} clipped)",
                 fontsize=15, color=MUTED)
    zoom = (common_y[1] - common_y[0]) / (detail_y[1] - detail_y[0])
    fig.text(.055, .105, f"{zoom:.2f}× zoom of box in b · {c['visible_nodes']}/{n} visible",
             fontsize=15, color=NAVY)
    fig.text(.055, .081, f"{c['clipped_nodes']} clipped from detail; catalogue totals unchanged.",
             fontsize=15, color=MUTED)
    ax = fig.add_axes((.555, .165, .385, .25))
    ax.axis('off')
    rows = [(.80, BLUE, f"{positive['inside']:,} / {positive['total']:,}", 'checked arrows inside'),
            (.55, RED, f"{negative['total']-negative['inside']:,} / {negative['total']:,}", 'reversals rejected'),
            (.30, AMBER, f"{other['inside']:,} / {other['total']:,}", 'unlisted pairs inside')]
    for y, color, number, label in rows:
        ax.text(.02, y, number, transform=ax.transAxes, fontsize=28,
                fontweight='bold', color=color, va='center')
        ax.text(.02, y-.087, label, transform=ax.transAxes, fontsize=16, color=NAVY, va='center')
        if y > .31:
            ax.plot([.02, .96], [y-.17, y-.17], transform=ax.transAxes, color=GRID, linewidth=1)
    ax.text(.02, .035, f"Catalogue precision {positive['inside']/(positive['inside']+other['inside']):.1%}",
            transform=ax.transAxes, fontsize=16, color=NAVY)
    fig.text(.055, .038, 'Unlisted pairs are not proven false. Seen-node fitting is not a fresh-interface test.',
             fontsize=15, color=MUTED)
    files = save_pair(fig, folder, 'hyperbolic-overview', dpi)
    plt.close(fig)
    return {'files': files, 'source': names[source], 'via': names[via], 'result': names[result],
            'before': b, 'after': a, 'detail': c,
            'common_domains': {'x': list(common_x), 'y': list(common_y)},
            'inset_domains': {'x': list(detail_x), 'y': list(detail_y)},
            'linear_detail_scale_factor': zoom,
            'D0_counts': counted['D0']['counts']}


def static_fallback(state, dataset, source, via, result, k, folder, name, xlim, ylim):
    fig = plt.figure(figsize=(6.0, 3.3), dpi=100, facecolor=PAPER)
    ax = fig.add_axes((.15, .20, .80, .73))
    record = draw_fold(ax, state, source, via, result, k, xlim, ylim,
                       'before' if state['update'] == 0 else 'after', short=True)
    buffer = io.StringIO()
    fig.savefig(buffer, format='svg', facecolor=PAPER,
                metadata={'Creator': 'Lean Context Search saved-state figure builder', 'Date': None})
    plt.close(fig)
    svg = buffer.getvalue()
    svg = svg[svg.index('<svg '):svg.rindex('</svg>') + 6]
    # Both fallbacks are embedded in one HTML document. Matplotlib's generated
    # element IDs repeat across figures, so namespace every internal reference.
    internal_ids = re.findall(r'\bid="([^"]+)"', svg)
    for internal_id in internal_ids:
        qualified = name + '-' + internal_id
        svg = svg.replace(f'id="{internal_id}"', f'id="{qualified}"')
        svg = svg.replace(f'url(#{internal_id})', f'url(#{qualified})')
        svg = svg.replace(f'="#{internal_id}"', f'="#{qualified}"')
    svg = svg.replace('<svg ', f'<svg id="{name}" class="cone-plot" role="img" aria-label="D0 update {state["update"]} measured common-extent cone" ', 1)
    path = folder / (name + '.svg')
    path.write_text(svg + '\n')
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': sha(path), 'readout': record}


def matrix(ax, display, counted, learned):
    packet = display['datasets']['D0']
    n = len(packet['nodes'])
    order = packet['order']
    position = {node: index for index, node in enumerate(order)}
    labels = counted['D0']['labels']
    inside = counted['D0']['inside']
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f7f8')
    for i in range(n):
        p = position[i]
        ax.add_patch(Rectangle((p, p), 1, 1, facecolor='#d7dee3', linewidth=0))
    cells = {'positive': 0, 'negative': 0, 'extra': 0}
    for s in range(n):
        for t in range(n):
            if s == t or (learned and not inside[s, t]) or (not learned and labels[s, t] != 1):
                continue
            category = 'positive' if labels[s, t] == 1 else 'negative' if labels[s, t] == 2 else 'extra'
            color = {'positive': BLUE, 'negative': RED, 'extra': AMBER}[category]
            ax.add_patch(Rectangle((position[t], position[s]), 1, 1,
                                   facecolor=color, linewidth=0))
            cells[category] += 1
    for boundary in packet['boundaries'][1:-1]:
        ax.axhline(boundary, color='#b6c3cd', linewidth=.8)
        ax.axvline(boundary, color='#b6c3cd', linewidth=.8)
    ticks = [0, 45, 90, 135, 180]
    ax.set_xticks([x+.5 for x in ticks], [str(x+1) for x in ticks], fontsize=12, color=MUTED)
    ax.set_yticks([x+.5 for x in ticks], [str(x+1) for x in ticks], fontsize=12, color=MUTED)
    ax.tick_params(length=0)
    ax.set_xlabel('Result B · grouped index', fontsize=12, color=NAVY)
    ax.set_ylabel('Source A · same order', fontsize=12, color=NAVY)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    return cells


def fidelity(display, counted, folder, dpi):
    d0 = counted['D0']['counts']
    d1 = counted['D1']['counts']
    pos, neg, other = (d0[str(code)] for code in (1, 2, 0))
    p1, n1, u1 = (d1[str(code)] for code in (1, 2, 0))
    n = counted['D0']['node_count']
    fig = plt.figure(figsize=(14, 8.6), dpi=100, facecolor=PAPER)
    fig.text(.055, .956, 'Checked relation and fitted containment', fontsize=27,
             fontweight='bold', color=NAVY)
    fig.text(.055, .919, f'D0 · {n} nodes · {n*(n-1):,} directed pairs · update {counted["D0"]["update"]}',
             fontsize=14, color=MUTED)
    ax1 = fig.add_axes((.08, .235, .38, .58))
    ax2 = fig.add_axes((.55, .235, .38, .58))
    known = matrix(ax1, display, counted, False)
    learned = matrix(ax2, display, counted, True)
    fig.text(.075, .842, 'a   Checked ancestry', fontsize=20, fontweight='bold', color=NAVY)
    fig.text(.545, .842, 'b   Fitted containment', fontsize=20, fontweight='bold', color=NAVY)
    fig.legend(handles=[Patch(facecolor=BLUE, label='checked positive'),
                        Patch(facecolor=AMBER, label='unlisted catalogue pair contained'),
                        Patch(facecolor='#d7dee3', label='diagonal excluded')],
               loc='lower left', bbox_to_anchor=(.06, .13), ncol=3,
               frameon=False, fontsize=13, labelcolor=NAVY)
    fig.text(.075, .097, f"D0  {pos['inside']}/{pos['total']} positives · {neg['total']-neg['inside']}/{neg['total']} reversals rejected · {other['inside']}/{other['total']:,} unlisted inside",
             fontsize=13, color=NAVY)
    fig.text(.075, .069, f"D1  {p1['inside']}/{p1['total']} positives · {n1['total']-n1['inside']}/{n1['total']} countermodels rejected · {u1['inside']}/{u1['total']} unknown inside",
             fontsize=13, color=NAVY)
    fig.text(.075, .032, 'Amber D0 pairs are unlisted, not proved false. D1 unknowns are unjudged.',
             fontsize=13, color=MUTED)
    files = save_pair(fig, folder, 'relation-fidelity', dpi)
    plt.close(fig)
    assert known == {'positive': 212, 'negative': 0, 'extra': 0}
    assert learned == {'positive': 212, 'negative': 0, 'extra': 601}
    return {'files': files, 'recorded_cells': known, 'learned_cells': learned,
            'D0_counts': counted['D0']['counts'], 'D1_counts': counted['D1']['counts']}


def build(root, output_dir, dpi):
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = json.loads((root/'studies/cones/data/coordinates.json').read_text())
    display = json.loads((root/'studies/cones/data/display_data.json').read_text())
    metrics = json.loads((root/'results/structural_metrics.json').read_text())
    assert coordinates['K'] == display['K'] == .1
    assert coordinates['head_dimension'] == display['dimension'] == 128
    assert coordinates['seed'] == display['seed'] == 20261001
    counted = label_counts(coordinates, display, metrics)
    main = overview(coordinates, display, counted, output_dir, dpi)
    separate = fidelity(display, counted, output_dir, dpi)
    names = [x['name'] for x in display['datasets']['D0']['nodes']]
    s, v, t = (names.index(x) for x in ('AddAction', 'AddSemigroupAction', 'VAdd'))
    common = main['common_domains']
    fallbacks = {
        'before': static_fallback(coordinates['datasets']['D0']['states'][0], 'D0', s, v, t,
                                  coordinates['K'], output_dir, 'before-cone', common['x'], common['y']),
        'after': static_fallback(coordinates['datasets']['D0']['states'][-1], 'D0', s, v, t,
                                 coordinates['K'], output_dir, 'after-cone', common['x'], common['y']),
    }
    return {'status': 'passed', 'root': str(root.resolve()), 'output_dir': str(output_dir.resolve()),
            'input_hashes': {x: sha(root/x) for x in ('studies/cones/data/coordinates.json',
                             'studies/cones/data/display_data.json', 'results/structural_metrics.json')},
            'overview': main, 'fidelity': separate, 'static_fallbacks': fallbacks,
            'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
            'png_dpi': dpi, 'training_executed': False, 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--png-dpi', type=int, default=200)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root/'site/figures'
    if args.png_dpi < 150 or args.png_dpi > 400:
        parser.error('PNG DPI must be between 150 and 400')
    result = build(root, output, args.png_dpi)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
