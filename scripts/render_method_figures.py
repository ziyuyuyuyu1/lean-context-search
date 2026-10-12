"""Author the four release diagrams as native, accessible SVG.

The SVG path needs only Python's standard library. Optional PNG fallbacks use
resvg-py 0.2.4: python scripts/render_method_figures.py --png
The conceptual cone is labelled schematic; its summary counts are read from
results/structural_metrics.json. Existing measured figures are never rewritten.
"""
import argparse
from math import cos, sin, atan2, hypot
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'site/figures'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
BG = '#f7f8fa'
WHITE = '#ffffff'
INK = '#203248'
MUTED = '#576777'
BORDER = '#dce3e9'
BLUE = '#235f9b'
GREEN = '#317b6c'
AMBER = '#b86c13'
SOFT = '#edf1f5'
CONE = '#e6f1ed'
FONT = 'Arial, Helvetica, DejaVu Sans, sans-serif'
MONO = 'DejaVu Sans Mono, monospace'


def fmt(value):
    return f'{value:.3f}'.rstrip('0').rstrip('.') if isinstance(value, float) else str(value)


class Diagram:
    def __init__(self, name, title, description, height=450):
        self.name, self.height = name, height
        self.root = ET.Element(f'{{{NS}}}svg', {
            'viewBox': f'0 0 1200 {height}', 'width': '1200', 'height': str(height),
            'role': 'img', 'aria-labelledby': 'title description',
            'font-family': FONT, 'fill': INK})
        self.add('title', {'id': 'title'}, title)
        self.add('desc', {'id': 'description'}, description)
        defs = self.add('defs')
        for color in (INK, MUTED, BLUE, GREEN, AMBER):
            marker = ET.SubElement(defs, f'{{{NS}}}marker', {
                'id': self.marker(color), 'viewBox': '0 0 10 10', 'refX': '9',
                'refY': '5', 'markerWidth': '10', 'markerHeight': '10',
                'markerUnits': 'userSpaceOnUse', 'orient': 'auto-start-reverse'})
            ET.SubElement(marker, f'{{{NS}}}path', {'d': 'M 1 1 L 9 5 L 1 9 Z', 'fill': color})
        self.rect(0, 0, 1200, height, fill=BG, stroke='none', radius=18)

    @staticmethod
    def marker(color):
        return 'arrow-' + color.lstrip('#')

    def add(self, tag, attrs=None, text=None):
        element = ET.SubElement(self.root, f'{{{NS}}}{tag}',
                                {k: fmt(v) for k, v in (attrs or {}).items()})
        if text is not None:
            element.text = str(text)
        return element

    def rect(self, x, y, w, h, fill=WHITE, stroke=BORDER, radius=12, width=1.6):
        return self.add('rect', {'x': x, 'y': y, 'width': w, 'height': h,
                                'rx': radius, 'fill': fill, 'stroke': stroke, 'stroke-width': width})

    def text(self, x, y, text, size=26, color=INK, weight=400, anchor='middle', mono=False):
        attrs = {'x': x, 'y': y, 'font-size': size, 'font-weight': weight,
                 'fill': color, 'text-anchor': anchor}
        if mono:
            attrs['font-family'] = MONO
        return self.add('text', attrs, text)

    def path(self, points, color=MUTED, width=2.5, arrow=False):
        """A route with modest rounded corners and a compact arrowhead."""
        d = [f'M {fmt(points[0][0])} {fmt(points[0][1])}']
        for i in range(1, len(points)-1):
            a, b, c = points[i-1], points[i], points[i+1]
            incoming = hypot(b[0]-a[0], b[1]-a[1])
            outgoing = hypot(c[0]-b[0], c[1]-b[1])
            radius = min(10, incoming/2, outgoing/2)
            before = (b[0]+(a[0]-b[0])*radius/incoming, b[1]+(a[1]-b[1])*radius/incoming)
            after = (b[0]+(c[0]-b[0])*radius/outgoing, b[1]+(c[1]-b[1])*radius/outgoing)
            d.append(f'L {fmt(before[0])} {fmt(before[1])} Q {fmt(b[0])} {fmt(b[1])} {fmt(after[0])} {fmt(after[1])}')
        d.append(f'L {fmt(points[-1][0])} {fmt(points[-1][1])}')
        attrs = {'d': ' '.join(d), 'fill': 'none', 'stroke': color,
                 'stroke-width': width, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}
        if arrow:
            attrs['marker-end'] = f'url(#{self.marker(color)})'
        return self.add('path', attrs)

    def circle(self, x, y, r, fill=WHITE, stroke=BORDER, width=1.6):
        return self.add('circle', {'cx': x, 'cy': y, 'r': r, 'fill': fill,
                                   'stroke': stroke, 'stroke-width': width})

    def document(self, x, y, w, h):
        self.add('path', {'d': f'M {x+12} {y} H {x+w-28} L {x+w} {y+28} V {y+h-12} Q {x+w} {y+h} {x+w-12} {y+h} H {x+12} Q {x} {y+h} {x} {y+h-12} V {y+12} Q {x} {y} {x+12} {y} Z',
                          'fill': WHITE, 'stroke': BORDER, 'stroke-width': 1.6})
        self.path([(x+w-28, y), (x+w-28, y+28), (x+w, y+28)], BORDER, 1.6)

    def frozen(self, cx, y):
        x = cx-50
        self.add('path', {'d': f'M {x+4} {y+11} V {y+5} A 6 6 0 0 1 {x+16} {y+5} V {y+11}',
                          'fill': 'none', 'stroke': MUTED, 'stroke-width': 2})
        self.rect(x, y+10, 20, 17, fill=SOFT, stroke=MUTED, radius=3, width=1.5)
        self.text(x+31, y+26, 'Frozen', 24, MUTED, anchor='start')

    def save(self, png=False):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        ET.indent(self.root, space='  ')
        path = OUTPUT / (self.name + '.svg')
        path.write_bytes(ET.tostring(self.root, encoding='utf-8', xml_declaration=True) + b'\n')
        if png:
            import resvg_py
            path.with_suffix('.png').write_bytes(resvg_py.svg_to_bytes(
                svg_path=str(path), width=2400, height=self.height*2,
                sans_serif_family='DejaVu Sans', monospace_family='DejaVu Sans Mono'))
        return path


def enrichment():
    d = Diagram('method-1-typed-text', 'Enrich a declaration with checked parent information',
                'Original Monoid text plus Lean-checked parent projections produces an enriched retrieval passage. Arrows run from Monoid to its parents Semigroup and MulOneClass. Token counts are 100 original and 761 enriched. The displayed excerpt is abbreviated.', 450)
    for x, title in ((164, 'Original passage'), (600, 'Checked parents'), (1044, 'Enriched passage')):
        d.text(x, 48, title, 26, weight=600)
    d.document(44, 84, 240, 255)
    d.text(70, 143, 'Monoid', 30, weight=600, anchor='start', mono=True)
    for y, end in ((180, 242), (201, 216), (222, 231)):
        d.path([(70, y), (end, y)], BORDER, 7)
    d.path([(70, 291), (221, 291)], BORDER, 7)
    d.text(329, 227, '+', 36, MUTED)
    d.rect(491, 92, 190, 65)
    d.text(586, 134, 'Monoid', 27, BLUE, 600, mono=True)
    d.path([(586, 158), (586, 194), (469, 194), (469, 235)], GREEN, arrow=True)
    d.path([(586, 194), (720, 194), (720, 235)], GREEN, arrow=True)
    d.rect(363, 244, 212, 64, fill=CONE)
    d.text(469, 284, 'Semigroup', 24, GREEN, 600)
    d.rect(600, 244, 240, 64, fill=CONE)
    d.text(720, 284, 'MulOneClass', 24, GREEN, 600)
    d.text(469, 351, 'mul_assoc', 24, GREEN, mono=True)
    d.text(720, 351, 'one_mul', 24, GREEN, mono=True)
    d.text(720, 387, 'mul_one', 24, GREEN, mono=True)
    d.path([(863, 218), (903, 218)], GREEN, arrow=True)
    d.document(924, 84, 240, 282)
    d.text(950, 143, 'Monoid', 30, weight=600, anchor='start', mono=True)
    for y, end in ((180, 1122), (201, 1096)):
        d.path([(950, y), (end, y)], BORDER, 7)
    d.rect(942, 224, 204, 124, fill=CONE, stroke='none', radius=6)
    for y, text in ((256, 'mul_assoc'), (292, 'one_mul'), (328, 'mul_one')):
        d.text(960, y, text, 24, GREEN, anchor='start', mono=True)
    d.text(164, 409, '100 tokens', 28, MUTED, 600)
    d.text(1044, 409, '761 tokens', 28, GREEN, 600)
    return d


def embedding():
    d = Diagram('method-2-embedding-head', 'Two weighted cosine paths retrieve the top 200',
                'A frozen Qwen3-Embedding-8B encodes the query and all 310579 enriched corpus passages. Query and document vectors feed base cosine at weight 70 percent and a shared 128-dimensional head cosine at weight 30 percent. Their sum scores the full corpus and selects the top 200.', 450)
    d.rect(35, 79, 168, 72)
    d.text(119, 124, 'Query', 28, weight=600)
    d.rect(35, 288, 168, 91)
    d.text(119, 327, '310,579', 28, weight=600)
    d.text(119, 359, 'passages', 24, MUTED)
    d.path([(205, 115), (227, 115), (227, 208), (253, 208)], BLUE, arrow=True)
    d.path([(205, 333), (227, 333), (227, 270), (253, 270)], BLUE, arrow=True)
    d.rect(263, 157, 236, 168)
    d.text(381, 201, 'Qwen3', 30, weight=600)
    d.text(381, 244, 'Embedding-8B', 26, weight=600)
    d.frozen(380, 275)
    d.path([(501, 241), (548, 241)], BLUE)
    d.text(549, 213, 'q, d', 24, MUTED)
    d.circle(550, 241, 4, fill=BLUE, stroke='none')
    d.path([(550, 241), (566, 241), (566, 129), (581, 129)], BLUE, arrow=True)
    d.path([(550, 241), (566, 241), (566, 346), (581, 346)], GREEN, arrow=True)
    d.rect(591, 65, 260, 126)
    d.text(721, 111, 'Base cosine', 27, BLUE, 600)
    d.text(721, 159, '70%', 39, BLUE, 600)
    d.rect(591, 282, 260, 126, fill=CONE)
    d.text(721, 327, '128-D head', 27, GREEN, 600)
    d.text(721, 373, 'Cosine · 30%', 29, GREEN)
    d.path([(853, 129), (896, 129), (896, 229), (923, 229)], BLUE, arrow=True)
    d.path([(853, 346), (896, 346), (896, 258), (923, 258)], GREEN, arrow=True)
    d.circle(951, 244, 26)
    d.text(951, 255, '+', 35, weight=400)
    d.path([(978, 244), (1014, 244)], AMBER, arrow=True)
    d.rect(1025, 185, 140, 123)
    d.text(1095, 224, 'Top', 27, MUTED)
    d.text(1095, 277, '200', 45, AMBER, 600)
    return d


def rank_rows(d, x, y, order, width):
    colors = (BLUE, GREEN, AMBER)
    for row, index in enumerate(order):
        yy = y + row*36
        d.rect(x, yy, 13, 13, fill=colors[index], stroke='none', radius=3)
        d.path([(x+25, yy+6.5), (x+width, yy+6.5)], colors[index], 4)


def reranking():
    d = Diagram('method-3-reranking-head', 'Rerank enriched passages with the frozen Qwen margin',
                'The same 200 candidates from embedding retrieval enter frozen Qwen3-Reranker-8B with the query and enriched passages. The efficient yes-minus-no margin directly orders the candidates, with no added reranker head. The top 100 are returned from the same pool. Colored row motifs are schematic and do not encode measured example ranks. The historical figure filename is retained for link compatibility.', 450)
    d.rect(35, 134, 225, 238)
    d.text(147, 181, '200', 40, BLUE, 600)
    d.text(147, 216, 'candidates', 26)
    rank_rows(d, 73, 240, (0, 1, 2), 146)
    d.text(147, 349, 'Enriched text', 26, GREEN, 600)
    d.path([(262, 241), (292, 241)], BLUE, arrow=True)
    d.rect(323, 45, 218, 64)
    d.text(432, 87, 'Query', 28, weight=600)
    d.path([(432, 111), (432, 148)], BLUE, arrow=True)
    d.rect(302, 157, 260, 168)
    d.text(432, 201, 'Qwen3', 30, weight=600)
    d.text(432, 244, 'Reranker-8B', 28, weight=600)
    d.frozen(431, 275)
    d.path([(564, 241), (616, 241)], BLUE, arrow=True)
    d.rect(626, 164, 272, 154, fill=CONE)
    d.text(762, 207, 'Pair score', 28, GREEN, 600)
    d.text(762, 252, 'yes − no', 36, GREEN, 600)
    d.text(762, 293, 'Efficient margin', 25, GREEN)
    d.path([(900, 241), (971, 241)], AMBER, arrow=True)
    d.rect(981, 146, 185, 206)
    d.text(1073, 189, 'Top 100', 30, AMBER, 600)
    rank_rows(d, 1010, 216, (1, 0, 2), 122)
    d.text(1073, 329, 'Same pool', 25, MUTED)
    d.text(600, 418, 'Score each query–passage pair, then reorder the same 200 candidates', 25, MUTED)
    return d


def cone_boundaries(radius, aperture):
    """Poincare geodesics via a disk translation; schematic parameters only."""
    sides = []
    for sign in (1, -1):
        points = []
        for i in range(101):
            q = min(.9999999, 1-(1-i/100)**3)
            zx, zy = q*cos(aperture), sign*q*sin(aperture)
            den = 1 + 2*radius*zx + radius*radius*q*q
            points.append((((1+2*radius*zx+q*q)*radius+(1-radius*radius)*zx)/den,
                           (1-radius*radius)*zy/den))
        sides.append(points)
    a = atan2(sides[0][-1][1], sides[0][-1][0])
    arc = [(cos(a*(1-2*i/40)), sin(a*(1-2*i/40))) for i in range(1, 41)]
    return sides[0], sides[1], sides[0] + arc + sides[1][::-1]


def cone_study():
    metrics = json.loads((ROOT / 'results/structural_metrics.json').read_text())['D0']
    positive = metrics['train_positive'] + metrics['held_composed_positive']
    negative = metrics['train_strict_reverse'] + metrics['held_strict_reverse']
    d = Diagram('cone-study-structure', 'Checked structure and cone containment',
                'Left: a real checked AddAction to AddSemigroupAction to VAdd supplier path. Right: an explicitly schematic Poincare cone; A supplies an inside result B while X is outside. The bottom reports the saved known-interface fit, not the schematic coordinates: 212 of 212 recorded positives inside and 212 of 212 strict reversals rejected. These checks concern known interfaces, not general search accuracy.', 600)
    d.text(291, 48, 'Checked supplier maps', 26, weight=600)
    d.rect(105, 105, 371, 67)
    d.text(290, 149, 'A · AddAction', 27, BLUE, 600)
    d.path([(290, 174), (290, 220)], GREEN, arrow=True)
    d.rect(64, 230, 452, 67, fill=CONE)
    d.text(290, 274, 'AddSemigroupAction', 27, GREEN, 600)
    d.path([(290, 299), (290, 345)], GREEN, arrow=True)
    d.rect(163, 355, 254, 67, fill=CONE)
    d.text(290, 399, 'B · VAdd', 27, GREEN, 600)
    d.path([(548, 262), (646, 262)], MUTED, arrow=True)
    d.text(906, 48, 'Cone rule · schematic', 26, weight=600)
    cx, cy, scale = 906, 249, 175
    d.circle(cx, cy, scale)
    upper, lower, area = cone_boundaries(.22, .75)
    screen = lambda p: (cx+scale*p[0], cy-scale*p[1])
    d.add('polygon', {'points': ' '.join(f'{fmt(screen(p)[0])},{fmt(screen(p)[1])}' for p in area), 'fill': CONE})
    d.path([screen(p) for p in upper], GREEN, 2.5)
    d.path([screen(p) for p in lower], GREEN, 2.5)
    d.path([(cx+scale*.22, cy), (cx+scale*.96, cy)], BORDER, 1.5)
    for x, y, label, color, label_y in ((cx+scale*.22, cy, 'A', GREEN, cy+39),
                                        (cx+scale*.71, cy-10, 'B', BLUE, cy-33),
                                        (cx+scale*.47, cy+103, 'X', AMBER, cy+144)):
        d.circle(x, y, 8, fill=color, stroke=WHITE, width=2)
        d.text(x, label_y, label, 26, color, 600)
    d.path([(36, 455), (1164, 455)], BORDER, 1.5)
    d.text(38, 495, 'Known-interface fit', 24, MUTED, 600, anchor='start')
    values = (
        (300, f"{metrics['final_positive_inside']} / {positive}", 'Positives inside', GREEN),
        (900, f"{negative-metrics['final_negative_inside']} / {negative}", 'Reversals outside', BLUE),
    )
    for x, value, label, color in values:
        d.text(x, 539, value, 33, color, 600)
        d.text(x, 577, label, 24, MUTED)
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true', help='also export 2x PNG fallbacks with resvg-py')
    parser.add_argument('--figure', choices=('all', 'enrichment', 'embedding', 'reranking', 'cone-study'),
                        default='all', help='render one diagram or all four (default)')
    args = parser.parse_args()
    builders = {'enrichment': enrichment, 'embedding': embedding,
                'reranking': reranking, 'cone-study': cone_study}
    selected = builders.values() if args.figure == 'all' else (builders[args.figure],)
    for builder in selected:
        path = builder().save(png=args.png)
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
