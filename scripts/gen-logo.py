# Generates the ppforest2 hex logo (scripts/logo.svg): an aerial forest whose tree
# crowns are observations coloured by class (green / gold / orange), carved into
# oblique decision regions by a projection-pursuit tree — a depth-2 partition
# (root split + one child split) drawn as two cream boundary lines. Each leaf is a
# convex region, lightly tinted its class colour. Framed by the PPforest-lineage
# orange hex, wordmark at the top.
# Output is Inkscape-friendly: layers (Canopy / Frame / Wordmark), grouped regions,
# one group per tree, and a boundaries group. Crown/region colours are driven by CSS
# classes in a <style> block, named by group (g1/g2/g3), not by colour — so recolouring
# never makes a class name lie. Recolour in Inkscape via Object > Selectors and CSS
# (pick e.g. `.g3-b`, change the fill) or edit the default hex values in the GROUPS
# table below and regenerate.
# Regenerate + rasterise the shipped PNG with:
#   python3 scripts/gen-logo.py
#   rsvg-convert -w 520 -h 600 scripts/logo.svg -o bindings/R/man/figures/logo.png
import math, os, random

W, H = 520, 600
HEXPTS = [(260, 12), (509, 156), (509, 444), (260, 588), (11, 444), (11, 156)]
HEX = "M" + " L".join(f"{x},{y}" for x, y in HEXPTS) + " Z"
KEY = "M260,34 L490,167 L490,433 L260,566 L30,433 L30,167 Z"
FRAME = "#e35e28"
CREAM = "#f5f1e6"
GROUND = "#13251a"

# Default fills per group, as (dark, base, light, tint). Group ids are colour-agnostic
# (g1/g2/g3); the tuple values are just the starting colours. -b is the main tone;
# -d/-l shade the foliage; -t is the region wash. Emitted as CSS classes `.<gid>-<d|b|l|t>`.
GROUPS = [("g1", ("#123d24", "#1f5e37", "#3f8a56", "#2b7043")),   # default green
          ("g2", ("#b07d18", "#e0a52c", "#f2ce6b", "#e0a52c")),   # default gold
          ("g3", ("#b0531a", "#ec7f31", "#f6b06b", "#ec7f31"))]   # default orange

def style_block():
    rows = ['/* ===== Group colours — edit the fills here (or via Inkscape > Object >',
            '   Selectors and CSS). Classes are named by group, not colour:',
            '   g1/g2/g3 = the three groups; -b main, -d dark, -l light, -t region tint ===== */']
    for gid, pal in GROUPS:
        rows.append(f'.{gid}-d{{fill:{pal[0]}}} .{gid}-b{{fill:{pal[1]}}} '
                    f'.{gid}-l{{fill:{pal[2]}}} .{gid}-t{{fill:{pal[3]}}}')
    return "\n".join(rows)

# oblique splits, each a line through two passthrough points
L1P = ((20, 300), (500, 210))   # root split: TOP (green) vs BOTTOM
L2P = ((290, 255), (210, 588))  # bottom split: gold (left) vs orange (right)

def line_side(P):
    (x1, y1), (x2, y2) = P
    dx, dy = x2 - x1, y2 - y1
    return lambda x, y: dx * (y - y1) - dy * (x - x1)

side1, side2 = line_side(L1P), line_side(L2P)

def group(x, y):
    # returns the colour-agnostic group id for a point's region
    # sign conventions (checked): side1<0 = TOP; below, side2>0 = LEFT
    if side1(x, y) < 0:
        return "g1"
    return "g2" if side2(x, y) > 0 else "g3"

def clip(poly, fn):
    # Sutherland-Hodgman half-plane clip; keeps vertices with fn<=0
    res = []
    n = len(poly)
    for i in range(n):
        cx, cy = poly[i]
        nx, ny = poly[(i + 1) % n]
        sc, sn = fn(cx, cy), fn(nx, ny)
        if sc <= 0:
            res.append((cx, cy))
        if (sc <= 0) != (sn <= 0):
            t = sc / (sc - sn)
            res.append((cx + t * (nx - cx), cy + t * (ny - cy)))
    return res

def poly_d(p):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p) + " Z"

def intersect(P, Q):
    (x1, y1), (x2, y2) = P
    (x3, y3), (x4, y4) = Q
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

def inhex(px, py):
    inside = False
    n = len(HEXPTS)
    j = n - 1
    for i in range(n):
        xi, yi = HEXPTS[i]
        xj, yj = HEXPTS[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def crown(cx, cy, R, gname):
    # leafy crown = cluster of foliage puffs, toned outer(dark)->inner(light) for soft
    # depth; each puff carries the group's shade class so colours stay editable
    random.seed(int(cx * 7 + cy * 13 + R))
    puffs = []
    for _ in range(int(R * 0.55)):
        a = random.uniform(0, 2 * math.pi)
        d = R * (random.random() ** 0.5) * 0.92
        pr = R * random.uniform(0.22, 0.42)
        t = d / R
        shade = "d" if t > 0.68 else ("b" if t > 0.30 else "l")
        puffs.append((cx + d * math.cos(a), cy + d * math.sin(a) * 0.9, pr, shade, t))
    puffs.sort(key=lambda p: -p[4])
    out = [f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{pr:.1f}" class="{gname}-{shade}"/>'
           for (px, py, pr, shade, _t) in puffs]
    out.append(f'<circle cx="{cx-R*0.24:.1f}" cy="{cy-R*0.26:.1f}" r="{R*0.18:.1f}" class="{gname}-l" opacity="0.85"/>')
    return "".join(out)

def draw_split(P, other, up):
    # a child split lives only inside its parent region: start at the parent
    # boundary intersection and extend into the child (up=toward the top)
    i0 = intersect(P, other)
    (x1, y1), (x2, y2) = P
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    if (uy < 0) != up:
        ux, uy = -ux, -uy
    return (i0[0], i0[1], i0[0] + ux * 430, i0[1] + uy * 430)

# convex leaf regions (nested half-plane clips)
TOP = clip(HEXPTS, side1)
BOTTOM = clip(HEXPTS, lambda x, y: -side1(x, y))
BL = clip(BOTTOM, lambda x, y: -side2(x, y))
BR = clip(BOTTOM, side2)
REGIONS = [("g1", TOP), ("g2", BL), ("g3", BR)]

# observation crowns on a jittered grid, coloured by the region they fall in
random.seed(9)
trees = []
gy, row = 90, 0
while gy < 540:
    gx = 80 + (48 if row % 2 else 0)
    while gx < 470:
        x = gx + random.uniform(-16, 16)
        y = gy + random.uniform(-16, 16)
        if inhex(x, y):
            trees.append((x, y, random.uniform(40, 60)))
        gx += 96
    gy += 88
    row += 1
trees.sort(key=lambda t: t[1])  # paint back-to-front

l1y = lambda x: 300 - 90 * (x - 20) / 480
l2 = draw_split(L2P, L1P, up=False)

INK = ('xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
       'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd"')
# Soft drop shadow built from classic primitives (feDropShadow is unsupported in
# older Inkscape, which would blank the filtered crowns). Equivalent to
# feDropShadow(dx=0, dy=3, stdDeviation=5, flood black, opacity 0.32).
DEFS = (f'<clipPath id="hx" clipPathUnits="userSpaceOnUse"><path d="{HEX}"/></clipPath>'
        '<filter id="soft" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur in="SourceAlpha" stdDeviation="5" result="blur"/>'
        '<feOffset in="blur" dx="0" dy="3" result="off"/>'
        '<feComponentTransfer in="off" result="shadow"><feFuncA type="linear" slope="0.32"/></feComponentTransfer>'
        '<feMerge><feMergeNode in="shadow"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
# Inkscape document settings so the file opens natively (layers panel, page, units)
NAMEDVIEW = ('<sodipodi:namedview id="namedview" pagecolor="#ffffff" bordercolor="#666666" '
             'borderopacity="1" inkscape:pageopacity="0" inkscape:pageshadow="2" '
             'inkscape:document-units="px" showgrid="false" units="px" '
             'inkscape:current-layer="layer-canopy"/>')

STYLE = f'<style type="text/css" id="palette"><![CDATA[\n{style_block()}\n]]></style>'

s = [f'<svg xmlns="http://www.w3.org/2000/svg" {INK} viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="ppforest2 hex logo">',
     f'<defs>{DEFS}</defs>',
     NAMEDVIEW,
     STYLE]
# --- Canopy layer (clipped to hex): ground, decision-region tints, crowns, boundaries ---
s.append('<g inkscape:groupmode="layer" inkscape:label="Canopy" id="layer-canopy" clip-path="url(#hx)">')
s.append(f'<rect id="background" width="{W}" height="{H}" fill="{GROUND}"/>')
s.append('<g inkscape:label="regions" id="regions">')
for gid, poly in REGIONS:
    op = 0.17 if gid == "g3" else 0.15
    s.append(f'<path inkscape:label="region-{gid}" id="region-{gid}" class="{gid}-t" d="{poly_d(poly)}" opacity="{op}"/>')
s.append('</g>')
s.append('<g id="crowns" filter="url(#soft)">')
for i, (x, y, R) in enumerate(trees):
    gid = group(x, y)
    s.append(f'<g inkscape:label="tree-{i+1} ({gid})" id="tree-{i+1}">{crown(x, y, R, gid)}</g>')
s.append('</g>')
s.append('<g inkscape:label="boundaries" id="boundaries">')
s.append(f'<line id="split-root" x1="-40" y1="{l1y(-40):.1f}" x2="560" y2="{l1y(560):.1f}" stroke="{CREAM}" stroke-width="4.5" stroke-linecap="round" opacity="0.9"/>')
s.append(f'<line id="split-child" x1="{l2[0]:.1f}" y1="{l2[1]:.1f}" x2="{l2[2]:.1f}" y2="{l2[3]:.1f}" stroke="{CREAM}" stroke-width="4.5" stroke-linecap="round" opacity="0.9"/>')
s.append('</g></g>')
# --- Frame layer: hex border + inner keyline ---
s.append('<g inkscape:groupmode="layer" inkscape:label="Frame" id="layer-frame">')
s.append(f'<path id="hex-border" d="{HEX}" fill="none" stroke="{FRAME}" stroke-width="16" stroke-linejoin="round"/>')
s.append(f'<path id="hex-keyline" d="{KEY}" fill="none" stroke="{FRAME}" stroke-width="2.2" opacity="0.35"/>')
s.append('</g>')
# --- Wordmark layer: editable text, placed at the top ---
s.append('<g inkscape:groupmode="layer" inkscape:label="Wordmark" id="layer-text">')
s.append(f'<text id="wordmark" x="260" y="135" text-anchor="middle" font-family="\'Helvetica Neue\', Arial, sans-serif" '
         f'font-weight="800" font-size="50" letter-spacing="0.5" fill="{CREAM}" stroke="{FRAME}" stroke-width="7" '
         f'stroke-linejoin="round" paint-order="stroke">ppforest2</text>')
s.append('</g>')
s.append('</svg>')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.svg")
open(out, "w").write("\n".join(s))
print(f"wrote {out}  (crowns={len(trees)})")
