#!/usr/bin/env python3
"""Generate favicon.svg, favicon.png (32), and apple-touch-icon (180)."""

from pathlib import Path
from PIL import Image, ImageDraw
import math

PUB = Path(__file__).resolve().parent.parent / "public"

BG  = (10, 10, 15)
AC  = (0, 212, 170)
TX  = (232, 232, 240)


def draw_dna_icon(img, size):
    """Stylized double-helix mark scaled to `size`."""
    d = ImageDraw.Draw(img, "RGBA")
    # Rounded-square background
    r = size // 5
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG)

    # Helix inside — fewer turns, thinner strands, generous padding
    cx = size / 2
    cy = size / 2
    h = size * 0.58
    rad = size * 0.17
    turns = 1.5
    segs = 64
    pa, pb = [], []
    for i in range(segs + 1):
        f = i / segs
        y = cy - h / 2 + f * h
        th = f * turns * 2 * math.pi
        pa.append((cx + math.cos(th) * rad, y))
        pb.append((cx + math.cos(th + math.pi) * rad, y))

    # Rungs — sparse, only at crossover points
    rung_w = max(1, size // 28)
    for i in range(0, segs + 1, 8):
        d.line([pa[i], pb[i]], fill=(*AC, 110), width=rung_w)

    # Strands — thinner, front/back fade
    strand_w = max(2, size // 16)
    for pts, col in [(pa, AC), (pb, TX)]:
        for i in range(len(pts) - 1):
            f = i / segs
            th = f * turns * 2 * math.pi
            phase = 0 if col is AC else math.pi
            depth = (math.sin(th + phase) + 1) / 2  # 0 back → 1 front
            alpha = int(120 + 135 * depth)
            w = strand_w if depth > 0.5 else max(1, strand_w - 1)
            d.line([pts[i], pts[i + 1]], fill=(*col, alpha), width=w)


# 32x32 PNG favicon
img32 = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
draw_dna_icon(img32, 32)
img32.save(PUB / "favicon-32.png", "PNG", optimize=True)
print(f"Wrote {PUB / 'favicon-32.png'}")

# 180x180 apple-touch-icon
img180 = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
draw_dna_icon(img180, 180)
img180.save(PUB / "apple-touch-icon.png", "PNG", optimize=True)
print(f"Wrote {PUB / 'apple-touch-icon.png'}")

# 192 and 512 for PWA manifest
for sz in (192, 512):
    im = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    draw_dna_icon(im, sz)
    im.save(PUB / f"icon-{sz}.png", "PNG", optimize=True)
    print(f"Wrote {PUB / f'icon-{sz}.png'}")
