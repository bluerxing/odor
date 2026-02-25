#!/usr/bin/env python3
"""
Generate Figure 3: Composite Empirical Dashboard (from existing PNGs)
=====================================================================
Intelligently crops, resizes, and composites 8 sub-panels into a single
publication-quality figure with unified panel labels and tight layout.

Unlike the old plot_11 which naively pasted 6 images into a 3x2 grid,
this script:
  - Auto-crops whitespace from each sub-image
  - Normalizes each panel within its row to equal height
  - Uses per-row width ratios based on content
  - Produces both PNG (300 dpi) and PDF

Layout (4 rows x 2 cols):
  Row 1:  (A) Odor distribution       | (B) Pathway length
  Row 2:  (C) Transformation heatmap  | (D) Top EC sequences
  Row 3:  (E) EC by source odor       | (F) EC by target odor
  Row 4:  (G) EC function distribution| (H) Transformation flow
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / 'figures'

# Panel definitions: (label, filename, crop_top_frac, crop_bottom_frac)
PANELS_DEF = [
    ('A', '02_odor_distribution.png',           0.00, 0.00),
    ('B', '03_pathway_length.png',              0.00, 0.00),
    ('C', '04_transformation_heatmap.png',      0.05, 0.00),
    ('D', '05_top_ec_sequences.png',            0.05, 0.00),
    ('E', '07a_ec_by_source.png',               0.05, 0.00),
    ('F', '07b_ec_by_target.png',               0.05, 0.00),
    ('G', '08a_ec_function_pie_normalized.png', 0.05, 0.00),
    ('H', '06_transformation_flow.png',         0.05, 0.00),
]

LAYOUT = [
    ['A', 'B'],
    ['C', 'D'],
    ['E', 'F'],
    ['G', 'H'],
]


def auto_crop(img, threshold=248, pad=6):
    """Crop near-white borders."""
    arr = np.array(img.convert('RGB'))
    mask = np.any(arr < threshold, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = arr.shape[:2]
    return img.crop((max(0, cmin - pad), max(0, rmin - pad),
                      min(w, cmax + pad + 1), min(h, rmax + pad + 1)))


def load_panel(fname, crop_top=0.0, crop_bottom=0.0):
    path = FIGURES_DIR / fname
    if not path.exists():
        return None
    img = Image.open(path).convert('RGB')
    w, h = img.size
    t = int(h * crop_top)
    b = h - int(h * crop_bottom)
    if t > 0 or crop_bottom > 0:
        img = img.crop((0, t, w, b))
    return auto_crop(img)


def compose_row_pil(img_left, img_right, label_l, label_r, target_width, gap=30):
    """
    Compose two PIL images into a single row image,
    scaling both to the same height and placing them side by side.
    Returns a new PIL Image.
    """
    from PIL import ImageDraw, ImageFont

    # Determine common height: scale both so they fit in target_width
    wl, hl = img_left.size
    wr, hr = img_right.size
    ar_l = wl / hl  # aspect ratio
    ar_r = wr / hr

    # Allocate width proportional to aspect ratio
    total_ar = ar_l + ar_r
    alloc_l = int((target_width - gap) * ar_l / total_ar)
    alloc_r = target_width - gap - alloc_l

    # Common height from allocation
    common_h_l = int(alloc_l / ar_l)
    common_h_r = int(alloc_r / ar_r)
    common_h = min(common_h_l, common_h_r)

    # Resize
    new_wl = int(common_h * ar_l)
    new_wr = int(common_h * ar_r)
    img_l = img_left.resize((new_wl, common_h), Image.LANCZOS)
    img_r = img_right.resize((new_wr, common_h), Image.LANCZOS)

    # Compose onto white canvas
    total_w = new_wl + gap + new_wr
    canvas = Image.new('RGB', (total_w, common_h), (255, 255, 255))
    canvas.paste(img_l, (0, 0))
    canvas.paste(img_r, (new_wl + gap, 0))

    # Draw panel labels
    draw = ImageDraw.Draw(canvas)
    font_size = max(28, common_h // 18)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    label_pad = 8
    for label, x_offset in [(label_l, label_pad), (label_r, new_wl + gap + label_pad)]:
        text = f'({label})'
        # White background box
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bx, by = x_offset, label_pad
        draw.rectangle([bx - 2, by - 2, bx + tw + 6, by + th + 6],
                       fill=(255, 255, 255, 230))
        draw.text((bx + 2, by + 2), text, fill=(0, 0, 0), font=font)

    return canvas


def build_composite():
    """Build the 8-panel composite figure using PIL for precise pixel control."""
    from PIL import ImageDraw, ImageFont

    # Load all panels
    images = {}
    for label, fname, ct, cb in PANELS_DEF:
        img = load_panel(fname, ct, cb)
        if img is not None:
            images[label] = img
            print(f"  [{label}] {fname:45s} -> {img.size[0]}x{img.size[1]}")
        else:
            print(f"  [{label}] {fname:45s} -> MISSING")

    if len(images) < 6:
        print("ERROR: Need at least 6 panels")
        return

    # Target width for the whole figure (pixels at 300 dpi for ~17 inch width)
    target_w = 5100  # ~17 inches * 300 dpi
    row_gap = 40

    # Compose each row
    row_images = []
    for row_labels in LAYOUT:
        l, r = row_labels
        if l in images and r in images:
            row_img = compose_row_pil(images[l], images[r], l, r, target_w, gap=40)
            row_images.append(row_img)
            print(f"  Row [{l},{r}]: {row_img.size[0]}x{row_img.size[1]}")
        else:
            print(f"  Row [{l},{r}]: SKIPPED (missing panel)")

    if not row_images:
        print("ERROR: No rows composed")
        return

    # Normalize all rows to same width
    max_w = max(r.size[0] for r in row_images)
    normalized_rows = []
    for r in row_images:
        if r.size[0] < max_w:
            # Center on white canvas
            canvas = Image.new('RGB', (max_w, r.size[1]), (255, 255, 255))
            offset = (max_w - r.size[0]) // 2
            canvas.paste(r, (offset, 0))
            normalized_rows.append(canvas)
        else:
            normalized_rows.append(r)

    # Title bar
    title_h = 80
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except (OSError, IOError):
        title_font = ImageFont.load_default()

    title_bar = Image.new('RGB', (max_w, title_h), (255, 255, 255))
    draw_title = ImageDraw.Draw(title_bar)
    title_text = "Empirical Overview of Odor Transformation Rules"
    bbox = draw_title.textbbox((0, 0), title_text, font=title_font)
    tw = bbox[2] - bbox[0]
    draw_title.text(((max_w - tw) // 2, 15), title_text, fill=(0, 0, 0), font=title_font)

    # Stack everything vertically
    total_h = title_h + sum(r.size[1] for r in normalized_rows) + row_gap * (len(normalized_rows) - 1)
    final = Image.new('RGB', (max_w, total_h), (255, 255, 255))

    y = 0
    final.paste(title_bar, (0, y))
    y += title_h

    for i, row_img in enumerate(normalized_rows):
        final.paste(row_img, (0, y))
        y += row_img.size[1]
        if i < len(normalized_rows) - 1:
            y += row_gap

    # Save
    out_png = FIGURES_DIR / 'fig3_composite_dashboard.png'
    out_pdf = FIGURES_DIR / 'fig3_composite_dashboard.pdf'

    final.save(out_png, dpi=(300, 300))

    # For PDF, use matplotlib to wrap the PIL image
    fig, ax = plt.subplots(figsize=(max_w / 300, total_h / 300))
    ax.imshow(np.array(final), interpolation='lanczos')
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
    plt.close(fig)

    print(f"\n  Saved: {out_png}  ({final.size[0]}x{final.size[1]})")
    print(f"  Saved: {out_pdf}")
    return out_png, out_pdf


if __name__ == '__main__':
    print("Generating Figure 3: Composite Empirical Dashboard")
    print("=" * 60)
    build_composite()
    print("\nDone!")
