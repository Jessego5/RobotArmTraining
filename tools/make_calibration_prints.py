#!/usr/bin/env python3
"""Print-ready calibration sheets at fixed, true size, as PDFs whose page is exactly the paper.

- checkerboard_letter.pdf: US Letter, portrait. 7 x 10 squares of 20 mm (9 x 6 inner
  corners, the calibration scripts' default), for the wrist cameras and the overhead lens.
- tag_sheet_tabloid.pdf: US Tabloid (11 x 17 in), landscape. Six tag36h11 tags of 90 mm
  with 25 mm gaps and the numbered touch corners, for the overhead camera's pose.

Print at "Actual size" / 100%, never "Fit". Each page has a card-check bar 85.60 mm
long, the ISO ID-1 width of bank and ID cards: if a card's long edge spans it exactly,
the print is true to size, no ruler needed. Then use the printed sizes as given:

    python tools/calibrate_wrist_camera.py ... --square-mm 20
    python tools/calibrate_overhead_camera.py ... --tag-mm 90 --gap-mm 25

    python tools/make_calibration_prints.py --output outputs/print
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calibrate_overhead_camera import Sheet

DPI = 300
CARD_MM = (85.60, 53.98)  # ISO/IEC 7810 ID-1
LETTER_MM = (215.9, 279.4)
TABLOID_MM = (431.8, 279.4)  # landscape


def px(mm):
    return int(round(mm / 25.4 * DPI))


def text(page, words, x, y, size_mm, weight_mm=.35):
    cv2.putText(page, words, (x, y), cv2.FONT_HERSHEY_SIMPLEX, px(size_mm) / 22, 0, max(1, px(weight_mm)))


def card_bar(page, x, y, below=False):
    """A bar exactly as long as a bank or ID card (85.60 mm), with end ticks: lay the card's
    long edge on it; the ends match when the print is true to size."""
    w, tick = px(CARD_MM[0]), px(4)
    cv2.line(page, (x, y), (x + w, y), 0, px(.4))
    for end in (x, x + w):
        cv2.line(page, (end, y - tick), (end, y + tick), 0, px(.4))
    if below:  # narrow page: the note under the bar
        text(page, 'card check: a bank or ID card\'s long edge', x, y + px(9), 3)
        text(page, 'spans exactly this bar', x, y + px(14), 3)
    else:
        text(page, 'card check: a bank or ID card\'s long edge spans exactly this bar', x + w + px(4), y + px(1.2), 3)


def save_pdf(page, path):
    from PIL import Image
    Image.fromarray(page).convert('L').save(path, 'PDF', resolution=DPI)
    return path


def checkerboard(path, inner=(9, 6), square_mm=20.):
    """Letter portrait: (inner+1) squares, the long side down the page."""
    page = np.full((px(LETTER_MM[1]), px(LETTER_MM[0])), 255, np.uint8)
    across, down = inner[1] + 1, inner[0] + 1  # 7 across, 10 down
    sq = px(square_mm)
    x0 = (page.shape[1] - across * sq) // 2
    y0 = px(14)
    for r in range(down):
        for c in range(across):
            if (r + c) % 2 == 0:
                page[y0 + r * sq:y0 + (r + 1) * sq, x0 + c * sq:x0 + (c + 1) * sq] = 0
    y = y0 + down * sq + px(10)
    text(page, f'{across} x {down} squares of {square_mm:g} mm ({inner[0]} x {inner[1]} inner corners)', x0, y, 3.2)
    text(page, 'print at Actual size / 100% on US Letter', x0, y + px(6), 3.2)
    card_bar(page, x0, y + px(16), below=True)
    return save_pdf(page, path)


def tag_sheet(path, tag_mm=90., gap_mm=25.):
    """Tabloid landscape: the six-tag grid of calibrate_overhead_camera.py at true size."""
    sheet = Sheet(tag_mm / 1000, gap_mm / 1000)
    width_mm, height_mm = sheet.extent() * 1000
    image, _ = sheet.render(int(round(px(width_mm) * 20 / 18)))  # render() keeps a 1/20 margin per side
    page = np.full((px(TABLOID_MM[1]), px(TABLOID_MM[0])), 255, np.uint8)
    top = px(8)
    left = (page.shape[1] - image.shape[1]) // 2
    page[top:top + image.shape[0], left:left + image.shape[1]] = image[..., 0] if image.ndim == 3 else image
    scale = image.shape[1] / 800
    for k, (x, y) in enumerate(sheet.corner_pixels * scale):
        dx = -px(12) if x < image.shape[1] / 2 else px(3)
        dy = -px(3) if y < image.shape[0] / 2 else px(12)
        cv2.putText(page, str(k + 1), (int(left + x + dx), int(top + y + dy)), cv2.FONT_HERSHEY_SIMPLEX,
                    px(9) / 22, 0, px(1.2))
    y = top + image.shape[0] + px(6)
    x0 = int(left + sheet.corner_pixels[0][0] * scale)
    text(page, f'tag36h11 ids 0-5  |  tags {tag_mm:g} mm, gaps {gap_mm:g} mm  |  touch corners 1, 2, 3, 4  |  '
         'print at Actual size on 11 x 17 in', x0, y + px(4), 3.2)
    card_bar(page, x0, y + px(14))
    return save_pdf(page, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('outputs/print'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (checkerboard(args.output / 'checkerboard_letter.pdf'), tag_sheet(args.output / 'tag_sheet_tabloid.pdf')):
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
