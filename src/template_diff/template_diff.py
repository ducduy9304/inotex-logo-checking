"""Find missing or added print by comparing each crop with a good reference crop.

For the print-only groups (frontview-sidelight, backview). The reference is the group's first train/good
crop. Each crop is aligned to it (ECC, homography), both are split into print / not print, and
two differences are kept: print missing where the reference has it, and print added near the reference
print (backview only: under frontview sidelight the film's glare is as bright as the print).
Thin differences along the letter edges come from the alignment and are removed by an opening.
Background far from the print is not checked.

Usage: uv run python src/template_diff/template_diff.py [--crops data_crop]
Prints the largest region and the number of missing / added regions of every crop (no pictures).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

GROUPS = ("frontview-sidelight", "backview")
ALIGN_SCALE = 0.25  # alignment is found on a small copy
NEAR_PRINT = 40     # px around the reference print where added print counts; farther is background
OPEN = 15           # px: differences thinner than this are alignment error along the letter edges
MIN_AREA = 2000     # px: smallest region reported as a defect
EDGE_PIECE_AREA = 10000   # px: reference pieces smaller than this ...
EDGE_PIECE_DISTANCE = 80  # px: ... and this close to the crop border are dropped (letters are 28000+ px)
FILM_SIZE = 41     # px on the 1/4 copy (~160 px full size, wider than a letter stroke): film brightness around the print


def print_mask(gray, bright_print):
    # 255 on the print: bright letters on the dark film (frontview sidelight), dark ink (backview, backlight).
    # Each pixel is measured against the film around it (an opening / closing wider than a stroke wipes the print
    # out): the side light is brighter at one edge, so one threshold over the crop takes film for print and back.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    small = cv2.resize(blurred, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    film = cv2.morphologyEx(small, cv2.MORPH_OPEN if bright_print else cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FILM_SIZE, FILM_SIZE)))
    film = cv2.resize(film, gray.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ratio = blurred / np.maximum(film, 1)
    if bright_print:  # print up to ~4x the film: 64 per film level keeps it in 0..255
        _, mask = cv2.threshold(np.clip(ratio * 64, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(np.clip(ratio * 255, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def align(mask, reference_mask):
    # Homography that puts `mask` on `reference_mask` (a plain shift, by template matching, leaves strips along
    # the letters at the label ends: lens distortion makes the print up to ~35 px longer, 35/54 right instead of
    # 49/54 on 2026-11-20). Found on the blurred masks, so the light does not matter. For full size images, use
    # with WARP_INVERSE_MAP.
    def small(m):
        m = cv2.resize(m, None, fx=ALIGN_SCALE, fy=ALIGN_SCALE, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(m, (0, 0), 3).astype(np.float32)
    warp = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    _, warp = cv2.findTransformECC(small(reference_mask), small(mask), warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 1)
    scale = np.diag([ALIGN_SCALE, ALIGN_SCALE, 1]).astype(np.float32)
    return np.linalg.inv(scale) @ warp @ scale


def reference_print(gray, bright_print):
    # The reference's print mask without pieces touching the crop border (dark panel corners, not print: the
    # crop has PRINT_PAD px around the print) and without small pieces near it (film corners against the dark
    # panel, the date): they fall elsewhere on every label and would show as missing print.
    mask = print_mask(gray, bright_print)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    h, w = mask.shape
    x, y, bw, bh, area = stats.T
    edge = np.minimum.reduce([x, y, w - x - bw, h - y - bh])
    near = np.flatnonzero((area < EDGE_PIECE_AREA) & (edge <= EDGE_PIECE_DISTANCE))
    mask[np.isin(labels, np.union1d(border[border > 0], near[near > 0]))] = 0
    return mask


def regions(mask):
    # Bounding boxes and areas of the pieces of `mask` that are at least MIN_AREA px.
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(int(v) for v in stats[i]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]


def compare(gray, reference_mask, bright_print):
    # (missing regions, added regions, missing mask, added mask); the masks are put back on the crop.
    mask = print_mask(gray, bright_print)
    warp = align(mask, reference_mask)
    size = reference_mask.shape[::-1]
    mask = cv2.warpPerspective(mask, warp, size, flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near = cv2.dilate(reference_mask, disk(NEAR_PRINT))
    missing = cv2.morphologyEx(reference_mask & ~mask, cv2.MORPH_OPEN, disk(OPEN // 2))
    added = cv2.morphologyEx(mask & ~reference_mask & near, cv2.MORPH_OPEN, disk(OPEN // 2))
    if bright_print:  # sidelight glare on the film is as bright as the print: added print is checked on backview
        added[:] = 0
    back = lambda m: cv2.warpPerspective(m, warp, gray.shape[::-1], flags=cv2.INTER_NEAREST)
    return regions(missing), regions(added), back(missing), back(added)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, default=Path("data_crop"))
    args = parser.parse_args()

    for group in sorted(p for p in args.crops.glob("*/*") if p.name.startswith(GROUPS)):
        bright_print = group.name.startswith("frontview")
        paths = sorted(group.glob("train/good/*.png")) + sorted(group.glob("test/*/*.png"))
        reference_mask = reference_print(cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE), bright_print)
        print(f"== {group.relative_to(args.crops)}  (reference {paths[0].name})")
        for path in paths:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            missing_regions, added_regions, _, _ = compare(gray, reference_mask, bright_print)
            largest = max([r[4] for r in missing_regions + added_regions], default=0)
            kind = path.parent.name if path.parent.parent.name == "test" else "train"
            print(f"{largest:7d} px  missing {len(missing_regions)}  added {len(added_regions)}  {kind}/{path.stem}")


if __name__ == "__main__":
    main()
