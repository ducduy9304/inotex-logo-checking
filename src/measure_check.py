"""Layout check of the frontview-backlight crops, in mm on the label plane.

The method of label_defect's app/measure_check.py, cut down for these crops:
- crop_label.py already cut every film out straight, the same way up, with the crop border on the film
  edge, so the film edges are the crop border (no label mask, no turning).
- the print is dark on bright film; each block's place is found by template matching, then its box is
  the ink found there (label_defect's ink_box "image").
- distances in mm: the two ends of each measurement (found in crop pixels) are taken back to the photo
  (the crop's box in rects.json, crop_label.py), through the lens model and onto the label plane with the
  plane calibration the SKU's photos were taken with (data/<sku>/calibration.json, copied there by
  camera/collect.py; camera/calibration.json for a SKU without one), as camera/rectify.py does for whole
  images. The models still see the photos as they are. nominal = mean over train/good.

Usage: uv run python src/measure_check.py [--src data] [--crops data_crop]
Checks every group that has a data/<sku>/<group>/measure_settings.json (frontview-backlight: the only
group cropped to the film edges). Prints the verdict and faults of every crop; no pictures (src/demo.py
draws them).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_label  # noqa: E402

SETTINGS_NAME = "measure_settings.json"
CALIBRATION = Path(__file__).resolve().parents[1] / "camera" / "calibration.json"  # for a SKU without its own
MATCH_BLUR = 9          # blur before matching, so small print noise does not decide where a template fits
FILM_BORDER = 60        # film-colored px added around every crop: the film is cut close to the date on
                        # some labels, and its template (with TEMPLATE_PAD) must still fit there
TEMPLATE_PAD = 8        # film px kept around a block's ink box in its template, so its outline counts
INK_MARGIN = 40         # the ink is looked for this far around the template's ink box: lens distortion makes
                        # a letter up to ~35 px bigger than its template near the photo corners
MIN_INK_AREA = 10       # darker specks smaller than this are not ink (the date's thin strokes are ~13 px)
GAP_LINE_OFFSET = 12
FAULT_COLOR = (0, 0, 255)
PASS_COLOR = (80, 185, 63)
BLOCK_COLOR = (255, 0, 255)
LINE_WIDTH = 2
TEXT_SCALE = 0.8
TEXT_WIDTH = 2


def load_plane(path=CALIBRATION):
    # The lens model and the homography from undistorted camera pixels to the rectified image
    # (mm_per_pixel mm per pixel) of camera/calibrate.py.
    calibration = json.loads(Path(path).read_text())
    return {"camera_matrix": np.array(calibration["camera_matrix"]), "dist_coeffs": np.array(calibration["dist_coeffs"]),
            "camera_to_rectified": np.array(calibration["camera_to_rectified"]), "mm_per_pixel": calibration["mm_per_pixel"]}


def plane_mapper(plane, rect, crop_width, flip):
    # Function: points in the bordered crop (px) -> mm on the label plane.
    to_photo = cv2.invertAffineTransform(crop_label.rect_matrix(rect))

    def to_mm(points):
        points = np.asarray(points, np.float64).reshape(-1, 2) - FILM_BORDER
        if flip:
            points[:, 0] = crop_width - 1 - points[:, 0]
        photo = points @ to_photo[:, :2].T + to_photo[:, 2]
        undistorted = cv2.undistortPoints(photo.reshape(-1, 1, 2), plane["camera_matrix"], plane["dist_coeffs"],
                                          P=plane["camera_matrix"])
        return cv2.perspectiveTransform(undistorted, plane["camera_to_rectified"]).reshape(-1, 2) * plane["mm_per_pixel"]
    return to_mm


def with_border(crop):
    # Constant film brightness, not replicated edge rows: print cut by the edge must not be smeared out.
    return cv2.copyMakeBorder(crop, FILM_BORDER, FILM_BORDER, FILM_BORDER, FILM_BORDER,
                              cv2.BORDER_CONSTANT, value=int(np.median(crop)))


def film_edges(bordered):
    h, w = bordered.shape
    return {"left": float(FILM_BORDER), "right": float(w - 1 - FILM_BORDER),
            "top": float(FILM_BORDER), "bottom": float(h - 1 - FILM_BORDER)}


def smooth(crop):
    return cv2.GaussianBlur(crop, (MATCH_BLUR, MATCH_BLUR), 0).astype(np.float32)


def template_box(block):
    # Block box from the settings (px from the film's top-left corner) -> its template box in the bordered crop.
    offset = FILM_BORDER - TEMPLATE_PAD
    return {"left": block["left"] + offset, "right": block["right"] + offset + 2 * TEMPLATE_PAD,
            "top": block["top"] + offset, "bottom": block["bottom"] + offset + 2 * TEMPLATE_PAD}


def cut(image, box):
    return image[int(box["top"]):int(box["bottom"]) + 1, int(box["left"]):int(box["right"]) + 1]


def find_block(smoothed, expected, template, search_margin):
    # Best place of the template within search_margin around its place in the reference: the template's
    # top-left corner in crop pixels and the score (normalized correlation, 1 = same as the template).
    template_height, template_width = template.shape
    image_height, image_width = smoothed.shape
    left = max(0, int(expected["left"]) - search_margin)
    top = max(0, int(expected["top"]) - search_margin)
    right = min(image_width, int(expected["left"]) + template_width + search_margin)
    bottom = min(image_height, int(expected["top"]) + template_height + search_margin)
    window = smoothed[top:bottom, left:right]
    if window.shape[0] < template_height or window.shape[1] < template_width:
        return expected["left"], expected["top"], 0.0  # block (partly) off the film: not found
    scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    return float(left + x), float(top + y), float(score)


def ink_box_in_image(crop, placed_box):
    # The dark ink around the template's ink box. Only blobs whose middle lies in that box count: the
    # letters touch or overlap their neighbours' boxes (f and a, e and t).
    h, w = crop.shape
    left = max(0, int(placed_box["left"]) - INK_MARGIN)
    right = min(w - 1, int(placed_box["right"]) + INK_MARGIN)
    top = max(0, int(placed_box["top"]) - INK_MARGIN)
    bottom = min(h - 1, int(placed_box["bottom"]) + INK_MARGIN)
    region = cv2.GaussianBlur(crop[top:bottom + 1, left:right + 1], (5, 5), 0)
    _, ink = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, components, stats, centroids = cv2.connectedComponentsWithStats(ink)
    kept = []
    for component in range(1, count):
        x, y = centroids[component] + (left, top)
        if (stats[component, cv2.CC_STAT_AREA] >= MIN_INK_AREA
                and placed_box["left"] <= x <= placed_box["right"] and placed_box["top"] <= y <= placed_box["bottom"]):
            kept.append(component)
    if not kept:
        return placed_box  # no ink (missing print, its score shows it): keep the template's box
    kept_ink = np.isin(components, kept)
    rows = np.flatnonzero(kept_ink.any(axis=1))
    columns = np.flatnonzero(kept_ink.any(axis=0))
    box = {"left": float(left + columns[0]), "right": float(left + columns[-1]),
           "top": float(top + rows[0]), "bottom": float(top + rows[-1])}
    # Ink that runs to the region border is a neighbour joined to this block (sku02: w, a and y touch):
    # that side is taken from the template.
    for side, border in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
        if box[side] == border:
            box[side] = placed_box[side]
    return box


def all_blocks(settings):
    # [(block, in_chain)]
    return [(b, True) for b in settings["blocks"]] + [(b, False) for b in settings["free_blocks"]]


def find_blocks(crop, templates, settings):
    smoothed = smooth(crop)
    blocks = []
    for (block, in_chain), template in zip(all_blocks(settings), templates):
        x, y, score = find_block(smoothed, template_box(block), template, settings["search_margin"])
        placed = {"left": x + TEMPLATE_PAD, "top": y + TEMPLATE_PAD,
                  "right": x + TEMPLATE_PAD + block["right"] - block["left"],
                  "bottom": y + TEMPLATE_PAD + block["bottom"] - block["top"]}
        box = ink_box_in_image(crop, placed) if settings.get("ink_box", "image") == "image" else dict(placed)
        box.update(name=block["name"], score=score, in_chain=in_chain)
        blocks.append(box)
    return blocks


def gap_name(previous, number):
    return "gap_%d_%d" % (previous, number)


def chain_numbers(blocks):
    return [n for n, b in enumerate(blocks, start=1) if b["in_chain"]]


def measurement_lines(edges, blocks):
    # Every distance of one crop as its two ends in crop px, by name: a block to each film edge through the
    # block's middle, the gap between chain blocks near the bottom edge, the film's width and height.
    left, right, top, bottom = (edges[k] for k in ("left", "right", "top", "bottom"))
    lines = {"film_width": ((left, (top + bottom) / 2), (right, (top + bottom) / 2)),
             "film_height": (((left + right) / 2, top), ((left + right) / 2, bottom))}
    for number, block in enumerate(blocks, start=1):
        name = "block_%d" % number
        cx, cy = (block["left"] + block["right"]) / 2, (block["top"] + block["bottom"]) / 2
        lines[name + "_to_left"] = ((left, cy), (block["left"], cy))
        lines[name + "_to_right"] = ((block["right"], cy), (right, cy))
        lines[name + "_to_top"] = ((cx, top), (cx, block["top"]))
        lines[name + "_to_bottom"] = ((cx, block["bottom"]), (cx, bottom))
    chain = chain_numbers(blocks)
    for previous, number in zip(chain, chain[1:]):
        gap_y = bottom - GAP_LINE_OFFSET
        lines[gap_name(previous, number)] = ((blocks[previous - 1]["right"], gap_y), (blocks[number - 1]["left"], gap_y))
    return lines


def layout_values(edges, blocks, to_mm):
    # Every distance of one crop, in mm on the label plane, by name.
    lines = measurement_lines(edges, blocks)
    ends = to_mm([point for line in lines.values() for point in line]).reshape(-1, 2, 2)
    return {name: float(np.linalg.norm(end[1] - end[0])) for name, end in zip(lines, ends)}


def checked_measurements(blocks):
    # What decides the verdict, as in label_defect: the chain left edge -> block -> gap -> ... -> right
    # edge and the top / bottom of each chain block; a free block to the left and top edges. Film size and
    # a block to the far edge are sums of these, only reported.
    chain = chain_numbers(blocks)
    names = set()
    for number in range(1, len(blocks) + 1):
        name = "block_%d" % number
        if number not in chain:
            names.update([name + "_to_left", name + "_to_top"])
            continue
        names.update([name + "_to_top", name + "_to_bottom"])
        place = chain.index(number)
        if place == 0:
            names.add(name + "_to_left")
        else:
            names.add(gap_name(chain[place - 1], number))
        if place == len(chain) - 1:
            names.add(name + "_to_right")
    return names


def describe(name, blocks):
    parts = name.split("_")
    if parts[0] == "film":
        return "Film " + parts[1]
    if parts[0] == "gap":
        return "Gap %s - %s" % (blocks[int(parts[1]) - 1]["name"], blocks[int(parts[2]) - 1]["name"])
    return "%s -> %s edge" % (blocks[int(parts[1]) - 1]["name"], parts[-1])


def prepare(settings_dir, crop_dir):
    # Settings, plane calibration, templates (mean over train/good) and the nominal value of every distance.
    settings = json.loads((settings_dir / SETTINGS_NAME).read_text())
    sku_calibration = settings_dir.parent / CALIBRATION.name
    plane = load_plane(sku_calibration if sku_calibration.is_file() else CALIBRATION)
    flip = crop_label.mirrored(crop_dir.name)
    rects = crop_label.load_rects(crop_dir)
    paths = group_crops(crop_dir, "train/good")
    golden = [with_border(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]
    reference = smooth(golden[0])
    reference_templates = [cut(reference, template_box(block)) for block, _ in all_blocks(settings)]
    templates = []
    for (block, _), reference_template in zip(all_blocks(settings), reference_templates):
        cuts = []
        for crop in golden:
            smoothed = smooth(crop)
            x, y, _ = find_block(smoothed, template_box(block), reference_template, settings["search_margin"])
            h, w = reference_template.shape
            cuts.append(smoothed[int(y):int(y) + h, int(x):int(x) + w])
        templates.append(np.mean(cuts, axis=0))
    golden_values = [layout_values(film_edges(c), find_blocks(c, templates, settings),
                                   plane_mapper(plane, rects[p.relative_to(crop_dir).as_posix()], c.shape[1] - 2 * FILM_BORDER, flip))
                     for p, c in zip(paths, golden)]
    nominal = {name: float(np.mean([v[name] for v in golden_values])) for name in golden_values[0]}
    return {"settings": settings, "templates": templates, "nominal": nominal, "golden_values": golden_values,
            "plane": plane, "flip": flip}


def draw_dimension(picture, start, end, text, color):
    cv2.line(picture, start, end, color, LINE_WIDTH, cv2.LINE_AA)
    cv2.circle(picture, start, 2 * LINE_WIDTH, color, -1)
    cv2.circle(picture, end, 2 * LINE_WIDTH, color, -1)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_WIDTH)
    x = min((start[0] + end[0]) // 2 + 6, picture.shape[1] - text_width - 2)
    y = max((start[1] + end[1]) // 2 - 6, text_height + 2)
    cv2.putText(picture, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, color, TEXT_WIDTH, cv2.LINE_AA)


def draw(crop, edges, blocks, values, failed, damaged):
    # The crop with every block box (magenta, red when damaged) and the checked distances in mm (green, red
    # when failed), placed as in label_defect.
    picture = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for number, block in enumerate(blocks, start=1):
        bl, br, bt, bb = (int(block[k]) for k in ("left", "right", "top", "bottom"))
        cv2.rectangle(picture, (bl, bt), (br, bb), FAULT_COLOR if number in damaged else BLOCK_COLOR, LINE_WIDTH)
    checked = checked_measurements(blocks)
    for name, (start, end) in measurement_lines(edges, blocks).items():
        if name in checked:
            draw_dimension(picture, tuple(int(v) for v in start), tuple(int(v) for v in end), "%.1f" % values[name],
                           FAULT_COLOR if name in failed else PASS_COLOR)
    return picture


def check(crop, prepared, rect):
    # Verdict, picture and one line per fault for one crop (with_border already added) cut from its photo
    # with the box rect.
    settings = prepared["settings"]
    edges = film_edges(crop)
    blocks = find_blocks(crop, prepared["templates"], settings)
    values = layout_values(edges, blocks, plane_mapper(prepared["plane"], rect, crop.shape[1] - 2 * FILM_BORDER,
                                                       prepared["flip"]))
    faults = []
    damaged = []
    for number, block in enumerate(blocks, start=1):
        if block["score"] < settings["min_score"]:
            damaged.append(number)
            faults.append("%s: print damaged or missing (match %.2f)" % (block["name"], block["score"]))
    failed = []
    for name in sorted(checked_measurements(blocks)):
        tolerance = settings["tolerances_mm"].get(name, settings["tolerance_mm"])
        if abs(values[name] - prepared["nominal"][name]) > tolerance:
            failed.append(name)
            faults.append("%s: %.2f mm (nominal %.2f +- %.2f)"
                          % (describe(name, blocks), values[name], prepared["nominal"][name], tolerance))
    verdict = "NG" if faults else "OK"
    return {"verdict": verdict, "faults": faults, "image": draw(crop, edges, blocks, values, failed, damaged)}


def group_crops(group_crop_dir, pattern):
    return sorted(group_crop_dir.glob(pattern + "/*.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data"))
    parser.add_argument("--crops", type=Path, default=Path("data_crop"))
    args = parser.parse_args()

    for group in sorted(p.parent for p in args.src.glob(f"*/*/{SETTINGS_NAME}")):
        crop_dir = args.crops / group.relative_to(args.src)
        prepared = prepare(group, crop_dir)
        rects = crop_label.load_rects(crop_dir)
        print(f"== {group}")
        # Spread of every checked distance over train/good, to pick the tolerance.
        print("train/good spread (max |value - nominal|, mm):")
        for name in sorted(prepared["nominal"]):
            spread = max(abs(v[name] - prepared["nominal"][name]) for v in prepared["golden_values"])
            print(f"  {name:22s} nominal {prepared['nominal'][name]:7.2f}  spread {spread:5.2f}")

        for path in group_crops(crop_dir, "train/good") + group_crops(crop_dir, "test/*"):
            result = check(with_border(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)), prepared,
                           rects[path.relative_to(crop_dir).as_posix()])
            print(f"{result['verdict']}  {path.relative_to(args.crops)}")
            for fault in result["faults"]:
                print("      " + fault)

if __name__ == "__main__":
    main()
