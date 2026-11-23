"""Cut the label out of every image by matching the group's template, at any place and angle.

Each group that has a data/<sku>/<group>/template.png is cropped: the template is a straight crop of
one good label with TEMPLATE_MARGIN px of background around it. Every crop has the label straight and
the same way up. Frontview-backlight then moves each edge to the film edge found near it (the film is
cut a little differently each time), to measure it; the other groups only need the whole print, so they
keep the template's box with PRINT_PAD px more on every side. Backview crops are flipped left-right
so the print reads (the photo sees it from behind).

Usage: uv run python src/crop_label.py [--src data] [--dst data_crop]
(anomaly_dino.py build and demo.py cut each group themselves first: running this alone is only needed to
look at the crops.)
Keeps the folder layout (data/<sku>/<group>/<split>/<kind>/x.bmp -> data_crop/.../x.png)
writes each crop's box in its photo to <dst>/<sku>/<group>/rects.json (measure_check turns its
measurements into mm on the photo: load_rects) and a small picture with the found box per image to
<dst>/_check/ for a quick look.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_NAME = "template.png"
RECTS_NAME = "rects.json"  # per cropped group: each crop's box in its photo
TEMPLATE_MARGIN = 0      # px of background around the label in the template, not part of the crop
COARSE_SCALE = 0.0625    # first search: every angle on a small copy (~150 x 130 px: on the CPU, ~0.35 ms per angle)
COARSE_STEP = 1.0        # degrees
COARSE_CANDIDATES = 4    # best different coarse places that get a fine search
CANDIDATE_ANGLE = 10     # degrees: coarse places closer than this in angle and half the label short side apart are one place
FINE_SCALE = 0.5         # second search: around the best coarse angle on a larger copy
FINE_RANGE = 1.5         # degrees either side of the coarse angle
FINE_STEP = 0.1          # degrees
FINE_WINDOW = 40         # px at FINE_SCALE either side of where the coarse search put the label: the fine
                         # search matches only there, not over the whole photo
# CUDA OpenCV (link_opencv_cuda.sh) runs the fine search; the coarse one stays on the CPU, where its small images are
# faster (each GPU call costs ~2 ms whatever the size). pip OpenCV runs both on the CPU.
GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
PRINT_PAD = 50           # px added around the print-only box: lens distortion makes the print up to
                         # ~35 px longer than the template near the photo corners
EDGE_SEARCH = 80        # px either side of the template's film edge where the real edge is looked for:
                         # the film is cut up to ~60 px off from the print, which the template follows


def turn_matrix(shape, degrees):
    # The 2x3 matrix that turns an image of this shape around its center onto a square canvas big enough
    # to keep the corners, and the canvas side.
    h, w = shape
    side = int(np.ceil(np.hypot(w, h)))
    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    rotation[0, 2] += side / 2 - w / 2
    rotation[1, 2] += side / 2 - h / 2
    return rotation, side


def best_angle(gray, template, scale, angles, near=None):
    # Turn the photo by every angle and match the straight template: [(score, angle, template center in
    # photo pixels)], the best place at each angle, best first. Turning the photo, not the template, needs no mask for the corners.
    # near: the template center in photo pixels from an earlier search; only FINE_WINDOW px around it is matched.
    image = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gpu = GPU and near is not None
    if gpu:
        gpu_image, gpu_small = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        gpu_image.upload(image)
        gpu_small.upload(small)
        matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
    th, tw = small.shape
    found = []
    for angle in angles:
        rotation, side = turn_matrix(image.shape, angle)
        x0, y0, x1, y1 = 0, 0, side, side
        if near is not None:
            cx, cy = rotation @ np.array([near[0] * scale, near[1] * scale, 1.0])
            x0, y0 = max(int(cx - tw / 2) - FINE_WINDOW, 0), max(int(cy - th / 2) - FINE_WINDOW, 0)
            x1, y1 = min(int(cx + tw / 2) + FINE_WINDOW, side), min(int(cy + th / 2) + FINE_WINDOW, side)
        if gpu:
            image_turned = cv2.cuda.warpAffine(gpu_image, rotation, (side, side), flags=cv2.INTER_LINEAR)
            scores = matcher.match(image_turned.rowRange(y0, y1).colRange(x0, x1), gpu_small)
            _, score, _, (x, y) = cv2.cuda.minMaxLoc(scores)
        else:
            image_turned = cv2.warpAffine(image, rotation, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
            _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(image_turned[y0:y1, x0:x1], small, cv2.TM_CCOEFF_NORMED))
        x, y = x + x0, y + y0
        center_turned = np.array([x + small.shape[1] / 2, y + small.shape[0] / 2, 1.0])
        center = cv2.invertAffineTransform(rotation) @ center_turned / scale
        found.append((float(score), float(angle), (float(center[0]), float(center[1]))))
    return sorted(found, key=lambda f: -f[0])


def find_label(gray, template):
    # The label's box in the photo, as cv2.minAreaRect gives it, and the match score (0..1).
    # Coarse over the full turn and the whole photo, then fine around each of the COARSE_CANDIDATES best different
    # coarse places: on the small copy a faint print (backview) can win at a wrong place, the fine search corrects it.
    h, w = template.shape
    candidates = []
    for _, angle, center in best_angle(gray, template, COARSE_SCALE, np.arange(0, 360, COARSE_STEP)):
        if all(abs((angle - a + 180) % 360 - 180) > CANDIDATE_ANGLE or np.hypot(center[0] - c[0], center[1] - c[1]) > min(w, h) / 2
               for a, c in candidates):
            candidates.append((angle, center))
            if len(candidates) == COARSE_CANDIDATES:
                break
    score, angle, center = max(best_angle(gray, template, FINE_SCALE,
                                          np.arange(a - FINE_RANGE, a + FINE_RANGE + 1e-6, FINE_STEP), c)[0]
                               for a, c in candidates)
    return (center, (w - 2 * TEMPLATE_MARGIN, h - 2 * TEMPLATE_MARGIN), angle), score


def rect_matrix(rect):
    # 2x3 matrix from photo pixels to crop pixels: turn by the box angle around the box center,
    # then move the box's top-left corner to (0, 0).
    center, (w, h), angle = rect
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation[0, 2] += round(w) / 2 - center[0]
    rotation[1, 2] += round(h) / 2 - center[1]
    return rotation


def crop_rect(gray, rect):
    # Cut the box out of the photo, straight (no scaling).
    _, (w, h), _ = rect
    return cv2.warpAffine(gray, rect_matrix(rect), (int(round(w)), int(round(h))),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def edge_profile(values):
    # Brightness along one axis: the 90th percentile across the other axis, so dark print does not count
    # (the film between the letters is still seen), then smoothed.
    return cv2.GaussianBlur(values.astype(np.float32).reshape(-1, 1), (1, 9), 0).ravel()


def step_at(profile, start, stop, darker_after):
    # Index in [start, stop) where the profile drops most (panel -> film) or rises most (film -> panel).
    slope = np.diff(profile[start:stop + 1])
    return start + int(np.argmin(slope) if darker_after else np.argmax(slope)) + 1


def film_edges(wide, margin):
    # Film edges in a straight crop that has about `margin` px of panel on every side:
    # (left, top, right, bottom), searched within `margin` px of where the template puts them.
    # Profiles over the middle 60% of the other side only: the panel's dark corners fall outside it.
    h, w = wide.shape
    columns = edge_profile(np.percentile(wide[int(h * 0.2):int(h * 0.8)], 90, axis=0))
    rows = edge_profile(np.percentile(wide[:, int(w * 0.2):int(w * 0.8)], 90, axis=1))
    left = step_at(columns, 0, 2 * margin, darker_after=True)
    right = step_at(columns, w - 2 * margin - 1, w - 1, darker_after=False)
    top = step_at(rows, 0, 2 * margin, darker_after=True)
    bottom = step_at(rows, h - 2 * margin - 1, h - 1, darker_after=False)
    return left, top, right, bottom


def tight_rect(gray, rect):
    # The template box, moved and resized to the film edges found around it.
    center, (w, h), angle = rect
    wide_rect = (center, (w + 2 * EDGE_SEARCH, h + 2 * EDGE_SEARCH), angle)
    left, top, right, bottom = film_edges(crop_rect(gray, wide_rect), EDGE_SEARCH)
    middle = np.array([(left + right) / 2, (top + bottom) / 2, 1.0])
    new_center = cv2.invertAffineTransform(rect_matrix(wide_rect)) @ middle
    return (float(new_center[0]), float(new_center[1])), (right - left, bottom - top), angle


def box_points(rect):
    # Corners of the box in the photo: the crop's corners mapped back with the crop's own matrix.
    _, (w, h), _ = rect
    corners = np.array([[0, 0, 1], [round(w), 0, 1], [round(w), round(h), 1], [0, round(h), 1]], np.float64)
    return (corners @ cv2.invertAffineTransform(rect_matrix(rect)).T).astype(np.int32)


def mirrored(group_name):
    # Backview photos show the print from behind: those crops are flipped left-right so the print reads.
    return group_name.startswith("backview")


def group_paths(group):
    return sorted(group.glob("train/good/*.bmp")) + sorted(group.glob("test/*/*.bmp"))


def crop_group(group):
    # Every image of one group: [(path, rect, score, crop)].
    template = cv2.imread(str(group / TEMPLATE_NAME), cv2.IMREAD_GRAYSCALE)
    results = []
    for path in group_paths(group):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rect, score = find_label(gray, template)
        if group.name.startswith("frontview-backlight"):  # whole film, to measure; other groups only the print
            rect = tight_rect(gray, rect)
        else:
            center, (w, h), angle = rect
            rect = center, (w + 2 * PRINT_PAD, h + 2 * PRINT_PAD), angle
        crop = crop_rect(gray, rect)
        results.append((path, rect, score, cv2.flip(crop, 1) if mirrored(group.name) else crop))
    return results


def load_rects(crop_group):
    # {crop path relative to its group folder: rect} as save_group wrote it.
    rects = json.loads((crop_group / RECTS_NAME).read_text())
    return {name: ((r[0], r[1]), (r[2], r[3]), r[4]) for name, r in rects.items()}


def save_group(results, src, dst):
    # Crops to dst (same layout as src), their boxes to dst/<sku>/<group>/rects.json and a small picture of
    # each box to dst/_check/.
    rects = {}
    for path, rect, score, crop in results:
        out = dst / path.relative_to(src).with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop)
        (cx, cy), (w, h), angle = rect
        rects[out.relative_to(out.parents[2]).as_posix()] = [float(v) for v in (cx, cy, w, h, angle)]

        check = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(check, [box_points(rect)], -1, (0, 0, 255), 10)
        cv2.putText(check, f"score {score:.2f}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
        check_path = dst / "_check" / ("__".join(path.relative_to(src).parts[:-1]) + "__" + path.stem + ".jpg")
        check_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(check_path), cv2.resize(check, None, fx=0.25, fy=0.25))
    (out.parents[2] / RECTS_NAME).write_text(json.dumps(rects, indent=1))


def cut(group, src=Path("data"), dst=Path("data_crop")):
    # Every photo of one group (src/<sku>/<group>) cut and saved under dst; [(path, rect, score, crop)].
    results = crop_group(group)
    save_group(results, src, dst)
    return results


def groups(src=Path("data")):
    # The groups that can be cut: the ones with a template.
    return sorted(p.parent for p in src.glob(f"*/*/{TEMPLATE_NAME}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data"))
    parser.add_argument("--dst", type=Path, default=Path("data_crop"))
    args = parser.parse_args()

    for group in groups(args.src):
        for path, rect, score, _ in cut(group, args.src, args.dst):
            print(f"score {score:.2f}  angle {rect[2]:6.1f}  {path}")


if __name__ == "__main__":
    main()
