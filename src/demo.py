"""Demo: what each check finds on every test crop, one folder per shot in the order they are taken.

1_frontview-backlight: dimensions (measure_check), missing / added print (template_diff; the backlight
    print is dark and sharp, no glare) and dirt or scratches on the film (AnomalyDINO ViT-B, whole film).
2_frontview-sidelight, 3_backview-sidelight: dirt and scratches on the letter surface (AnomalyDINO ViT-B,
    map kept on the letters only).

Usage: uv run python src/demo.py [--src data] [--crops data_crop] [--models models] [--out results/demo]
AnomalyDINO uses the models saved by  uv run python src/anomaly_dino/anomaly_dino.py build.
Each group's photos are cut first (crop_label.py: data -> data_crop), so the crops always match data/.
Writes <out>/<sku>/<n>_<group>/<kind>_<name>.jpg: the crop, then what each check found, each titled.
No verdict: the pictures show how the checks work. Labels are not matched across the three shots.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "anomaly_dino"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "template_diff"))
import crop_label  # noqa: E402
import measure_check  # noqa: E402
from anomaly_dino import anomaly_maps, checked_area  # noqa: E402
from template_diff import MIN_AREA as PRINT_MIN_AREA, compare, reference_print  # noqa: E402

SHOTS = ("frontview-backlight", "frontview-sidelight", "backview-sidelight")  # capture order
RED, MAGENTA, CYAN = (0, 0, 255), (255, 0, 255), (255, 255, 0)


def titled(image, title):
    # The picture with its title on a white strip below.
    strip = np.full((90, image.shape[1], 3), 255, np.uint8)
    (w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    cv2.putText(strip, title, ((image.shape[1] - w) // 2, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return np.vstack([image, strip])


def outline(picture, mask, color, min_area):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(picture, [c for c in contours if cv2.contourArea(c) >= min_area], -1, color, 4)
    return picture


def heatmap(gray, amap, area, limit):
    # Colored so the threshold is mid scale (the raw distances are small, ~0.1 to 0.6).
    original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((np.clip(np.where(area, amap, 0) * 0.5 / limit, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    return cv2.addWeighted(original, 0.5, heat, 0.5, 0)


def stacked(panels):
    gap = np.full((30, panels[0].shape[1], 3), 255, np.uint8)
    out = np.vstack([p for panel in panels for p in (panel, gap)][:-1])
    return cv2.resize(out, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)


def sidelight_pictures(group, maps, limit):
    bright_print = group.name.startswith("frontview")
    train0 = cv2.imread(str(sorted(group.glob("train/good/*.png"))[0]), cv2.IMREAD_GRAYSCALE)
    reference_mask = reference_print(train0, bright_print)
    for path, amap in maps.items():
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        on_print = checked_area(group, gray, reference_mask)
        found = outline(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), on_print & (amap >= limit), RED, 0)  # every pixel over the threshold
        yield path, stacked([titled(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "1. Crop"),
                             titled(heatmap(gray, amap, on_print, limit), "2. AnomalyDINO map, print only"),
                             titled(found, f"3. Print surface defects (red: map >= {limit:.3f})")])


def backlight_pictures(group, maps, settings_dir, limit):
    # Everything on the crop with measure_check's film border, so print cut by the film edge stays print.
    prepared = measure_check.prepare(settings_dir, group)
    rects = measure_check.crop_label.load_rects(group)
    train0 = cv2.imread(str(sorted(group.glob("train/good/*.png"))[0]), cv2.IMREAD_GRAYSCALE)
    reference_mask = reference_print(measure_check.with_border(train0), False)
    b = measure_check.FILM_BORDER
    for path, amap in maps.items():
        gray = measure_check.with_border(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        amap = cv2.copyMakeBorder(amap, b, b, b, b, cv2.BORDER_CONSTANT, value=0)
        film = np.zeros(gray.shape, bool)
        film[b:-b, b:-b] = True
        _, _, missing, added = compare(gray, reference_mask, False)
        found = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        outline(found, amap >= limit, RED, 0)  # every pixel over the threshold
        outline(found, missing, MAGENTA, PRINT_MIN_AREA)
        outline(found, added, CYAN, PRINT_MIN_AREA)
        dimensions = measure_check.check(gray, prepared, rects[path.relative_to(group).as_posix()])["image"]
        yield path, stacked([titled(dimensions, "1. Dimensions (red: off tolerance / print not found)"),
                             titled(heatmap(gray, amap, film, limit), "2. AnomalyDINO map, whole film"),
                             titled(found, f"3. Red: defect (map >= {limit:.3f}), magenta: missing, cyan: added")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data"))
    parser.add_argument("--crops", type=Path, default=Path("data_crop"))
    parser.add_argument("--out", type=Path, default=Path("results/demo"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()

    for source in crop_label.groups(args.src):
        shot = next((n for n, s in enumerate(SHOTS, start=1) if source.name.startswith(s)), None)
        if shot is None:
            continue
        crop_label.cut(source, args.src, args.crops)  # the labels cut out of the photos as they are now
        group = args.crops / source.relative_to(args.src)
        out = args.out / group.parent.name / f"{shot}_{group.name}"
        out.mkdir(parents=True, exist_ok=True)
        maps, limit = anomaly_maps(group, args.models)
        if "backlight" in group.name:
            pictures = backlight_pictures(group, maps, args.src / group.relative_to(args.crops), limit)
        else:
            pictures = sidelight_pictures(group, maps, limit)
        for path, picture in pictures:
            cv2.imwrite(str(out / f"{path.parent.name}_{path.stem}.jpg"), picture)
        print(f"wrote {out}  (threshold {limit:.3f})")


# Guard: DataLoader workers started with spawn/forkserver (default from Python 3.14) re-import this file.
if __name__ == "__main__":
    main()
