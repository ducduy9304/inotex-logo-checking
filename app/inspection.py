# The three checks of one label for the live app, in the order the label is photographed:
#   1. dimensions and film (frontview-backlight): dimensions (measure_check), missing / added print
#      (template_diff) and dirt or scratches on the whole film (AnomalyDINO)
#   2. front letter surface (frontview-sidelight): dirt, scratches, chips on the letters (AnomalyDINO, the
#      letter surface only) and missing print (template_diff)
#   3. back letter surface (backview-sidelight, the label turned over): the same on the back
# Each check takes a whole camera photo: the label is found and cut out as crop_label.py does it, then
# checked with the same functions and saved models as src/demo.py (anomaly_dino.py build).
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ("src", "src/anomaly_dino", "src/template_diff"):
    sys.path.insert(0, str(ROOT / folder))

import anomaly_dino  # noqa: E402
import crop_label  # noqa: E402
import demo  # noqa: E402
import measure_check  # noqa: E402
from template_diff import MIN_AREA as PRINT_MIN_AREA, compare, reference_print  # noqa: E402

DATA, CROPS, MODELS = ROOT / "data", ROOT / "data_crop", ROOT / "models"
# What the operator sees: the defects each step looks for, not how the photo is taken.
STEPS = [
    {"group": "frontview-backlight", "name": "Dimensions & film"},
    {"group": "frontview-sidelight", "name": "Front print surface"},
    {"group": "backview-sidelight", "name": "Back print surface"},
]
CAMERA_SETTINGS_NAME = "camera_settings.json"  # per group: ExposureTime (us), Gain (dB), as camera/collect.py
RED, MAGENTA, CYAN = (0, 0, 255), (255, 0, 255), (255, 255, 0)
LINE_WIDTH = 10  # the ~2700 px crop is shown ~600 px wide


def list_skus():
    # SKUs with a data folder for every step.
    return sorted(p.name for p in DATA.iterdir()
                  if p.is_dir() and all(any(g.name.startswith(s["group"]) for g in p.iterdir()) for s in STEPS))


def group_name(sku, step):
    # The step's group folder name, e.g. frontview-backlight-5000.
    return next(p.name for p in (DATA / sku).iterdir() if p.name.startswith(STEPS[step]["group"]))


def camera_settings(sku, step):
    # The camera settings the step's photos were taken with: data/<sku>/<group>/camera_settings.json, or the
    # exposure at the end of the group name (frontview-backlight-5000 -> 5000 us) and no gain.
    path = DATA / sku / group_name(sku, step) / CAMERA_SETTINGS_NAME
    if path.exists():
        return json.loads(path.read_text())
    return {"ExposureTime": float(group_name(sku, step).rsplit("-", 1)[1]), "Gain": 0.0}


def save_camera_settings(sku, step, settings):
    (DATA / sku / group_name(sku, step) / CAMERA_SETTINGS_NAME).write_text(json.dumps(settings, indent=2))


def prepare(sku, step):
    # Everything one step needs, loaded once: template, AnomalyDINO model and threshold, reference print
    # (and the dimension templates and nominal values for step 1).
    group = group_name(sku, step)
    crops = CROPS / sku / group
    model, size, threshold = anomaly_dino.load(crops, MODELS)
    prepared = {"step": step, "group": group, "crops": crops, "model": model, "size": size, "threshold": threshold,
                "template": cv2.imread(str(DATA / sku / group / crop_label.TEMPLATE_NAME), cv2.IMREAD_GRAYSCALE)}
    if step == 0:
        prepared["measure"] = measure_check.prepare(DATA / sku / group, crops)
        train0 = cv2.imread(str(sorted(crops.glob("train/good/*.png"))[0]), cv2.IMREAD_GRAYSCALE)
        prepared["reference"] = reference_print(measure_check.with_border(train0), False)
    else:
        prepared["reference"] = anomaly_dino.group_reference(crops)
    return prepared


def cut_label(gray, prepared):
    # (the label cut out of the photo, straight and the same way up; its box in the photo), as
    # crop_label.crop_group does it for one photo.
    rect, _ = crop_label.find_label(gray, prepared["template"])
    if prepared["step"] == 0:
        rect = crop_label.tight_rect(gray, rect)
    else:
        center, (w, h), angle = rect
        rect = center, (w + 2 * crop_label.PRINT_PAD, h + 2 * crop_label.PRINT_PAD), angle
    crop = crop_label.crop_rect(gray, rect)
    return (cv2.flip(crop, 1) if crop_label.mirrored(prepared["group"]) else crop), rect


def outline(picture, mask, color, min_area=0):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(picture, [c for c in contours if cv2.contourArea(c) >= min_area], -1, color, LINE_WIDTH)


def check(gray, prepared):
    # {"verdict": "OK" / "NG", "faults": [what is wrong, in words], "score", "threshold", "image": the crop
    # with the defects drawn, "process": the steps on the way (crop or dimensions, anomaly map, defects)}
    # for one grayscale photo.
    crop, rect = cut_label(gray, prepared)
    amap = anomaly_dino.crop_map(prepared["model"], crop, prepared["size"])
    threshold = prepared["threshold"]
    faults = []
    if prepared["step"] == 0:
        bordered = measure_check.with_border(crop)
        b = measure_check.FILM_BORDER
        amap = cv2.copyMakeBorder(amap, b, b, b, b, cv2.BORDER_CONSTANT, value=0)
        measured = measure_check.check(bordered, prepared["measure"], rect)
        missing_regions, added_regions, missing, added = compare(bordered, prepared["reference"], False)
        score = float(amap.max())
        first = demo.titled(measured["image"].copy(), "Dimensions")
        second = demo.titled(demo.heatmap(bordered, amap, np.ones(amap.shape, bool), threshold), "Anomaly map, whole film")
        picture = measured["image"]
        outline(picture, amap >= threshold, RED)
        outline(picture, missing, MAGENTA, PRINT_MIN_AREA)
        outline(picture, added, CYAN, PRINT_MIN_AREA)
        if measured["faults"]:
            faults.append("Dimension / print position (%d)" % len(measured["faults"]))
        if added_regions:
            faults.append("Extra print (%d)" % len(added_regions))
    else:
        area = anomaly_dino.checked_area(prepared["crops"], crop, prepared["reference"])
        missing_regions, _, missing, _ = compare(crop, prepared["reference"], prepared["group"].startswith("frontview"))
        score = float(amap[area].max())
        first = demo.titled(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), "Label")
        second = demo.titled(demo.heatmap(crop, amap, area, threshold), "Anomaly map, letters only")
        picture = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        outline(picture, area & (amap >= threshold), RED)
        outline(picture, missing, MAGENTA, PRINT_MIN_AREA)
    if missing_regions:
        faults.append("Missing print (%d)" % len(missing_regions))
    if score >= threshold:
        faults.append("Dirt / scratch / chip" if prepared["step"] else "Dirt / scratch on the film")
    third = demo.titled(picture.copy(), "Defects (red: map >= %.3f, magenta: missing, cyan: extra)" % threshold)
    return {"verdict": "NG" if faults else "OK", "faults": faults, "score": score, "threshold": threshold,
            "image": picture, "process": demo.stacked([first, second, third])}
