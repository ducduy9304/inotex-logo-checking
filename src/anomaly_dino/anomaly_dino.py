"""AnomalyDINO on each whole crop, with only the regions on the print reported.

For the print-only groups (frontview-sidelight, backview). Each crop goes in whole, resized to 252 px high
(anomalib's default height) and as wide as the group's mean crop shape allows (a multiple of 14, ~700-1000 px),
so a DINOv2 patch (14 px) covers ~33 px of the crop; DINOv2 takes the non-square input
(timm dynamic_img_size). Each crop goes in twice: as it is, and with half a patch cut off its top-left
corner, so the patch grid falls half a patch further on and a small defect cut in two by a patch border in
one view sits inside a patch in the other; the map is the larger of the two. The memory bank is built from
every patch of both views of the train/good crops (DINOv2 stays as it is: nothing is trained).
The map is the raw cosine distance to the nearest good patch. The threshold is leave-one-out on train/good:
each good crop is scored by a memory bank of the other good crops, and the highest value on the checked
area becomes the group's threshold.
The map is then kept only on the print: the print of the group's first train/good crop (template_diff),
warped onto each crop with a homography (lens distortion): the letter surface only, so glare, stains and
scratches on the background do not count. A chip on a letter edge is still inside the reference print.
Missing print is added from template_diff: AnomalyDINO's memory bank does not know where print should be,
so a missing letter looks like normal film to it.
frontview-backlight: the crop is the film itself (cut on its edges), so the whole map counts, background
and print alike; missing or moved print is left to measure_check.py on the same crops.

Everything goes through anomalib's Engine. Each group is built once: its checkpoint (memory bank inside)
is saved to models/<sku>/<group>.ckpt, its input size and threshold to models/<sku>/<group>.json;
checking a crop only loads them. Build again after adding or changing a group's train/good crops.

Usage: uv run python src/anomaly_dino/anomaly_dino.py build [sku02/backview-sidelight-10000 ...]
           cuts the labels out of the given groups' photos (data -> data_crop, crop_label.py), then builds
           and saves their models (all groups with a template.png if none is given)
       uv run python src/anomaly_dino/anomaly_dino.py [--crops data_crop] [--models models]
           prints the highest value on the print and the number of missing print regions of every test
           crop with the saved models (frontview-backlight: on the whole film); no pictures (src/demo.py)
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO
from torch.nn import functional as F  # noqa: N812

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "template_diff"))
import crop_label  # noqa: E402
from template_diff import GROUPS, align, compare, print_mask, reference_print  # noqa: E402


def print_region(gray, reference_mask, bright_print):
    # The reference print moved onto this crop.
    warp = align(print_mask(gray, bright_print), reference_mask)
    return cv2.warpPerspective(reference_mask, warp, gray.shape[::-1], flags=cv2.INTER_NEAREST) > 0


def checked_area(group, gray, reference_mask):
    # Where the map counts: the whole film for backlight, the letter surface otherwise.
    if "backlight" in group.name:
        return np.ones(gray.shape, bool)
    return print_region(gray, reference_mask, group.name.startswith("frontview"))


def group_reference(group):
    # The print of the group's first train/good crop (template_diff), where print_region starts from.
    train0 = cv2.imread(str(sorted(group.glob("train/good/*.png"))[0]), cv2.IMREAD_GRAYSCALE)
    return reference_print(train0, group.name.startswith("frontview") and "backlight" not in group.name)


def new_model(size):
    return AnomalyDINO(num_neighbours=1, encoder_name="vit_base_patch14_dinov2",
                       pre_processor=AnomalyDINO.configure_pre_processor(image_size=size))


def fit(normal_dir, size, work):
    # AnomalyDINO with a memory bank of every crop in normal_dir. No validation set (no synthetic defects), so
    # anomalib does not normalize and the map is the raw cosine distance to the nearest good patch.
    # The crops in normal_dir are already at the input size (write_views): anomalib resizes a batch of
    # different-sized images to its largest one first, which would stretch the smaller crops.
    datamodule = Folder(name="good", normal_dir=normal_dir, test_split_mode="none",
                        val_split_mode="none", train_batch_size=8, num_workers=4)
    model = new_model(size)
    shutil.rmtree(work, ignore_errors=True)  # anomalib adds a v0, v1, ... folder per run: keep one
    engine = Engine(default_root_dir=work)
    datamodule.setup()
    engine.fit(model=model, train_dataloaders=datamodule.train_dataloader())  # no val loader to ask for
    return engine, model


def views(gray, size):
    # The crop as it is and with half a patch (7 px of the input) cut off its top-left corner, each resized
    # to the input size here (anomalib resizes a batch of different-sized images to its largest one first,
    # which would stretch the smaller crops): [(view, (x, y) px cut off the crop)].
    h, w = gray.shape
    return [(cv2.resize(gray[y:, x:], size[::-1], interpolation=cv2.INTER_AREA), (x, y))
            for x, y in [(0, 0), (round(7 * w / size[1]), round(7 * h / size[0]))]]


def write_views(paths, folder, size):
    # views() of every crop as files, for anomalib's datasets: {view path (resolved): (crop path, (x, y))}.
    folder.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in paths:
        for view, (x, y) in views(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), size):
            file = folder / f"{path.stem}__{x}_{y}.png"
            cv2.imwrite(str(file), view)
            found[file.resolve()] = (path, (x, y))
    return found


def place(full, amap, x, y):
    # A view's map resized back to the view and kept where it is larger than the other view's.
    amap = cv2.resize(amap, (full.shape[1] - x, full.shape[0] - y), interpolation=cv2.INTER_LINEAR)
    full[y:, x:] = np.maximum(full[y:, x:], amap)


def model_paths(group, models):
    # (checkpoint, settings) files of a group.
    stem = models / group.parent.name / group.name
    return stem.with_suffix(".ckpt"), stem.with_suffix(".json")


def build(group, models, image_size=252):
    # Fits the group's memory bank on its train/good crops, sets its threshold by leave-one-out and saves both.
    # image_size: input height (multiple of 14; 252 is anomalib's default); the width follows the group's mean crop shape.
    train = sorted(group.glob("train/good/*.png"))
    test = sorted(group.glob("test/*/*.png"))
    aspect = np.mean([w / h for h, w in (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape for p in train + test)])
    size = (image_size, round(image_size * aspect / 14) * 14)
    reference_mask = group_reference(group)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        train_views = write_views(train, work / "views", size)

        engine, model = fit(work / "views", size, work / "anomalib")
        model = model.cuda().eval()
        full_bank = model.model.memory_bank

        # Leave-one-out: each good crop scored by a memory bank of the others; the highest score is the threshold.
        # A crop's patch features do not depend on the other crops, so they are taken once (as fit takes them:
        # pre-processor, DINOv2, L2 norm) and each held-out crop is scored against the others' features: the
        # numbers of a model fitted without it (within 0.001, float rounding: views one at a time, not batches of 8),
        # with one DINOv2 pass per view instead of one per view per fit (sku01 backview: 75 s instead of ~7.5 min).
        features = {}
        with torch.no_grad():
            for view_path, (path, _) in train_views.items():
                view = cv2.imread(str(view_path), cv2.IMREAD_GRAYSCALE)
                image = model.pre_processor(torch.from_numpy(view).cuda().float().div(255).expand(1, 3, *view.shape))
                found = F.normalize(model.model.extract_features(image).flatten(0, 1), p=2, dim=1)
                features.setdefault(path, []).append(found)
        scores = []
        for held_out in train:
            model.model.memory_bank = torch.cat([f for path, fs in features.items() if path != held_out for f in fs])
            gray = cv2.imread(str(held_out), cv2.IMREAD_GRAYSCALE)
            amap = crop_map(model, gray, size)
            scores.append(float(amap[checked_area(group, gray, reference_mask)].max()))
        model.model.memory_bank = full_bank
        threshold = max(scores)
        print(f"{group}: leave-one-out on {len(train)} good crops {' '.join(f'{v:.3f}' for v in scores)} -> threshold {threshold:.3f}")

        checkpoint, settings = model_paths(group, models)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        engine.trainer.save_checkpoint(checkpoint)
        settings.write_text(json.dumps({"size": size, "threshold": threshold}))
    print(f"saved {checkpoint}  ({checkpoint.stat().st_size / 1e6:.0f} MB)")


def load(group, models=Path("models")):
    # (model on the GPU, input size, threshold) saved by build().
    checkpoint, settings = model_paths(group, models)
    if not checkpoint.exists():
        sys.exit(f"{checkpoint} not found: run  uv run python src/anomaly_dino/anomaly_dino.py build")
    saved = json.loads(settings.read_text())
    size = tuple(saved["size"])
    model = AnomalyDINO.load_from_checkpoint(checkpoint, pre_processor=AnomalyDINO.configure_pre_processor(image_size=size))
    return model.cuda().eval(), size, saved["threshold"]


@torch.no_grad()
def crop_map(model, gray, size):
    # The map of one crop at crop size. model(image) is anomalib's forward: its pre-processor, AnomalyDINO and
    # its post-processor, as in engine.predict, without a Lightning run (and files) per call.
    full = np.zeros(gray.shape, np.float32)
    for view, (x, y) in views(gray, size):
        image = torch.from_numpy(view).cuda().float().div(255).expand(1, 3, *view.shape)  # gray -> RGB in [0, 1]
        place(full, model(image).anomaly_map.squeeze().cpu().numpy(), x, y)
    return full


def anomaly_maps(group, models=Path("models")):
    # ({test crop path: map at crop size}, threshold) with the group's saved model.
    model, size, threshold = load(group, models)
    return {p: crop_map(model, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), size)
            for p in sorted(group.glob("test/*/*.png"))}, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=["build", "check"], default="check")
    parser.add_argument("groups", nargs="*", help="build only: <sku>/<group> folders under --src (default: all)")
    parser.add_argument("--src", type=Path, default=Path("data"))
    parser.add_argument("--crops", type=Path, default=Path("data_crop"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()

    if args.command == "build":
        for source in [args.src / g for g in args.groups] or crop_label.groups(args.src):
            crop_label.cut(source, args.src, args.crops)
            build(args.crops / source.relative_to(args.src), args.models)
        return
    for group in sorted(p for p in args.crops.glob("*/*") if p.name.startswith(GROUPS + ("frontview-backlight",))):
        maps, threshold = anomaly_maps(group, args.models)
        reference_mask = group_reference(group)
        print(f"== {group.relative_to(args.crops)}   on print / whole crop, missing print regions   (threshold {threshold:.3f})")
        for path, amap in maps.items():
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            on_print = float(amap[checked_area(group, gray, reference_mask)].max())
            missing = 0 if "backlight" in group.name else len(compare(gray, reference_mask, group.name.startswith("frontview"))[0])
            print(f"{on_print:.3f} / {amap.max():.3f}  missing {missing}  "
                  f"{'DEFECT' if on_print >= threshold or missing else 'ok    '}  {path.parent.name}/{path.stem}")


# Guard: DataLoader workers started with spawn/forkserver (default from Python 3.14) re-import this file.
if __name__ == "__main__":
    main()
