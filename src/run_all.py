"""Everything in one run: cut the labels and build every group's model, then check every test photo.

1. Build: each group's photos are cut (crop_label.py: data -> data_crop) and its AnomalyDINO model is built
   (anomaly_dino.py build -> models/).
2. Check: every data/<sku>/<group>/test/*/ photo goes through the live app's check (app/inspection.py: find
   and cut the label, then dimensions, missing / extra print and dirt or scratches), and its verdict is
   compared with its folder: test/good should be OK, test/defect NG.

Usage: uv run python src/run_all.py [--skip-build]
Writes <out>/<sku>/<n>_<group>/<kind>_<name>.jpg (the app's process picture, its verdict in the name) and
prints one line per photo and a summary per group.
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import inspection  # noqa: E402  (puts src/ and src/anomaly_dino/ on the path)
import anomaly_dino  # noqa: E402
import crop_label  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true", help="use the data_crop/ and models/ already there")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "demo")
    args = parser.parse_args()

    if not args.skip_build:
        for source in crop_label.groups(inspection.DATA):
            crop_label.cut(source, inspection.DATA, inspection.CROPS)
            anomaly_dino.build(inspection.CROPS / source.relative_to(inspection.DATA), inspection.MODELS)

    summary, misses = [], []
    for sku in inspection.list_skus():
        for step in range(len(inspection.STEPS)):
            prepared = inspection.prepare(sku, step)
            group = prepared["group"]
            out = args.out / sku / f"{step + 1}_{group}"
            out.mkdir(parents=True, exist_ok=True)
            right = total = 0
            print(f"== {sku}/{group}   (threshold {prepared['threshold']:.3f})")
            for path in sorted((inspection.DATA / sku / group).glob("test/*/*.bmp")):
                result = inspection.check(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), prepared)
                expected = "NG" if path.parent.name == "defect" else "OK"
                ok = result["verdict"] == expected
                right, total = right + ok, total + 1
                line = (f"{result['verdict']}  {'right' if ok else 'WRONG'}  score {result['score']:.3f}  "
                        f"test/{path.parent.name}/{path.stem}  {', '.join(result['faults'])}")
                print("  " + line)
                if not ok:
                    misses.append(f"{sku}/{group}  {line}")
                cv2.imwrite(str(out / f"{path.parent.name}_{path.stem}_{result['verdict']}.jpg"), result["process"])
            summary.append(f"{sku}/{group:27s} {right:2d}/{total:2d} right")
    print("\n== summary (test/good should be OK, test/defect NG)")
    print("\n".join(summary))
    print("\n== wrong verdicts\n" + ("\n".join(misses) if misses else "none"))


# Guard: DataLoader workers started with spawn/forkserver (default from Python 3.14) re-import this file.
if __name__ == "__main__":
    main()
