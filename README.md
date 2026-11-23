# inotex-logo-checking

Defect checks for transparent heat-transfer labels ("betway", "dafabet" printed on clear film). Every
label is photographed three times, in this order:

1. **frontview-backlight**: front, diffuse backlight. Checks dimensions, missing or extra print, and
   dirt or scratches on the film around the print.
2. **frontview-sidelight**: front, side light. Checks dirt and scratches on the letter surface.
3. **backview-sidelight**: back, side light (the label turned over). Checks dirt and scratches on the
   back of the letters.

The current data does not pair the three photos of one label (file names are capture timestamps), so
each shot is checked on its own and there is no overall OK/NG per label yet.

## Layout

```
data/<sku>/<group>/{train/good,test/good,test/defect}/*.bmp   photos; measure_settings.json and
                                                              template.png per group
data_crop/<sku>/<group>/...                                   label crops (*.png)
models/<sku>/<group>.{ckpt,json}                              AnomalyDINO checkpoint (not in git: build it); input size and threshold
src/crop_label.py          cuts the label out of every photo (data -> data_crop); build and demo call it
src/measure_check.py       backlight dimensions, in mm, against the mean of train/good
src/template_diff/         missing / added print against a good reference
src/anomaly_dino/          AnomalyDINO (DINOv2 ViT-B/14) on each whole crop, 252 px high
src/demo.py                runs all the checks above and writes one picture per test crop
app/                       the inspection app (PySide6): three steps per label, camera or saved photos
camera/                    Hikrobot camera: plane calibration, image collection (collect.py), focus
inspections/               what the app checked: photos, result pictures, inspections.csv
results/demo/              the demo pictures (not copied between machines; run the demo again)
archive/data_20261117/     the first data set (data, data_crop, models of 2026-11-17, models not in git); the results below are on it
```

`<group>` is `frontview-backlight-5000`, `frontview-sidelight-10000` or `backview-sidelight-10000`.

## Setup on a new machine

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (AnomalyDINO peaks at ~6 GB of VRAM on the largest group). The project runs on Python 3.11 (`.python-version`), the version the machine's
CUDA OpenCV build is made for; uv downloads it by itself.

```bash
git clone https://github.com/ducduy9304/inotex-logo-checking.git ~/inotex-logo-checking
cd ~/inotex-logo-checking
uv sync
./link_opencv_cuda.sh
uv run python src/anomaly_dino/anomaly_dino.py build   # the .ckpt models (~480 MB each) are not in git
```

### OpenCV

`uv sync` does **not** install OpenCV. `pyproject.toml` blocks the pip `opencv-python-headless` that
anomalib pulls in, so it does not shadow the CUDA build of OpenCV that `link_opencv_cuda.sh` links into
`.venv` (a `.pth` line to the first `~/.local/opencv/5.x-cuda*/lib/python3.11/site-packages`). Run the
script again after every `uv sync` that recreates `.venv`. The build itself is made with
`~/build_opencv_cuda.sh` (Python 3.11).

On a machine without that build, `uv pip install opencv-python-headless` works too: `crop_label.py` uses
`cv2.cuda` only when a CUDA device is found and falls back to the CPU otherwise.

### Check

```bash
uv run python -c "import cv2, torch; print('OpenCV', cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount(), '| torch CUDA', torch.cuda.is_available())"
```

The first AnomalyDINO run downloads the DINOv2 ViT-B/14 weights from Hugging Face (about 350 MB).

## New SKU or new setup

The camera tools need the Hikrobot MVS runtime and `hik_sdk/MvImport`; close MVS before starting them.

1. **Calibrate** (whenever the camera, lens, focus or label plane moves): lay the checkerboard flat where
   the labels lie, take 5 or more images with `uv run python camera/calib_capture.py`, then
   `uv run python camera/calibrate.py --square-mm 25`. It writes `camera/calibration.json` and prints
   the error in mm on the other images.
2. **Collect** with `uv run python camera/collect.py`: choose the SKU (`+ New SKU`) and the group, set
   the lights, set exposure / gain (Set, or Match train/good brightness), then save frames with
   **TRAIN · good**, **TEST · good**, **TEST · defect**. Frames are saved as the camera takes them (gray
   BMP) to `data/<sku>/<group>/<split>/<kind>/`, the group's camera settings to
   `data/<sku>/<group>/camera_settings.json` (the inspection app loads them) and the calibration to
   `data/<sku>/calibration.json` (`measure_check.py` measures that SKU with it; a SKU taken with another
   calibration refuses new images).
3. **Template**: in `collect.py`, Make template on a good photo of each group: click the label's four
   corners, check the preview reads upright, save. `--no-camera` opens `collect.py` for this only.
4. **Dimension settings** (backlight group, by hand for now): `data/<sku>/frontview-backlight-*/measure_settings.json`,
   the block boxes in px of the first train/good crop and the tolerance in mm (copy sku02's from
   `archive/data_20261117/data/sku02/` as a start).
5. `uv run python src/anomaly_dino/anomaly_dino.py build` (cuts the labels out, then builds), then
   `uv run python src/measure_check.py` (train/good spread, to set the tolerances), then the app.

## Running

Every command runs from the repository root.

### AnomalyDINO models (once, and after changing train/good)

```bash
uv run python src/anomaly_dino/anomaly_dino.py build                                    # all six groups, ~5.5 min
uv run python src/anomaly_dino/anomaly_dino.py build sku02/backview-sidelight-10000     # one group
```

Cuts the labels out of each group's photos first (`crop_label.py`: `data/` -> `data_crop/`), then builds
the group's memory bank from its train/good crops and its threshold (leave-one-out, below), all through
anomalib's `Engine`, and saves the Lightning checkpoint to `models/<sku>/<group>.ckpt` (460 to
630 MB: DINOv2's weights and the float32 memory bank) and the input size and threshold to
`models/<sku>/<group>.json`. Build a group again after adding or changing its train/good photos or its
template, or after changing the checked area.

A check loads the checkpoint once (`AnomalyDINO.load_from_checkpoint`) and calls the model on each view
in memory: `model(image)` is anomalib's forward (its pre-processor, AnomalyDINO, its post-processor), the
same maps as `engine.predict` (checked on all 26 test crops: identical) without the ~0.3 s that every
`engine.predict` call adds (a Lightning run: moving the model to the GPU and back, the visualizer writing
images, a DataLoader, view files). `TorchInferencer` (after `engine.export`) does the same call, but
anomalib 2.7 marks it legacy and it needs `TRUST_REMOTE_CODE=1` to unpickle the model.

Nothing is trained: DINOv2 ViT-B/14 stays as it is, and the memory bank is the feature of every 14 px
patch of every good crop. Each crop goes in whole, resized to 252 px high (anomalib's default height; 518 until
2026-11-20, too much GPU memory with 20+ good crops) and as wide as the group's mean crop shape allows (a
multiple of 14), so a patch covers ~33 px of the crop. Each crop goes in twice, as it is and with half a patch cut off its top-left corner,
so a small defect cut in two by a patch border in one view sits inside a patch in the other; the map is
the larger of the two. The map is the raw cosine distance from each patch to the nearest good patch,
smoothed by anomalib (Gaussian, sigma 4 px; switching it off raised the thresholds more than the defect
scores, e.g. sku02 backview 0.358 -> 0.474).

Both views are resized to the input size with OpenCV before anomalib reads them: anomalib resizes a batch
of different-sized images to its largest one first, which would stretch the smaller crops. The batch size
(`train_batch_size=8` in `fit`) can then be lowered for a smaller GPU.

### Demo (start here)

```bash
uv run python src/demo.py                                   # cuts the test photos, then -> results/demo
```

Output: `results/demo/<sku>/<n>_<group>/<kind>_<name>.jpg`, numbered in capture order. Each picture
stacks three titled panels:

- sidelight groups: the crop, the anomaly map on the print only, and the defect regions in red.
- backlight group: the dimensions (red where out of tolerance or a letter is not found), the anomaly
  map on the whole film, and the found regions (red: film defect, magenta: missing print, cyan: added
  print).

Options: `--models` (default `models`), `--out` (default `results/demo`).

`anomaly_dino.py` on the current data, highest distance on the checked area (sidelight: the letter
surface only; backlight: the whole film):

| Group | threshold | good test crops | defect test crops |
|---|---|---|---|
| sku02 backview-sidelight | 0.225 | 0.20, **0.23** | 0.49, 0.56 |
| sku02 frontview-backlight | 0.268 | 0.23, 0.23 | 0.39, 0.62 |
| sku02 frontview-sidelight | 0.285 | 0.12, 0.15 | 0.39, 0.41 |
| sku03 backview-sidelight | 0.173 | 0.14, **0.18** | 0.46, 0.47 |
| sku03 frontview-backlight | 0.372 | 0.17, 0.20 | **0.34, 0.35**, 0.47 |
| sku03 frontview-sidelight | 0.212 | 0.13, 0.19 | **0.20**, 0.46, 0.58 |

11 of the 14 defect test crops are flagged (with template_diff's missing print) and 2 of the 12 good ones
(bold: on the wrong side of the threshold):

- sku03 frontview-backlight: two defect crops (0.34, 0.35) are under the threshold (0.372), which one good
  train crop (`Image_20261117162547102`) sets alone.
- sku03 frontview-sidelight `Image_20261117161457745`: its defect is next to the letters, not on them
  (0.34 on the whole crop, 0.20 on the letter surface).
- the backview thresholds are low (0.173 from only 4 good crops for sku03), so a good crop of each
  (0.18, 0.23) is just over it.

Until 2026-11-19 the checked area was the print widened by 20 px (defects next to the letter edges
counted): 12 of 14 defects and none of the good crops flagged, with thresholds 0.27 to 0.37.

Time to check one photo of sku03 (RTX 5060 8 GB, model already loaded; loading it takes 0.8 to 1.6 s once):

| Step | frontview-backlight | frontview-sidelight | backview-sidelight |
|---|---|---|---|
| find and cut the label | 0.19 s | 0.18 s | 0.17 s |
| AnomalyDINO (two views) | 0.82 s | 0.73 s | 0.62 s |
| dimensions / letter area | 0.02 s | 0.01 s | 0.01 s |
| missing (added) print | 0.02 s | 0.02 s | 0.01 s |
| **total** | **1.04 s** | **0.94 s** | **0.82 s** |
| GPU peak | 4.8 GB | 5.7 GB | 2.7 GB |

The three shots of one label take ~2.8 s of computing.

`crop_label.find_label` searches in two steps: every angle on a 1/16 copy on the CPU (0.13 s), then
±1.5 degrees in 0.1 degree steps on a 1/2 copy, matched only within `FINE_WINDOW` px of each of the
`COARSE_CANDIDATES` (4) best different places the coarse step found (4 x 0.05 s, on the GPU when OpenCV has
CUDA; ~0.37 s per photo in all). Several places, not only the best one: on the 1/16 copy the faint backview
print can win at a wrong place (2026-11-20: 4 sku02 backview photos with the label upside down; the real
label was always the coarse step's 2nd place). Until 2026-11-18 the coarse step used a 1/8
copy and the fine step matched over the whole photo: 1.8 s, with the same boxes on all 69 photos. The GPU
only helps the fine step a little (69 vs 77 ms); for the coarse step's small images each CUDA call costs
more than the CPU match.

### Inspection app

```bash
uv run python app/label_check_app.py                 # camera if the Hikrobot SDK and a camera are found
uv run python app/label_check_app.py --no-camera     # saved photos only ("Mở ảnh")
```

Choose the SKU (its three models are loaded once), then check the label in capture order: CHECK checks
the selected step and stays on it, clicking a step selects it (to check it or see its result), and CHECK
on step 1 starts a new label. The operator only sees what each step checks:

1. **Dimensions & film**: size, missing / extra print, dirt or scratches on the film (backlight).
2. **Front print surface**: dirt, scratches, chips on the letters (sidelight).
3. **Back print surface**: the same on the back (label turned over, sidelight).

The label is OK only when all three steps are OK. Each step has its own camera exposure and gain, loaded
when the step is selected from `data/<sku>/<group>/camera_settings.json` (the settings of data
collection; without it, the exposure at the end of the group name, 5000 or 10000 us, and 0 dB). They can
be changed in the panel (Apply) and saved as the SKU's default. The lights and turning the label over are
done by hand.

- **Save image**: the photo and the result picture on screen, to `inspections/<sku>/<date>/`.
- **Save process**: the three pictures of how it was found (label or dimensions, anomaly map, defects).
- **Export data**: every check of this session to a CSV (time, SKU, label, step, verdict, score,
  threshold, faults, ms).

Without a camera, click the camera view to open a saved photo. The checks themselves are in
`app/inspection.py` (the label found and cut as `crop_label.py` does, then the functions of `src/`).

The camera needs the Hikrobot MVS runtime and its Python wrapper in `hik_sdk/MvImport` (not in this
repository); without them the app starts offline. Close MVS before starting the app (exclusive access).

### Single checks

```bash
uv run python src/measure_check.py           # dimensions, numbers only
uv run python src/anomaly_dino/anomaly_dino.py   # AnomalyDINO scores with the saved models (no pictures)
uv run python src/template_diff/template_diff.py   # missing / added print, numbers only
uv run python src/crop_label.py              # only to look at the crops: build and demo cut by themselves
```

## Dimensions in mm

`measure_check.py` finds the film edges and each block's ink box in crop pixels, as before, and turns only
the two ends of each measurement into mm: back to the photo with the crop's box (`data_crop/<sku>/<group>/
rects.json`, written by `crop_label.py`), through the lens model and onto the label plane with
`camera/calibration.json` (`camera/calibrate.py`), the same transform `camera/rectify.py` applies to whole
images. The photos and the AnomalyDINO models stay as taken; only the measurement needs the calibration.

`camera/calibration.json` must come from the setup the photos were taken with (camera, lens, focus, label
plane height). The current file is from 2026-11-02, before the current data (2026-11-17): its mm are not
checked against this data. After a new calibration, collect the data again, then run
`anomaly_dino.py build` and check the train/good spread printed by `measure_check.py` before setting the
tolerances.

## Thresholds

AnomalyDINO's threshold is set per group when the model is built, by leave-one-out on train/good (`build`
in `anomaly_dino.py`): every good crop is scored by a memory bank of the other good crops, and the highest
distance on the checked area (the letter surface for sidelight, the whole film for backlight) is the
threshold. Each leave-one-out round is its own `engine.fit`. Only good crops are used;
the test crops stay out of it. The saved memory bank holds every good crop, so a new good crop usually
scores a little below its leave-one-out score.

| Where | Value | Source |
|---|---|---|
| AnomalyDINO | per group, 0.17 to 0.37 on the current data | highest leave-one-out distance of train/good, in `models/<sku>/<group>.json` |
| dimensions | 1.0 mm | `tolerance_mm` (and per distance `tolerances_mm`) in `data/<sku>/<group>/measure_settings.json` |

Every threshold rests on very few images (4 to 10 good crops per group): one unusual good crop sets the
AnomalyDINO threshold alone. Add good crops to train/good when more data comes in, then build again.
