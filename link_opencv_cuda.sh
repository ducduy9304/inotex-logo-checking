#!/usr/bin/env bash
# Link the locally built CUDA OpenCV into .venv. Re-run whenever .venv is recreated.
set -euo pipefail
cd "$(dirname "$0")"

ls -d ~/.local/opencv/5.x-cuda*/lib/python3.11/site-packages | head -1 > .venv/lib/python3.11/site-packages/cv2_cuda.pth
.venv/bin/python -c "import cv2; print('OpenCV', cv2.__version__, '| CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
