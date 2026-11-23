# Straighten camera images to a top-down view of the label plane (fixed mm per pixel).
#
# Uses camera/calibration.json from camera/calibrate.py: the lens distortion (camera matrix +
# distortion coefficients) and the homography from undistorted pixels to the rectified image.
# Both are applied in ONE remap, so the image is interpolated only once. In the rectified image a
# distance in pixels times mm_per_pixel (0.1) is the distance in mm on the label plane.
#
# Run:  uv run camera/rectify.py image.png [more.png ...]     (writes image_rectified.png next to each)
#
# From other code:  from rectify import load_calibration, rectify

import json
import sys
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


def build_maps(calibration):
    # For every rectified pixel: where to read it in the camera image. initUndistortRectifyMap
    # takes a rectified pixel back through inverse(new_matrix) to undistorted lens coordinates,
    # then adds the lens distortion; new_matrix = camera_to_rectified @ camera_matrix makes that
    # first step undo our homography.
    camera_matrix = calibration["camera_matrix"]
    new_matrix = calibration["camera_to_rectified"] @ camera_matrix
    width, height = calibration["rectified_size"]
    map_fixed, map_fraction = cv2.initUndistortRectifyMap(
        camera_matrix, calibration["dist_coeffs"], None, new_matrix, (width, height), cv2.CV_16SC2)
    calibration["map_fixed"] = map_fixed
    calibration["map_fraction"] = map_fraction


def load_calibration(path=CALIBRATION_PATH):
    calibration = json.loads(Path(path).read_text())
    calibration["camera_matrix"] = np.array(calibration["camera_matrix"])
    calibration["dist_coeffs"] = np.array(calibration["dist_coeffs"])
    calibration["camera_to_rectified"] = np.array(calibration["camera_to_rectified"])
    # Built once here, so rectifying a frame is a single fast remap.
    build_maps(calibration)
    return calibration


def rectify(image, calibration):
    return cv2.remap(image, calibration["map_fixed"], calibration["map_fraction"], cv2.INTER_LINEAR)


def main():
    calibration = load_calibration()
    for name in sys.argv[1:]:
        image_path = Path(name)
        image = cv2.imread(str(image_path))
        rectified = rectify(image, calibration)
        output_path = image_path.with_name(image_path.stem + "_rectified.png")
        cv2.imwrite(str(output_path), rectified)
        print("saved %s (%.2f mm/px)" % (output_path, calibration["mm_per_pixel"]))


if __name__ == "__main__":
    main()
