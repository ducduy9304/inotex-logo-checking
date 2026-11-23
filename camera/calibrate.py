# Plane calibration: map the camera image to a straight top-down view of the label plane.
#
# Put the checkerboard FLAT where the labels lie, take images with camera/calib_capture.py (board
# moved a little between them), then run this script. Two steps:
#   1. Lens distortion (straight lines bent near the image edges), from ALL images with
#      cv2.calibrateCamera as in the OpenCV calibration tutorial, but with the image center fixed
#      and no tangential / k3 term: a flat board cannot tell the full camera model apart, the bend
#      and the focal length it can.
#   2. Homography from undistorted pixels to board millimetres, from the FIRST image.
# camera/calibration.json then holds what camera/rectify.py needs to turn a camera image into the
# rectified image (fixed scale, MM_PER_PIXEL everywhere). The other images are rectified and the
# board measured in mm on them, to check the accuracy where the first image did not cover.
#
# The board axes become the image axes of the rectified view: lay the board along the label edges.
# Redo this when the camera moves, the focus changes or the label plane changes height.
#
# Run:  uv run camera/calibrate.py --square-mm 25                  (all images in camera/calib_images/)
#       uv run camera/calibrate.py --square-mm 25 first.png other.png ...

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rectify import CALIBRATION_PATH, build_maps, rectify

CAMERA_DIR = Path(__file__).resolve().parent
CALIB_DIR = CAMERA_DIR / "calib_images"
BOARD_INNER_CORNERS = (8, 6)  # inner corners per row, per column (the board has 9 x 7 squares)
RECTIFIED_EXAMPLE_PATH = CAMERA_DIR / "calib_rectified_example.png"
MM_PER_PIXEL = 0.1
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Lens model: focal length (one value), k1 and k2 are estimated; the image center, the aspect
# ratio, the tangential terms and k3 stay fixed, since flat board images cannot pin them down.
LENS_FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
              | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)


def board_points_mm(square_mm):
    # Corner positions on the flat board in mm, row by row: (0,0), (square,0), ...
    columns, rows = BOARD_INNER_CORNERS
    points = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
    return points * square_mm


def find_corners(gray):
    found, corners = cv2.findChessboardCorners(gray, BOARD_INNER_CORNERS, None)
    if not found:
        raise Exception("board not found")
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    corners = corners.reshape(-1, 2)
    # The board looks the same turned by 180 degrees, so the corner order can start at either
    # end: start at the top-left so the rectified image is not upside down.
    first_corner = corners[0]
    last_corner = corners[-1]
    if first_corner.sum() > last_corner.sum():
        corners = np.flip(corners, axis=0)
    return corners


def detect_boards(image_paths):
    # The images where the board is found, with their corners.
    detections = []
    for image_path in image_paths:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        try:
            corners = find_corners(gray)
        except Exception as e:
            print("  %s: %s, skipped" % (image_path.name, e))
            continue
        detections.append({"path": image_path, "corners": corners})
    return detections


def estimate_lens(detections, square_mm, image_width, image_height):
    board_3d = np.zeros((len(board_points_mm(square_mm)), 3), np.float32)
    board_3d[:, :2] = board_points_mm(square_mm)
    object_points = []
    image_points = []
    for detection in detections:
        object_points.append(board_3d)
        image_points.append(detection["corners"].reshape(-1, 1, 2))
    # Starting guess only: the focal length comes out the same from any reasonable start.
    start_matrix = np.array([
        [image_width, 0, image_width / 2],
        [0, image_width, image_height / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    rms_px, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, (image_width, image_height), start_matrix, np.zeros(5), flags=LENS_FLAGS)
    return {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs.ravel(), "rms_px": float(rms_px)}


def undistort_points(points, lens):
    # Distorted camera pixels -> pixels of an ideal lens with the same camera matrix.
    undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64), lens["camera_matrix"],
                                      lens["dist_coeffs"], P=lens["camera_matrix"])
    return undistorted.reshape(-1, 2)


def rectified_frame(pixel_to_mm, lens, image_width, image_height):
    # Where the whole camera image lands on the plane (mm): the rectified image covers that box.
    image_corners = np.float32([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]])
    undistorted_corners = undistort_points(image_corners, lens)
    corners_mm = cv2.perspectiveTransform(undistorted_corners.reshape(-1, 1, 2), pixel_to_mm).reshape(-1, 2)
    min_mm = corners_mm.min(axis=0)
    max_mm = corners_mm.max(axis=0)
    width = int(np.ceil((max_mm[0] - min_mm[0]) / MM_PER_PIXEL))
    height = int(np.ceil((max_mm[1] - min_mm[1]) / MM_PER_PIXEL))
    # mm -> rectified pixel: shift the box to (0, 0), then divide by the scale.
    mm_to_rectified = np.array([
        [1 / MM_PER_PIXEL, 0, -min_mm[0] / MM_PER_PIXEL],
        [0, 1 / MM_PER_PIXEL, -min_mm[1] / MM_PER_PIXEL],
        [0, 0, 1],
    ])
    return {"mm_to_rectified": mm_to_rectified, "width": width, "height": height}


def board_size_mm(rectified, square_mm):
    # Board width and height measured on a rectified image, from the outer inner corners.
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    columns, rows = BOARD_INNER_CORNERS
    corners = find_corners(gray).reshape(rows, columns, 2)
    width_px = np.linalg.norm(corners[:, -1] - corners[:, 0], axis=1).mean()
    height_px = np.linalg.norm(corners[-1, :] - corners[0, :], axis=1).mean()
    return {"width": width_px * MM_PER_PIXEL, "height": height_px * MM_PER_PIXEL}


def check_images(image_paths, calibration, square_mm):
    # Rectify each image and measure the board: it should come out at its true size everywhere.
    columns, rows = BOARD_INNER_CORNERS
    true_width = (columns - 1) * square_mm
    true_height = (rows - 1) * square_mm
    print("check on %d other images (true board %.1f x %.1f mm):" % (len(image_paths), true_width, true_height))
    worst_error = 0.0
    for image_path in image_paths:
        rectified = rectify(cv2.imread(str(image_path)), calibration)
        try:
            size = board_size_mm(rectified, square_mm)
        except Exception as e:
            print("  %s: %s, skipped" % (image_path.name, e))
            continue
        width_error = size["width"] - true_width
        height_error = size["height"] - true_height
        worst_error = max(worst_error, abs(width_error), abs(height_error))
        print("  %s: %.2f x %.2f mm (error %+.2f, %+.2f)"
              % (image_path.name, size["width"], size["height"], width_error, height_error))
    print("worst error on the other images: %.2f mm" % worst_error)


def main():
    parser = argparse.ArgumentParser(description="Plane calibration from flat checkerboard images.")
    parser.add_argument("images", type=Path, nargs="*",
                        help="images with the board flat on the label plane; all are used for the lens, the "
                             "first one for the plane, the others to check it (default: all of camera/calib_images/)")
    parser.add_argument("--square-mm", type=float, required=True, help="side of one square in mm")
    args = parser.parse_args()

    image_paths = args.images
    if not image_paths:
        image_paths = sorted(CALIB_DIR.glob("*.png"))
    if not image_paths:
        raise SystemExit("no image: take some with camera/calib_capture.py first")
    detections = detect_boards(image_paths)
    if len(detections) < 5:
        raise SystemExit("board found in %d images only: take at least 5" % len(detections))
    first = detections[0]
    image_height, image_width = cv2.imread(str(first["path"]), cv2.IMREAD_GRAYSCALE).shape
    print("calibration image: %s (lens from %d images)" % (first["path"].name, len(detections)))

    lens = estimate_lens(detections, args.square_mm, image_width, image_height)
    board_mm = board_points_mm(args.square_mm)
    corners = undistort_points(first["corners"], lens)

    # Native resolution: how many mm one camera pixel covers on the label plane.
    columns, rows = BOARD_INNER_CORNERS
    grid = corners.reshape(rows, columns, 2)
    row_width_px = np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1).mean()
    camera_mm_per_pixel = float((columns - 1) * args.square_mm / row_width_px)

    pixel_to_mm, _ = cv2.findHomography(corners, board_mm)

    # Fit error in mm: how far the detected corners land from their true board position.
    corners_mm = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), pixel_to_mm).reshape(-1, 2)
    distances_mm = np.linalg.norm(corners_mm - board_mm, axis=1)
    rms_error_mm = float(np.sqrt(np.mean(distances_mm ** 2)))
    max_error_mm = float(distances_mm.max())

    frame = rectified_frame(pixel_to_mm, lens, image_width, image_height)
    camera_to_rectified = frame["mm_to_rectified"] @ pixel_to_mm

    result = {
        "source_image": first["path"].name,
        "lens_image_count": len(detections),
        "board_inner_corners": list(BOARD_INNER_CORNERS),
        "square_mm": args.square_mm,
        "mm_per_pixel": MM_PER_PIXEL,
        "camera_mm_per_pixel": camera_mm_per_pixel,
        "camera_image_size": [image_width, image_height],
        "rectified_size": [frame["width"], frame["height"]],
        "camera_matrix": lens["camera_matrix"].tolist(),
        "dist_coeffs": lens["dist_coeffs"].tolist(),
        "camera_to_rectified": camera_to_rectified.tolist(),
        "lens_fit_rms_px": lens["rms_px"],
        "fit_rms_error_mm": rms_error_mm,
        "fit_max_error_mm": max_error_mm,
    }
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2))

    print("lens: focal %.0f px, k1 %.4f, k2 %.4f, fit rms %.3f px"
          % (lens["camera_matrix"][0, 0], lens["dist_coeffs"][0], lens["dist_coeffs"][1], lens["rms_px"]))
    print("camera resolution on the label plane: %.4f mm/px" % camera_mm_per_pixel)
    print("fit error: rms %.3f mm, max %.3f mm" % (rms_error_mm, max_error_mm))
    print("rectified image: %d x %d px at %.2f mm/px" % (frame["width"], frame["height"], MM_PER_PIXEL))
    print("saved to %s" % CALIBRATION_PATH)

    # Same arrays as rectify.load_calibration() gives, without reading the file back.
    calibration = {
        "camera_matrix": lens["camera_matrix"],
        "dist_coeffs": lens["dist_coeffs"],
        "camera_to_rectified": camera_to_rectified,
        "rectified_size": [frame["width"], frame["height"]],
    }
    build_maps(calibration)
    rectified = rectify(cv2.imread(str(first["path"])), calibration)
    cv2.imwrite(str(RECTIFIED_EXAMPLE_PATH), rectified)
    print("rectified example: %s" % RECTIFIED_EXAMPLE_PATH)

    other_paths = []
    for detection in detections[1:]:
        other_paths.append(detection["path"])
    if other_paths:
        check_images(other_paths, calibration, args.square_mm)


if __name__ == "__main__":
    main()
