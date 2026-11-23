# Live view with buttons to take the checkerboard images for the plane calibration (camera/calibrate.py).
#
# Lay the board FLAT where the labels lie, its edges along the label edges. Save one image, or a
# few with the board moved around the label area: calibrate.py uses the first one and checks the
# mm accuracy on the others.
# The detected corners are drawn on the live image; an image is only saved when the board is found.
#
# Exposure and gain start at fixed values (START_EXPOSURE_US, 0 dB) with the white balance locked; the
# Exposure - / + buttons change it by hand, or type the exposure time in µs and press Set / Enter.
#
# IMPORTANT: close MVS first (exclusive access).
#
# Run:  uv run camera/calib_capture.py            (first camera)
#       uv run camera/calib_capture.py --camera 1

import argparse
import sys
import time

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from calibrate import BOARD_INNER_CORNERS, CALIB_DIR
from capture import get_float, grab_bgr
from collect import (FLASH_MS, IDLE_BORDER_COLOR, PANEL_WIDTH, SAVE_BUTTON_HEIGHT, STYLE, VIEW_WIDTH,
                     apply_camera_settings, close_camera, lock_white_balance, open_camera, to_pixmap)
from MvCameraControl_class import *

START_EXPOSURE_US = 1400.0


def measure_and_lock(camera):
    # The starting exposure and no gain, then the white balance found by the camera and locked.
    apply_camera_settings(camera, {"ExposureTime": START_EXPOSURE_US, "Gain": 0.0})
    lock_white_balance(camera)

# Corners are searched on a smaller copy so the live view stays fluid (full image = 2448 x 2048).
PREVIEW_SCALE = 0.25
FOUND_COLOR = "#2e9d5a"
NOT_FOUND_COLOR = "#d64545"


class CalibWindow(QWidget):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.latest_color = None   # last full-resolution frame, this is what gets saved
        self.board_found = False
        self.status_label = QLabel()  # before the first update_frame(), which writes to it

        self.setWindowTitle("Calibration images")
        self.setStyleSheet(STYLE)

        # Left: live view; its border turns green for a moment after a save.
        self.view_label = QLabel()
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_view_border(IDLE_BORDER_COLOR)
        self.update_frame()
        if self.latest_color is None:
            raise Exception("no frame from the camera")

        # Right: board status, save button, exposure.
        self.status_label.setObjectName("count")
        save_button = QPushButton("SAVE  ·  calibration image")
        save_button.setObjectName("support")  # the green button style of collect.py
        save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        save_button.setMinimumHeight(SAVE_BUTTON_HEIGHT)
        save_button.clicked.connect(self.save)

        self.count_label = QLabel()
        self.count_label.setObjectName("count")
        self.saved_label = QLabel("Nothing saved yet")
        self.saved_label.setWordWrap(True)

        self.camera_label = QLabel()
        exposure_down_button = QPushButton("Exposure −")
        exposure_down_button.clicked.connect(lambda: self.change_exposure(1 / 1.25))
        exposure_up_button = QPushButton("Exposure +")
        exposure_up_button.clicked.connect(lambda: self.change_exposure(1.25))
        exposure_row = QHBoxLayout()
        exposure_row.addWidget(exposure_down_button)
        exposure_row.addWidget(exposure_up_button)

        # Exposure typed in µs, limited to the range the camera accepts.
        exposure_range = MVCC_FLOATVALUE()
        self.camera["cam"].MV_CC_GetFloatValue("ExposureTime", exposure_range)
        self.exposure_box = QSpinBox()
        self.exposure_box.setRange(int(exposure_range.fMin), int(exposure_range.fMax))
        self.exposure_box.setSuffix(" µs")
        self.exposure_box.editingFinished.connect(self.set_exposure_from_box)
        set_exposure_button = QPushButton("Set")
        set_exposure_button.clicked.connect(self.set_exposure_from_box)
        exposure_input_row = QHBoxLayout()
        exposure_input_row.addWidget(self.exposure_box, stretch=1)
        exposure_input_row.addWidget(set_exposure_button)

        remeasure_button = QPushButton("Reset exposure")
        remeasure_button.clicked.connect(self.remeasure)

        folder_label = QLabel("Saving to\n%s" % CALIB_DIR)
        folder_label.setObjectName("muted")
        folder_label.setWordWrap(True)

        panel = QVBoxLayout()
        panel.addWidget(self.section_label("BOARD"))
        panel.addWidget(self.status_label)
        panel.addSpacing(12)
        panel.addWidget(save_button)
        panel.addSpacing(12)
        panel.addWidget(self.section_label("IMAGES SAVED"))
        panel.addWidget(self.count_label)
        panel.addWidget(self.saved_label)
        panel.addSpacing(12)
        panel.addWidget(self.section_label("CAMERA"))
        panel.addWidget(self.camera_label)
        panel.addLayout(exposure_row)
        panel.addLayout(exposure_input_row)
        panel.addWidget(remeasure_button)
        panel.addStretch()
        panel.addWidget(folder_label)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(PANEL_WIDTH)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        layout.addWidget(self.view_label)
        layout.addWidget(panel_widget)

        self.update_count()
        self.update_camera_info()

        # Grab the next frame whenever Qt is idle; the grab itself waits for the camera.
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(0)

    def section_label(self, text):
        label = QLabel(text)
        label.setObjectName("section")
        return label

    def set_view_border(self, color):
        self.view_label.setStyleSheet("border: 6px solid %s; border-radius: 8px;" % color)

    def update_frame(self):
        try:
            color, _ = grab_bgr(self.camera["cam"], self.camera["out_frame"])
        except Exception as e:
            print(e)
            return
        self.latest_color = color

        small = cv2.resize(color, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, interpolation=cv2.INTER_AREA)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # FAST_CHECK returns quickly when there is no board, so the view does not freeze.
        found, small_corners = cv2.findChessboardCorners(
            small_gray, BOARD_INNER_CORNERS, flags=cv2.CALIB_CB_FAST_CHECK)
        self.board_found = found

        # Draw on a copy so a saved frame stays clean.
        view = color.copy()
        if found:
            full_corners = small_corners / PREVIEW_SCALE
            cv2.drawChessboardCorners(view, BOARD_INNER_CORNERS, full_corners, found)
            self.status_label.setText("Board FOUND")
            self.status_label.setStyleSheet("color: %s;" % FOUND_COLOR)
        else:
            self.status_label.setText("Board not found")
            self.status_label.setStyleSheet("color: %s;" % NOT_FOUND_COLOR)
        self.view_label.setPixmap(to_pixmap(view, VIEW_WIDTH))

    def update_count(self):
        count = len(list(CALIB_DIR.glob("*.png")))
        self.count_label.setText("%d images" % count)

    def update_camera_info(self):
        cam = self.camera["cam"]
        exposure = get_float(cam, "ExposureTime")
        self.camera_label.setText("Exposure %.0f µs · Gain %.1f dB · locked" % (exposure, get_float(cam, "Gain")))
        # Keep the box on the real value, also after the - / + buttons or a re-measure.
        self.exposure_box.setValue(round(exposure))

    def set_exposure_from_box(self):
        cam = self.camera["cam"]
        ret = cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure_box.value()))
        if ret != 0:
            print("set exposure time fail! ret[0x%x]" % ret)
        self.update_camera_info()

    def save(self):
        if not self.board_found:
            self.show_saved_message("Board not found, NOT saved", NOT_FOUND_COLOR)
            return
        CALIB_DIR.mkdir(exist_ok=True)
        file_path = CALIB_DIR / ("calib_%s.png" % time.strftime("%Y%m%d_%H%M%S"))
        ok, png = cv2.imencode(".png", self.latest_color)
        if not ok:
            self.show_saved_message("NOT SAVED: encode png fail", NOT_FOUND_COLOR)
            return
        png.tofile(str(file_path))
        print("saved %s" % file_path)
        self.show_saved_message("Saved %s at %s" % (file_path.name, time.strftime("%H:%M:%S")), FOUND_COLOR)
        self.update_count()

        # Green border for a moment, so the save is visible on the live view too.
        self.set_view_border(FOUND_COLOR)
        QTimer.singleShot(FLASH_MS, lambda: self.set_view_border(IDLE_BORDER_COLOR))

    def show_saved_message(self, text, color):
        self.saved_label.setText(text)
        self.saved_label.setStyleSheet("color: %s; font-weight: 600;" % color)

    def change_exposure(self, factor):
        cam = self.camera["cam"]
        ret = cam.MV_CC_SetFloatValue("ExposureTime", get_float(cam, "ExposureTime") * factor)
        if ret != 0:
            print("set exposure time fail! ret[0x%x]" % ret)
        self.update_camera_info()

    def remeasure(self):
        # Blocks the window for a few seconds while auto white balance settles.
        self.camera_label.setText("Measuring... keep the scene still")
        QApplication.processEvents()
        measure_and_lock(self.camera)
        self.update_camera_info()


def main():
    parser = argparse.ArgumentParser(description="Take checkerboard images for the plane calibration.")
    parser.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    MvCamera.MV_CC_Initialize()
    camera = None
    try:
        camera = open_camera(args.camera)
        measure_and_lock(camera)
        window = CalibWindow(camera)
        window.show()
        app.exec()
    except Exception as e:
        print(e)
    finally:
        # Runs after the window is closed too, so the camera settings are always put back.
        if camera is not None:
            close_camera(camera)
        MvCamera.MV_CC_Finalize()

    print("\nDone")


if __name__ == "__main__":
    main()
