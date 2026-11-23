# Collect the label photos the checks learn from: live view with buttons to save each frame into
# data/<sku>/<group>/{train/good, test/good, test/defect}/, the layout src/crop_label.py, src/anomaly_dino
# and app/ read.
#
# A label is photographed in three groups, each with its own lights and camera settings:
#   frontview-backlight-5000    Dimensions & film (backlight)
#   frontview-sidelight-10000   Front print surface (side light)
#   backview-sidelight-10000    Back print surface (side light, label turned over)
# The number at the end of a group is its starting exposure in µs. Exposure and gain are FIXED per group
# (no auto exposure: a brightness change between a good image and a checked one would itself look like a
# defect) and saved with every image to data/<sku>/<group>/camera_settings.json; app/label_check_app.py
# loads the same file, so the inspection uses the settings the images were collected with. White balance
# is measured once at start (auto) and locked, as the frames are turned to gray.
#
# Images are saved as the camera takes them (not rectified), gray, lossless BMP named Image_<time>.bmp:
# the anomaly models learn from them as they are, and the dimension check turns its measurements into mm
# with the plane calibration (camera/calibrate.py). The calibration is copied to data/<sku>/calibration.json
# with the first image; images of one SKU must all come from one setup, so saving into a SKU taken with
# another calibration is refused.
#
# Make template: open a good photo of the group, click the label's four corners (top-left, top-right,
# bottom-right, bottom-left, with the print upright; mirrored on the back view), and save it straight as
# data/<sku>/<group>/template.png, what crop_label.py finds the label with. Works without a camera.
#
# After collecting: crop_label.py, anomaly_dino.py build, measure_check.py (see README).
#
# IMPORTANT: close MVS first (exclusive access). Keep the lights, the camera position and the focus fixed
# for the whole session (and later for inspection).
#
# Run:  uv run python camera/collect.py                    (starts on the last SKU)
#       uv run python camera/collect.py --sku sku02 --camera 1
#       uv run python camera/collect.py --no-camera        (Make template only)
import argparse
import json
import re
import shutil
import sys
import time
from ctypes import *
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crop_label  # noqa: E402
from rectify import CALIBRATION_PATH as PLANE_CALIBRATION_PATH  # noqa: E402

try:  # the Hikrobot SDK (hik_sdk/MvImport) is only on the camera PC
    from capture import GRAB_TIMEOUT_MS, get_enum, get_float, grab_bgr, restore
    from MvCameraControl_class import *
    CAMERA_IMPORT_ERROR = None
except ImportError as error:
    CAMERA_IMPORT_ERROR = error

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
SKU_PATTERN = re.compile(r"^sku(\d+)$")
GROUPS = {   # group folder -> what it checks, with which light
    "frontview-backlight-5000": "Dimensions & film (backlight)",
    "frontview-sidelight-10000": "Front print (side light)",
    "backview-sidelight-10000": "Back print (side light)",
}
VIEW_WIDTH = 1000        # live view width in pixels, the height follows the image aspect ratio
PANEL_WIDTH = 400
THUMBNAIL_WIDTH = 160    # width of the "last saved" preview
FLASH_MS = 400           # how long the colored border stays after a save
SAVE_BUTTON_HEIGHT = 44
THUMBNAIL_BORDER = 3
IDLE_BORDER_COLOR = "#2b2d31"
SETTINGS_FILE_NAME = "camera_settings.json"   # per group, written at every save
CALIBRATION_FILE_NAME = "calibration.json"    # per SKU, copy of camera/calibration.json
WHITE_BALANCE_CHANNELS = {"Red": 0, "Green": 1, "Blue": 2}  # BalanceRatioSelector values
WB_WARMUP_FRAMES = 30     # frames auto white balance gets before it is locked
BRIGHTNESS_SAMPLE_WIDTH = 640  # images are shrunk to this width before measuring label brightness
MATCH_STEPS = 4                # exposure corrections made by "Match train/good brightness"
TEMPLATE_VIEW_WIDTH = 1100     # width of the photo in the Make template window
# The kinds of image: which folder they go to, button color and text.
SAVE_KINDS = {
    "train_good": {"folder": "train/good", "color": "#2e9d5a", "hover": "#36b368", "text": "TRAIN  ·  good"},
    "test_good": {"folder": "test/good", "color": "#2f6fd6", "hover": "#4383ea", "text": "TEST  ·  good"},
    "test_defect": {"folder": "test/defect", "color": "#d64545", "hover": "#e55a5a", "text": "TEST  ·  defect"},
}

STYLE = """
QWidget { background: #1e1f22; color: #e8e8e8; font-size: 15px; }
QPushButton {
    background: #3a3d43; border: none; border-radius: 10px; padding: 6px; font-weight: 600;
}
QPushButton:hover { background: #4a4e55; }
QPushButton:pressed { background: #2c2e33; }
QPushButton:disabled { color: #6b6f77; }
QComboBox {
    background: #2b2d31; border: 1px solid #4a4e55; border-radius: 8px; padding: 8px 12px;
    font-size: 18px; font-weight: 700;
}
QComboBox QAbstractItemView { background: #2b2d31; selection-background-color: #4a4e55; }
QDoubleSpinBox {
    background: #2b2d31; border: 1px solid #4a4e55; border-radius: 8px; padding: 8px 10px;
    font-size: 16px;
}
QLabel#count { font-size: 16px; font-weight: 700; }
QLabel#muted { color: #8b8f97; font-size: 13px; }
QLabel#section { color: #8b8f97; font-size: 13px; font-weight: 700; letter-spacing: 1px; }
"""
for kind, settings in SAVE_KINDS.items():
    STYLE += "QPushButton#%s { background: %s; font-size: 18px; }\n" % (kind, settings["color"])
    STYLE += "QPushButton#%s:hover { background: %s; }\n" % (kind, settings["hover"])


def open_camera(camera_index):
    # Same steps as capture.py: enumerate, open exclusive, continuous grabbing.
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != 0:
        raise Exception("enum devices fail! ret[0x%x]" % ret)
    if device_list.nDeviceNum == 0:
        raise Exception("find no device!")
    if camera_index >= device_list.nDeviceNum:
        raise Exception("camera %d not found, only %d camera(s)" % (camera_index, device_list.nDeviceNum))
    info = cast(device_list.pDeviceInfo[camera_index], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(info)
    if ret != 0:
        raise Exception("create handle fail! ret[0x%x]" % ret)
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        cam.MV_CC_DestroyHandle()
        raise Exception("open device fail! ret[0x%x]" % ret)

    if info.nTLayerType == MV_GIGE_DEVICE:
        packet_size = cam.MV_CC_GetOptimalPacketSize()
        if int(packet_size) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
            if ret != 0:
                print("Warning: set packet size fail! ret[0x%x]" % ret)
        else:
            print("Warning: get packet size fail! ret[0x%x]" % packet_size)

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    # Color data, turned to gray by the caller: the same path here and in app/label_check_app.py, so the
    # collected images and the checked frames are made the same way (MVS may leave the camera in Mono8).
    ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)
    if ret != 0:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        raise Exception("set PixelFormat BayerRG8 fail! ret[0x%x]" % ret)

    white_balance = MVCC_ENUMVALUE()
    has_wb = cam.MV_CC_GetEnumValue("BalanceWhiteAuto", white_balance) == 0  # mono cameras have none
    original_settings = {
        "ExposureAuto": get_enum(cam, "ExposureAuto"),
        "GainAuto": get_enum(cam, "GainAuto"),
        "BalanceWhiteAuto": white_balance.nCurValue if has_wb else None,
        "ExposureTime": get_float(cam, "ExposureTime"),
        "Gain": get_float(cam, "Gain"),
    }

    cam.MV_CC_SetBayerCvtQuality(1)
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        restore(cam, original_settings)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        raise Exception("start grabbing fail! ret[0x%x]" % ret)

    out_frame = MV_FRAME_OUT()
    memset(byref(out_frame), 0, sizeof(out_frame))
    return {"cam": cam, "out_frame": out_frame, "has_wb": has_wb, "original_settings": original_settings}


def close_camera(camera):
    # Stop, put the camera back the way we found it, close.
    cam = camera["cam"]
    cam.MV_CC_StopGrabbing()
    restore(cam, camera["original_settings"])
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()


def grab_gray(camera):
    color, _ = grab_bgr(camera["cam"], camera["out_frame"])
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)


def lock_white_balance(camera):
    # Let the camera find the white balance, then freeze it.
    if not camera["has_wb"]:
        return
    cam = camera["cam"]
    print("measuring white balance, keep the scene still...")
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)
    cam.MV_CC_ClearImageBuffer()
    for _ in range(WB_WARMUP_FRAMES):
        if cam.MV_CC_GetImageBuffer(camera["out_frame"], GRAB_TIMEOUT_MS) == 0:
            cam.MV_CC_FreeImageBuffer(camera["out_frame"])
    # Off keeps the current values (they stay what auto found).
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_OFF)


def read_camera_settings(camera):
    # The values that decide how bright the images are.
    cam = camera["cam"]
    settings = {"ExposureTime": get_float(cam, "ExposureTime"), "Gain": get_float(cam, "Gain")}
    if camera["has_wb"]:
        settings["BalanceRatio"] = {}
        for channel, selector in WHITE_BALANCE_CHANNELS.items():
            cam.MV_CC_SetEnumValue("BalanceRatioSelector", selector)
            ratio = MVCC_INTVALUE_EX()
            if cam.MV_CC_GetIntValueEx("BalanceRatio", ratio) == 0:
                settings["BalanceRatio"][channel] = ratio.nCurValue
    return settings


def apply_camera_settings(camera, settings):
    # Auto must be Off first, or the camera overwrites the values right away.
    cam = camera["cam"]
    cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
    cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
    cam.MV_CC_SetFloatValue("ExposureTime", settings["ExposureTime"])
    cam.MV_CC_SetFloatValue("Gain", settings["Gain"])
    if camera["has_wb"]:
        cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_OFF)
        for channel, value in settings.get("BalanceRatio", {}).items():
            cam.MV_CC_SetEnumValue("BalanceRatioSelector", WHITE_BALANCE_CHANNELS[channel])
            ret = cam.MV_CC_SetIntValueEx("BalanceRatio", value)
            if ret != 0:
                print("set white balance %s fail! ret[0x%x]" % (channel, ret))


def list_skus():
    # SKU folders sorted by number: sku01, sku02, ...
    return sorted((p.name for p in DATA_ROOT.iterdir() if p.is_dir() and SKU_PATTERN.match(p.name)),
                  key=lambda name: int(SKU_PATTERN.match(name).group(1)))


def next_sku_name():
    numbers = [int(SKU_PATTERN.match(name).group(1)) for name in list_skus()]
    return "sku%02d" % (max(numbers, default=0) + 1)


def group_folder(sku, group):
    return DATA_ROOT / sku / group


def kind_images(sku, group, kind):
    return sorted((group_folder(sku, group) / SAVE_KINDS[kind]["folder"]).glob("*.bmp"))


def group_settings(sku, group):
    # The group's saved camera settings, or its starting exposure (the number at the end of its name), no gain.
    path = group_folder(sku, group) / SETTINGS_FILE_NAME
    if path.is_file():
        return json.loads(path.read_text())
    return {"ExposureTime": float(group.rsplit("-", 1)[1]), "Gain": 0.0}


def save_group_settings(sku, group, settings):
    group_folder(sku, group).mkdir(parents=True, exist_ok=True)
    (group_folder(sku, group) / SETTINGS_FILE_NAME).write_text(json.dumps(settings, indent=2))


def save_bmp(gray, file_path):
    # Raises if the file is not on disk afterwards.
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # imencode + tofile instead of imwrite: works even if the path has non-ASCII characters.
    ok, bmp = cv2.imencode(".bmp", gray)
    if not ok:
        raise Exception("encode bmp fail!")
    bmp.tofile(str(file_path))
    if not file_path.is_file() or file_path.stat().st_size == 0:
        raise Exception("file not written: %s" % file_path)


def save_image(gray, sku, group, kind):
    # Image_<date><time><ms>.bmp in the kind's folder, as MVS names them. Returns its path.
    now = time.time()
    name = "Image_%s%03d.bmp" % (time.strftime("%Y%m%d%H%M%S", time.localtime(now)), int(now * 1000) % 1000)
    path = group_folder(sku, group) / SAVE_KINDS[kind]["folder"] / name
    save_bmp(gray, path)
    return path


def save_sku_calibration(sku, calibration_path):
    shutil.copyfile(calibration_path, DATA_ROOT / sku / CALIBRATION_FILE_NAME)


def sku_calibration_matches(sku, calibration_path):
    # A SKU keeps the calibration of its first image: images of one SKU must all come from one setup.
    sku_calibration_path = DATA_ROOT / sku / CALIBRATION_FILE_NAME
    return not sku_calibration_path.is_file() or sku_calibration_path.read_bytes() == calibration_path.read_bytes()


def label_brightness(gray):
    # Mean gray level of the bright part of the image (Otsu): the label under side light, the panel under
    # backlight.
    scale = BRIGHTNESS_SAMPLE_WIDTH / gray.shape[1]
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.mean(small, mask=mask)[0]


def train_brightness(sku, group):
    # Median label brightness of the group's train/good images, or None if it has none.
    values = [label_brightness(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in kind_images(sku, group, "train_good")]
    return float(np.median(values)) if values else None


def to_pixmap(image, width):
    # Gray or BGR numpy image -> Qt pixmap of the given width.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height = image.shape[0] * width // image.shape[1]
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    qimage = QImage(small.data, width, height, small.strides[0], QImage.Format.Format_BGR888)
    # fromImage copies the pixels, so `small` may be freed afterwards.
    return QPixmap.fromImage(qimage)


def rect_from_corners(points):
    # The label's four corners (top-left, top-right, bottom-right, bottom-left, print upright) -> its box as
    # crop_label uses it: center, (width, height), angle.
    p = np.asarray(points, float)
    top, bottom = p[1] - p[0], p[2] - p[3]
    left, right = p[3] - p[0], p[2] - p[1]
    width = (np.linalg.norm(top) + np.linalg.norm(bottom)) / 2
    height = (np.linalg.norm(left) + np.linalg.norm(right)) / 2
    direction = top + bottom
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return (float(p[:, 0].mean()), float(p[:, 1].mean())), (round(width), round(height)), angle


class ClickableImage(QLabel):
    clicked = Signal(float, float)   # position in the shown pixmap

    def mousePressEvent(self, event):
        self.clicked.emit(event.position().x(), event.position().y())


class TemplateDialog(QDialog):
    # A good photo of the group: click the label's four corners, look at the straight crop, save it as
    # the group's template.png.
    CORNERS = ("top-left", "top-right", "bottom-right", "bottom-left")

    def __init__(self, parent, photo_path, group_dir):
        super().__init__(parent)
        self.photo = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
        self.group_dir = group_dir
        self.scale = TEMPLATE_VIEW_WIDTH / self.photo.shape[1]
        self.points = []
        self.crop = None
        self.setWindowTitle("Make template: %s / %s" % (group_dir.parent.name, group_dir.name))
        self.setStyleSheet(STYLE)

        self.view = ClickableImage()
        self.view.clicked.connect(self.add_point)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        self.save_button = QPushButton("Save template")
        self.save_button.clicked.connect(self.save)
        buttons = QHBoxLayout()
        buttons.addWidget(reset_button)
        buttons.addWidget(self.save_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.hint)
        layout.addWidget(self.view)
        layout.addWidget(self.preview)
        layout.addLayout(buttons)
        self.reset()

    def reset(self):
        self.points = []
        self.crop = None
        self.preview.clear()
        self.save_button.setEnabled(False)
        self.redraw()

    def redraw(self):
        picture = cv2.cvtColor(self.photo, cv2.COLOR_GRAY2BGR)
        for number, point in enumerate(self.points):
            cv2.circle(picture, (int(point[0]), int(point[1])), 18, (0, 0, 255), -1)
            cv2.putText(picture, str(number + 1), (int(point[0]) + 22, int(point[1]) - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        if len(self.points) > 1:
            cv2.polylines(picture, [np.int32(self.points)], len(self.points) == 4, (0, 0, 255), 6)
        self.view.setPixmap(to_pixmap(picture, TEMPLATE_VIEW_WIDTH))
        if len(self.points) < 4:
            mirrored = " (the print reads mirrored here)" if crop_label.mirrored(self.group_dir.name) else ""
            self.hint.setText("Click the label's %s corner, with the print upright%s."
                              % (self.CORNERS[len(self.points)], mirrored))
        else:
            self.hint.setText("Below: the label as crop_label.py will read it. It must read upright and not "
                              "mirrored; if not, Reset and start from another corner.")

    def add_point(self, x, y):
        if len(self.points) == 4:
            return
        self.points.append((x / self.scale, y / self.scale))
        if len(self.points) == 4:
            self.crop = crop_label.crop_rect(self.photo, rect_from_corners(self.points))
            shown = cv2.flip(self.crop, 1) if crop_label.mirrored(self.group_dir.name) else self.crop
            self.preview.setPixmap(to_pixmap(shown, TEMPLATE_VIEW_WIDTH // 2))
            self.save_button.setEnabled(True)
        self.redraw()

    def save(self):
        path = self.group_dir / crop_label.TEMPLATE_NAME
        if path.exists():
            answer = QMessageBox.question(self, "Make template", "Replace %s?" % path)
            if answer != QMessageBox.StandardButton.Yes:
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.crop)
        print("saved %s (%d x %d)" % (path, self.crop.shape[1], self.crop.shape[0]))
        self.accept()


class CollectWindow(QWidget):
    def __init__(self, camera, calibration_path, start_sku):
        super().__init__()
        self.camera = camera
        self.calibration_path = calibration_path
        self.latest = None             # last full-resolution gray frame, what gets saved
        self.target_brightness = None  # label brightness of the current group's train/good images

        self.setWindowTitle("Collect label images")
        self.setStyleSheet(STYLE)

        # Left: live view; its border turns the button color for a moment after a save.
        self.view_label = QLabel("No camera: Make template only" if camera is None else "")
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_label.setMinimumSize(VIEW_WIDTH, VIEW_WIDTH * 2048 // 2448)
        self.set_view_border(IDLE_BORDER_COLOR)

        # Right: SKU, group, save buttons, camera, template.
        self.sku_box = QComboBox()
        self.sku_box.addItems(list_skus())
        self.sku_box.setCurrentText(start_sku)
        new_sku_button = QPushButton("+ New SKU")
        new_sku_button.clicked.connect(self.add_sku)
        sku_row = QHBoxLayout()
        sku_row.addWidget(self.sku_box, stretch=1)
        sku_row.addWidget(new_sku_button)
        self.group_box = QComboBox()
        for group, name in GROUPS.items():
            self.group_box.addItem(name, group)

        self.save_buttons = []
        for kind, settings in SAVE_KINDS.items():
            button = QPushButton(settings["text"])
            button.setObjectName(kind)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(SAVE_BUTTON_HEIGHT)
            # kind=kind: bind the current value, not the loop variable.
            button.clicked.connect(lambda checked=False, kind=kind: self.save_to(kind))
            button.setEnabled(camera is not None)
            self.save_buttons.append(button)

        self.count_label = QLabel()
        self.count_label.setObjectName("count")
        self.saved_label = QLabel("Nothing saved yet")
        self.saved_label.setWordWrap(True)
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(THUMBNAIL_WIDTH + 2 * THUMBNAIL_BORDER,
                                          THUMBNAIL_WIDTH * 2048 // 2448 + 2 * THUMBNAIL_BORDER)
        self.thumbnail_label.setStyleSheet("border: %dpx solid %s; border-radius: 6px;"
                                           % (THUMBNAIL_BORDER, IDLE_BORDER_COLOR))

        self.camera_label = QLabel("No camera" if camera is None else "")
        self.brightness_label = QLabel()
        self.exposure_box = QDoubleSpinBox()
        self.exposure_box.setRange(20, 1000000)
        self.exposure_box.setDecimals(0)
        self.exposure_box.setSingleStep(500)
        self.exposure_box.setSuffix(" µs")
        self.gain_box = QDoubleSpinBox()
        self.gain_box.setRange(0, 24)
        self.gain_box.setDecimals(1)
        self.gain_box.setSingleStep(0.5)
        self.gain_box.setSuffix(" dB")
        set_button = QPushButton("Set")
        set_button.clicked.connect(self.set_from_boxes)
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Exposure"))
        settings_row.addWidget(self.exposure_box, stretch=1)
        settings_row.addWidget(QLabel("Gain"))
        settings_row.addWidget(self.gain_box, stretch=1)
        settings_row.addWidget(set_button)
        match_button = QPushButton("Match train/good brightness")
        match_button.clicked.connect(self.match_brightness)
        remeasure_button = QPushButton("Re-measure white balance")
        remeasure_button.clicked.connect(self.remeasure)
        for widget in (self.exposure_box, self.gain_box, set_button, match_button, remeasure_button):
            widget.setEnabled(camera is not None)
        self.settings_label = QLabel()
        self.settings_label.setObjectName("muted")
        self.settings_label.setWordWrap(True)

        template_button = QPushButton("Make template")
        template_button.clicked.connect(self.make_template)
        self.template_label = QLabel()
        self.template_label.setObjectName("muted")
        self.folder_label = QLabel()
        self.folder_label.setObjectName("muted")
        self.folder_label.setWordWrap(True)

        panel = QVBoxLayout()
        panel.addWidget(self.section_label("SKU"))
        panel.addLayout(sku_row)
        panel.addWidget(self.section_label("GROUP"))
        panel.addWidget(self.group_box)
        panel.addSpacing(8)
        for button in self.save_buttons:
            panel.addWidget(button)
        panel.addSpacing(8)
        panel.addWidget(self.section_label("IMAGES IN THIS GROUP"))
        panel.addWidget(self.count_label)
        panel.addWidget(self.section_label("LAST SAVED"))
        panel.addWidget(self.saved_label)
        # Left-aligned so the label (and its colored border) is only as wide as the thumbnail.
        panel.addWidget(self.thumbnail_label, alignment=Qt.AlignmentFlag.AlignLeft)
        panel.addSpacing(8)
        panel.addWidget(self.section_label("CAMERA"))
        panel.addWidget(self.camera_label)
        panel.addWidget(self.brightness_label)
        panel.addLayout(settings_row)
        panel.addWidget(match_button)
        panel.addWidget(remeasure_button)
        panel.addWidget(self.settings_label)
        panel.addSpacing(8)
        panel.addWidget(self.section_label("TEMPLATE"))
        panel.addWidget(template_button)
        panel.addWidget(self.template_label)
        panel.addStretch()
        panel.addWidget(self.folder_label)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(PANEL_WIDTH)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        layout.addWidget(self.view_label, stretch=1)
        layout.addWidget(panel_widget)

        self.sku_box.currentTextChanged.connect(self.on_group_changed)
        self.group_box.currentIndexChanged.connect(self.on_group_changed)
        self.on_group_changed()

        if camera is not None:
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

    def current_sku(self):
        return self.sku_box.currentText()

    def current_group(self):
        return self.group_box.currentData()

    def on_group_changed(self, *_):
        # The group's camera settings (saved, or its starting exposure), counts and template.
        sku, group = self.current_sku(), self.current_group()
        folder = group_folder(sku, group)
        self.folder_label.setText("Saving to\n%s" % folder)
        self.show_saved_message("Now saving to %s / %s" % (sku, group), "#e8e8e8")
        self.thumbnail_label.clear()
        self.update_counts()
        has_template = (folder / crop_label.TEMPLATE_NAME).is_file()
        self.template_label.setText("template.png: %s" % ("yes" if has_template else "not made yet"))
        settings = group_settings(sku, group)
        saved = (folder / SETTINGS_FILE_NAME).is_file()
        self.settings_label.setText("Camera settings %s" % ("loaded from %s/%s/%s" % (sku, group, SETTINGS_FILE_NAME)
                                                             if saved else "of a new group: saved with the first image"))
        if self.camera is None:
            self.exposure_box.setValue(settings["ExposureTime"])
            self.gain_box.setValue(settings["Gain"])
        else:
            apply_camera_settings(self.camera, settings)
            self.target_brightness = train_brightness(sku, group)
            self.update_camera_info()

    def add_sku(self):
        # Ask first: a click by mistake would otherwise leave an empty SKU folder.
        name = next_sku_name()
        answer = QMessageBox.question(self, "New SKU", "Create %s and save the next images there?" % name)
        if answer != QMessageBox.StandardButton.Yes:
            return
        for group in GROUPS:
            group_folder(name, group).mkdir(parents=True, exist_ok=True)
        self.sku_box.addItem(name)
        self.sku_box.setCurrentText(name)

    def update_counts(self):
        sku, group = self.current_sku(), self.current_group()
        parts = ["<span style='color:%s'>%s %d</span>" % (s["color"], s["text"].replace("  ·  ", " "),
                                                           len(kind_images(sku, group, kind)))
                 for kind, s in SAVE_KINDS.items()]
        self.count_label.setText("&nbsp;&nbsp;".join(parts))

    def update_camera_info(self):
        cam = self.camera["cam"]
        exposure, gain = get_float(cam, "ExposureTime"), get_float(cam, "Gain")
        self.camera_label.setText("Exposure %.0f µs · Gain %.1f dB · locked" % (exposure, gain))
        # Keep the boxes on the real values, also after a match or a group change.
        self.exposure_box.setValue(exposure)
        self.gain_box.setValue(gain)

    def set_from_boxes(self):
        cam = self.camera["cam"]
        for name, box in (("ExposureTime", self.exposure_box), ("Gain", self.gain_box)):
            ret = cam.MV_CC_SetFloatValue(name, float(box.value()))
            if ret != 0:
                print("set %s fail! ret[0x%x]" % (name, ret))
        self.update_camera_info()

    def update_frame(self):
        try:
            gray = grab_gray(self.camera)
        except Exception as e:
            print(e)
            return
        self.latest = gray
        self.view_label.setPixmap(to_pixmap(gray, VIEW_WIDTH))
        brightness = label_brightness(gray)
        if self.target_brightness is None:
            self.brightness_label.setText("Brightness %.0f" % brightness)
        else:
            self.brightness_label.setText("Brightness %.0f · train/good %.0f" % (brightness, self.target_brightness))

    def save_to(self, kind):
        error_color = SAVE_KINDS["test_defect"]["color"]
        sku, group = self.current_sku(), self.current_group()
        if self.latest is None:
            self.show_saved_message("No frame yet, nothing saved", error_color)
            return
        if not sku_calibration_matches(sku, self.calibration_path):
            self.show_saved_message("NOT SAVED: %s was taken with another plane calibration, use a new SKU"
                                    % sku, error_color)
            return
        try:
            file_path = save_image(self.latest, sku, group, kind)
        except Exception as e:
            self.show_saved_message("NOT SAVED: %s" % e, error_color)
            return
        # Saved with every image, so the group can be continued, and checked, with the same settings.
        save_group_settings(sku, group, read_camera_settings(self.camera))
        save_sku_calibration(sku, self.calibration_path)
        self.settings_label.setText("Camera settings saved to %s/%s/%s" % (sku, group, SETTINGS_FILE_NAME))

        color = SAVE_KINDS[kind]["color"]
        self.show_saved_message("Saved %s/%s (%.1f MB) at %s" % (SAVE_KINDS[kind]["folder"], file_path.name,
                                                                 file_path.stat().st_size / 1e6, time.strftime("%H:%M:%S")), color)
        self.thumbnail_label.setPixmap(to_pixmap(self.latest, THUMBNAIL_WIDTH))
        self.thumbnail_label.setStyleSheet("border: %dpx solid %s; border-radius: 6px;" % (THUMBNAIL_BORDER, color))
        self.update_counts()
        print("saved %s" % file_path)
        # Colored border for a moment, so the save is visible on the live view too.
        self.set_view_border(color)
        QTimer.singleShot(FLASH_MS, lambda: self.set_view_border(IDLE_BORDER_COLOR))

    def show_saved_message(self, text, color):
        self.saved_label.setText(text)
        self.saved_label.setStyleSheet("color: %s; font-weight: 600;" % color)

    def match_brightness(self):
        # Exposure and brightness are close to proportional, so scale the exposure by
        # target / measured, a few times (the first step may be off if the image is saturated).
        if self.target_brightness is None:
            self.settings_label.setText("This group has no train/good image to match")
            return
        cam = self.camera["cam"]
        self.camera_label.setText("Matching brightness... keep the scene still")
        QApplication.processEvents()
        for step in range(MATCH_STEPS):
            # Drop the frames taken with the old exposure.
            cam.MV_CC_ClearImageBuffer()
            grab_gray(self.camera)
            measured = label_brightness(grab_gray(self.camera))
            new_exposure = get_float(cam, "ExposureTime") * self.target_brightness / max(measured, 1.0)
            cam.MV_CC_SetFloatValue("ExposureTime", new_exposure)
            print("match step %d: brightness %.1f (target %.1f) -> ExposureTime %.0f us"
                  % (step + 1, measured, self.target_brightness, get_float(cam, "ExposureTime")))
        self.settings_label.setText("Exposure matched to the train/good images of %s / %s"
                                    % (self.current_sku(), self.current_group()))
        self.update_camera_info()

    def remeasure(self):
        # Blocks the window for a few seconds while auto white balance settles.
        self.camera_label.setText("Measuring... keep the scene still")
        QApplication.processEvents()
        lock_white_balance(self.camera)
        self.update_camera_info()

    def make_template(self):
        # A good photo of the current group (train/good to start with) -> its template.png.
        folder = group_folder(self.current_sku(), self.current_group())
        start = folder / "train" / "good"
        path, _ = QFileDialog.getOpenFileName(self, "Good photo for the template", str(start if start.is_dir() else folder),
                                              "Images (*.bmp *.png)")
        if path and TemplateDialog(self, Path(path), folder).exec():
            self.on_group_changed()


def main():
    parser = argparse.ArgumentParser(description="Collect label images into data/<sku>/<group>/.")
    parser.add_argument("--sku", help="SKU to start on, e.g. sku02 (default: the last one, sku01 if none)")
    parser.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    parser.add_argument("--calibration", type=Path, default=PLANE_CALIBRATION_PATH,
                        help="plane calibration of this setup (default: camera/calibration.json)")
    parser.add_argument("--no-camera", action="store_true", help="no camera: Make template only")
    args = parser.parse_args()

    skus = list_skus()
    start_sku = args.sku or (skus[-1] if skus else "sku01")
    if not SKU_PATTERN.match(start_sku):
        raise SystemExit("SKU names are sku<number>, e.g. sku02")
    if not args.calibration.is_file():
        raise SystemExit("no plane calibration at %s: run calibrate.py first (see README)" % args.calibration)
    for group in GROUPS:
        group_folder(start_sku, group).mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    use_camera = not args.no_camera and CAMERA_IMPORT_ERROR is None
    if not args.no_camera and CAMERA_IMPORT_ERROR is not None:
        print("no camera SDK (%s): Make template only" % CAMERA_IMPORT_ERROR)
    camera = None
    if use_camera:
        MvCamera.MV_CC_Initialize()
    try:
        if use_camera:
            try:
                camera = open_camera(args.camera)
                lock_white_balance(camera)
            except Exception as e:
                print("no camera (%s): Make template only" % e)
        window = CollectWindow(camera, args.calibration, start_sku)
        window.show()
        app.exec()
    finally:
        # Runs after the window is closed too, so the camera settings are always put back.
        if camera is not None:
            close_camera(camera)
        if use_camera:
            MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()
