# Live view to set up focus and lighting by hand.
#
# Shows the camera image continuously with, on top of it:
#   - sharpness of the center box (variance of the Laplacian, higher = sharper) and the best value
#     seen so far: turn the focus ring until the number is at its peak.
#   - exposure time / gain: with auto exposure on, more light = shorter exposure time.
#   - mean brightness and % of clipped pixels (pure white / pure black): watch for LED hot spots.
#
# Keys (click the image window first):
#   a       auto exposure + gain on / off (off = locked, so you see the real effect of the LEDs)
#   + / -   exposure up / down (turns auto off)
#   r       reset the best sharpness       s   save the current frame to ./output/
#   q / ESC quit
#
# IMPORTANT: close MVS first (exclusive access).
#
# Run:  uv run camera/live_focus.py            (first camera)
#       uv run camera/live_focus.py 1          (camera number 1)

import sys
import time
from ctypes import *

import cv2
import numpy as np

from capture import OUT_DIR, get_enum, get_float, grab_bgr, restore, sharpness
from MvCameraControl_class import *

WINDOW = "live focus (a: auto, +/-: exposure, r: reset best, s: save, q: quit)"
WINDOW_WIDTH = 1000  # window width in pixels, the height follows the image aspect ratio
ROI_FRACTION = 0.5   # sharpness is measured on the center box, this fraction of width / height


def set_auto(cam, on):
    ret = cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS if on else MV_EXPOSURE_AUTO_MODE_OFF)
    if ret != 0:
        print("set ExposureAuto fail! ret[0x%x]" % ret)
    ret = cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_CONTINUOUS if on else MV_GAIN_MODE_OFF)
    if ret != 0:
        print("set GainAuto fail! ret[0x%x]" % ret)


def set_exposure(cam, value):
    # While ExposureAuto is on, ExposureTime is not writable (see learn/step04).
    cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
    cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
    time.sleep(0.2)  # same wait as BasicDemo Set_parameter
    ret = cam.MV_CC_SetFloatValue("ExposureTime", value)
    if ret != 0:
        print("set exposure time fail! ret[0x%x]" % ret)


def draw_text(img, lines):
    # Big text with a dark outline so it stays readable on bright and dark parts of the image.
    y = 70
    for text, color in lines:
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 10)
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
        y += 75


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cam = MvCamera()
    orig = None
    try:
        MvCamera.MV_CC_Initialize()

        # --- Open the camera (same as capture.py) ---
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != 0:
            raise Exception("enum devices fail! ret[0x%x]" % ret)
        if device_list.nDeviceNum == 0:
            raise Exception("find no device!")
        if index >= device_list.nDeviceNum:
            raise Exception("camera %d not found, only %d camera(s)" % (index, device_list.nDeviceNum))
        info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise Exception("create handle fail! ret[0x%x]" % ret)
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception("open device fail! ret[0x%x]" % ret)

        if info.nTLayerType == MV_GIGE_DEVICE:
            packet_size = cam.MV_CC_GetOptimalPacketSize()
            if int(packet_size) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret != 0:
                    print("Warning: set packet size fail! ret[0x%x]" % ret)
            else:
                print("Warning: get packet size fail! ret[0x%x]" % packet_size)

        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise Exception("set trigger mode fail! ret[0x%x]" % ret)

        # --- Auto exposure / gain / white balance on, remember the original state to restore it ---
        wb = MVCC_ENUMVALUE()
        has_wb = cam.MV_CC_GetEnumValue("BalanceWhiteAuto", wb) == 0  # mono cameras have no white balance
        orig = {
            "ExposureAuto": get_enum(cam, "ExposureAuto"),
            "GainAuto": get_enum(cam, "GainAuto"),
            "BalanceWhiteAuto": wb.nCurValue if has_wb else None,
            "ExposureTime": get_float(cam, "ExposureTime"),
            "Gain": get_float(cam, "Gain"),
        }
        set_auto(cam, True)
        if has_wb:
            ret = cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)
            if ret != 0:
                print("Warning: set BalanceWhiteAuto fail! ret[0x%x]" % ret)

        ret = cam.MV_CC_SetBayerCvtQuality(1)
        if ret != 0:
            raise Exception("set Bayer convert quality fail! ret[0x%x]" % ret)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception("start grabbing fail! ret[0x%x]" % ret)

        out_frame = MV_FRAME_OUT()
        memset(byref(out_frame), 0, sizeof(out_frame))

        # --- The live loop (same structure as learn/step07: main thread grabs, shows, reads keys) ---
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        window_sized = False
        best = 0.0
        while True:
            try:
                color, frame_num = grab_bgr(cam, out_frame)
            except Exception as e:
                print(e)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            h, w = color.shape[:2]

            # Sharpness on the center box only: the target is usually there, and the edges of a
            # lens are softer anyway.
            rw, rh = int(w * ROI_FRACTION), int(h * ROI_FRACTION)
            x0, y0 = (w - rw) // 2, (h - rh) // 2
            score = sharpness(color[y0:y0 + rh, x0:x0 + rw])
            best = max(best, score)

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            over = 100.0 * np.count_nonzero(gray >= 250) / gray.size
            under = 100.0 * np.count_nonzero(gray <= 5) / gray.size
            auto_on = get_enum(cam, "ExposureAuto") != MV_EXPOSURE_AUTO_MODE_OFF

            # Draw on a copy so a saved frame stays clean.
            view = color.copy()
            cv2.rectangle(view, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 255), 4)
            near_best = score >= 0.95 * best
            draw_text(view, [
                ("sharp %.1f  (best %.1f)" % (score, best), (0, 255, 0) if near_best else (0, 165, 255)),
                ("exp %.0f us  gain %.1f dB  [%s]" % (get_float(cam, "ExposureTime"), get_float(cam, "Gain"),
                                                     "AUTO" if auto_on else "LOCKED"), (255, 255, 255)),
                ("mean %.0f  white %.1f%%  black %.1f%%" % (gray.mean(), over, under),
                 (0, 0, 255) if over > 1.0 else (255, 255, 255)),
            ])

            if not window_sized:
                cv2.resizeWindow(WINDOW, WINDOW_WIDTH, WINDOW_WIDTH * h // w)
                window_sized = True
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                set_auto(cam, not auto_on)
            elif key in (ord("+"), ord("=")):
                set_exposure(cam, get_float(cam, "ExposureTime") * 1.25)
            elif key in (ord("-"), ord("_")):
                set_exposure(cam, get_float(cam, "ExposureTime") / 1.25)
            elif key == ord("r"):
                best = 0.0
            elif key == ord("s"):
                OUT_DIR.mkdir(exist_ok=True)
                file_path = OUT_DIR / ("live_%s_fn%d.png" % (time.strftime("%Y%m%d_%H%M%S"), frame_num))
                ok, png = cv2.imencode(".png", color)
                if ok:
                    png.tofile(str(file_path))
                    print("saved to %s (sharpness %.1f)" % (file_path, score))

        # --- Stop, put the camera back the way we found it, close ---
        cv2.destroyAllWindows()
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise Exception("stop grabbing fail! ret[0x%x]" % ret)
        restore(cam, orig)
        ret = cam.MV_CC_CloseDevice()
        if ret != 0:
            raise Exception("close device fail! ret[0x%x]" % ret)
        cam.MV_CC_DestroyHandle()

    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        cam.MV_CC_StopGrabbing()
        if orig is not None:
            restore(cam, orig)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    finally:
        MvCamera.MV_CC_Finalize()

    print("\nDone")


if __name__ == "__main__":
    main()
