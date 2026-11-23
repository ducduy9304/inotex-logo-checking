# Take ONE picture with the camera: auto exposure / gain / white balance, auto focus, then save it.
#
# - Exposure: same as learn/step05 (ExposureAuto + GainAuto Continuous, drop frames until it settles).
# - Focus: same as LiquidLens/AutoFocus (MV_CC_LiquidLens_Open + MV_CC_LiquidLens_AutoFocus).
#   Only a liquid lens can be focused by software. With a normal (manual) lens the script skips
#   the auto focus and prints a sharpness score instead, so you can turn the focus ring by hand
#   and run it again (higher score = sharper).
# - Every setting is put back the way it was at the end.
#
# IMPORTANT: close MVS first (exclusive access).
#
# Run:  uv run camera/capture.py              (first camera)
#       uv run camera/capture.py 1            (camera number 1)

import sys
import time
from ctypes import *
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "hik_sdk" / "MvImport"))

from MvCameraControl_class import *

OUT_DIR = Path(__file__).resolve().parent / "output"

# Auto exposure: drop frames until ExposureTime and Gain stop moving (or MAX_WARMUP_FRAMES).
MAX_WARMUP_FRAMES = 100
STABLE_FRAMES = 5         # this many frames in a row with no real change = settled
STABLE_TOLERANCE = 0.02   # "no real change" = less than 2 %

# Auto focus: coarse steps over the whole lens range, range [3, 20], 6 is the SDK recommendation.
AF_COARSE_STEPS = 6

GRAB_TIMEOUT_MS = 3000


def get_float(cam, name):
    v = MVCC_FLOATVALUE()
    cam.MV_CC_GetFloatValue(name, v)
    return v.fCurValue


def get_enum(cam, name):
    v = MVCC_ENUMVALUE()
    cam.MV_CC_GetEnumValue(name, v)
    return v.nCurValue


def restore(cam, orig):
    # Auto must be Off before ExposureTime / Gain are writable again (see learn/step04).
    cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
    cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
    if orig["BalanceWhiteAuto"] is not None:
        cam.MV_CC_SetEnumValue("BalanceWhiteAuto", orig["BalanceWhiteAuto"])
    cam.MV_CC_SetFloatValue("ExposureTime", orig["ExposureTime"])
    cam.MV_CC_SetFloatValue("Gain", orig["Gain"])
    cam.MV_CC_SetEnumValue("ExposureAuto", orig["ExposureAuto"])
    cam.MV_CC_SetEnumValue("GainAuto", orig["GainAuto"])


def grab_bgr(cam, out_frame):
    # Grab one frame and convert it to BGR8 with the SDK (same as learn/step05).
    ret = cam.MV_CC_GetImageBuffer(out_frame, GRAB_TIMEOUT_MS)
    if not out_frame.pBufAddr or ret != 0:
        raise Exception("get one frame fail! ret[0x%x]" % ret)
    fi = out_frame.stFrameInfo

    bgr_size = fi.nWidth * fi.nHeight * 3
    convert = MV_CC_PIXEL_CONVERT_PARAM_EX()
    memset(byref(convert), 0, sizeof(convert))
    convert.nWidth = fi.nWidth
    convert.nHeight = fi.nHeight
    convert.pSrcData = out_frame.pBufAddr
    convert.nSrcDataLen = fi.nFrameLen
    convert.enSrcPixelType = fi.enPixelType
    convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
    convert.pDstBuffer = (c_ubyte * bgr_size)()
    convert.nDstBufferSize = bgr_size

    ret = cam.MV_CC_ConvertPixelTypeEx(convert)
    # Give the buffer back to the SDK (always paired with GetImageBuffer).
    cam.MV_CC_FreeImageBuffer(out_frame)
    if ret != 0:
        raise Exception("convert pixel fail! ret[0x%x]" % ret)

    bgr_buf = (c_ubyte * convert.nDstLen)()
    memmove(byref(bgr_buf), convert.pDstBuffer, convert.nDstLen)
    color = np.frombuffer(bgr_buf, dtype=np.uint8).reshape(fi.nHeight, fi.nWidth, 3)
    return color, fi.nFrameNum


def wait_exposure_settled(cam, out_frame):
    # Auto exposure / gain needs to SEE frames before it settles: grab and drop them,
    # stop once ExposureTime and Gain have stayed (almost) the same for STABLE_FRAMES frames.
    last_exp, last_gain = get_float(cam, "ExposureTime"), get_float(cam, "Gain")
    stable = 0
    for i in range(MAX_WARMUP_FRAMES):
        ret = cam.MV_CC_GetImageBuffer(out_frame, GRAB_TIMEOUT_MS)
        if ret != 0:
            raise Exception("warm-up get frame fail! ret[0x%x]" % ret)
        cam.MV_CC_FreeImageBuffer(out_frame)

        exp, gain = get_float(cam, "ExposureTime"), get_float(cam, "Gain")
        print("warm-up %3d: ExposureTime = %8.0f us, Gain = %5.2f dB" % (i + 1, exp, gain))
        exp_same = abs(exp - last_exp) <= STABLE_TOLERANCE * max(last_exp, 1.0)
        gain_same = abs(gain - last_gain) <= max(STABLE_TOLERANCE * last_gain, 0.05)
        stable = stable + 1 if exp_same and gain_same else 0
        last_exp, last_gain = exp, gain
        if stable >= STABLE_FRAMES:
            print("auto exposure settled.")
            return
    print("Warning: auto exposure did not settle after %d frames, taking the picture anyway." % MAX_WARMUP_FRAMES)


def auto_focus(cam):
    # Same calls as LiquidLens/AutoFocus/CamOperation_class.py Openlens + Set_AutoFocus.
    # Returns True if the lens was opened (so it must be closed later).
    stLensInfo = MV_CC_LIQUIDLENS_INFO()
    ret = cam.MV_CC_LiquidLens_Open(stLensInfo)
    if ret != MV_OK:
        print("No liquid lens found (ret[0x%x]): auto focus skipped, focus the lens ring by hand." % ret)
        return False
    print("Liquid lens opened, focal power range: %d .. %d"
          % (stLensInfo.nMinFocalPower, stLensInfo.nMaxFocalPower))

    stAutoFocusParam = MV_CC_LIQUIDLENS_AUTOFOCUS_PARAM()
    memset(byref(stAutoFocusParam), 0, sizeof(stAutoFocusParam))
    stAutoFocusParam.nFocalPowerStart = stLensInfo.nMinFocalPower
    stAutoFocusParam.nFocalPowerEnd = stLensInfo.nMaxFocalPower
    stAutoFocusParam.nCoarseSteps = AF_COARSE_STEPS
    stAutoFocusParam.nRoiEnable = 0  # judge the sharpness on the whole image

    # 0 = drop the frames taken while focusing, so the next GetImageBuffer is an in-focus frame.
    # The SDK switches the camera to software trigger while focusing and puts it back afterwards.
    nResultFocalPower = c_int(0)
    ret = cam.MV_CC_LiquidLens_AutoFocus(stAutoFocusParam, 0, nResultFocalPower)
    if ret != MV_OK:
        print("Warning: auto focus fail! ret[0x%x], taking the picture with the current focus." % ret)
    else:
        print("auto focus done, best focal power = %d" % nResultFocalPower.value)
    return True


def sharpness(color):
    # Variance of the Laplacian: a common focus measure, higher = sharper (only compare the same scene).
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cam = MvCamera()
    orig = None
    lens_open = False
    try:
        MvCamera.MV_CC_Initialize()

        # --- Open the camera (same as learn/step05) ---
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
        ret = cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
        if ret != 0:
            raise Exception("set ExposureAuto fail! ret[0x%x]" % ret)
        ret = cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_CONTINUOUS)
        if ret != 0:
            raise Exception("set GainAuto fail! ret[0x%x]" % ret)
        if has_wb:
            ret = cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)
            if ret != 0:
                print("Warning: set BalanceWhiteAuto fail! ret[0x%x]" % ret)

        ret = cam.MV_CC_SetBayerCvtQuality(1)
        if ret != 0:
            raise Exception("set Bayer convert quality fail! ret[0x%x]" % ret)

        # --- Start grabbing ---
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception("start grabbing fail! ret[0x%x]" % ret)

        out_frame = MV_FRAME_OUT()
        memset(byref(out_frame), 0, sizeof(out_frame))

        # 1) Exposure first: auto focus judges sharpness on the image, it needs a good brightness.
        wait_exposure_settled(cam, out_frame)

        # 2) Focus.
        lens_open = auto_focus(cam)
        if lens_open:
            # Brightness may move a little while the lens changes: let it settle again,
            # and throw away the frames that were already in the SDK buffer.
            cam.MV_CC_ClearImageBuffer()
            wait_exposure_settled(cam, out_frame)

        # 3) The picture.
        color, frame_num = grab_bgr(cam, out_frame)
        print("got frame %d: %dx%d, ExposureTime = %.0f us, Gain = %.2f dB"
              % (frame_num, color.shape[1], color.shape[0],
                 get_float(cam, "ExposureTime"), get_float(cam, "Gain")))
        print("sharpness = %.1f  (higher = sharper, compare only shots of the same scene)" % sharpness(color))
        print("mean brightness = %.1f  (0 = black, 255 = white)" % color.mean())

        OUT_DIR.mkdir(exist_ok=True)
        file_path = OUT_DIR / ("capture_%s.png" % time.strftime("%Y%m%d_%H%M%S"))
        # imencode + tofile instead of imwrite: works even if the path has non-ASCII characters.
        ok, png = cv2.imencode(".png", color)
        if not ok:
            raise Exception("encode png fail!")
        png.tofile(str(file_path))
        print("saved to %s" % file_path)

        # --- Stop, put the camera back the way we found it, close ---
        if lens_open:
            cam.MV_CC_LiquidLens_Close()
            lens_open = False
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
        if lens_open:
            cam.MV_CC_LiquidLens_Close()
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
