#!/usr/bin/env python3
# STVID/SVBONY V10 experimental acquisition bridge.
# Each STVID buffer is captured by a separate short-lived worker process.
# PySVB deliberately owns the complete camera lifecycle: open, start, capture, stop and close.
# The native SVBONY SDK library is loaded only for SDK compatibility; the
# native capture/lifecycle API is NOT used as a second camera-access path.
# This avoids the status=2 / INVALID_ID behaviour seen when the native API
# was mixed with the proven PySVB acquisition path.
import sys
import os
import numpy as np
import cv2
import time
import ctypes

# Load libusb globally before loading the SVBONY SDK. With multiprocessing
# using 'spawn', worker processes import this module again, so the order
# matters: libSVBCameraSDK.so depends on libusb symbols being visible.
# Earlier development versions sometimes loaded the wrong camera library;
# the required SVBONY library is libSVBCameraSDK.so.
# The native library is therefore loaded lazily, after libusb is available.
# multiprocessing uses spawn, so the child re-imports this module.  The old
# V10 tried to load libSVBCameraSDK.so at module import time, BEFORE libusb was
# made RTLD_GLOBAL in the worker.  In the spawned worker that load could fail,
# leaving _svb_native=None even though the same SDK works normally.
# Load the SVBONY native library only after libusb has been loaded globally.
_svb_native = None
_svb_native_path = None

# Make libusb globally visible before loading the SVBONY SDK.
# This is required because the SDK shared library depends on libusb symbols.
# It is especially important in multiprocessing workers created with 'spawn'.
try:
    ctypes.CDLL("libusb-1.0.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    try:
        ctypes.CDLL("/home/hacker3/svbony/SVBCameraSDK/lib/x64/libusb-1.0.so",
                    mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
_svb_candidates = (
    "/home/hacker3/svbony/SVBCameraSDK/lib/x64/libSVBCameraSDK.so",
    "/usr/local/lib/libSVBCameraSDK.so",
    "libSVBCameraSDK.so",
)

def _load_svb_native():
    global _svb_native, _svb_native_path
    if _svb_native is not None:
        return _svb_native

    last_error = None
    for _name in _svb_candidates:
        try:
            _svb_native = ctypes.CDLL(_name)
            _svb_native_path = _name
            _svb_native.SVBGetVideoData.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_long,
                ctypes.c_int,
            ]
            _svb_native.SVBGetVideoData.restype = ctypes.c_int

            # The native library exposes these lifecycle functions, but V10 does not use
            for _fname in ("SVBOpenCamera", "SVBStartVideoCapture",
                           "SVBStopVideoCapture", "SVBCloseCamera"):
                _fn = getattr(_svb_native, _fname)
                _fn.argtypes = [ctypes.c_int]
                _fn.restype = ctypes.c_int

            _svb_native.SVBSetControlValue.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_int
            ]
            _svb_native.SVBSetControlValue.restype = ctypes.c_int

            print("SVBONY V10 native library:", _svb_native_path, flush=True)
            return _svb_native
        except OSError as exc:
            last_error = exc

    raise OSError(
        "Could not load libSVBCameraSDK.so for V10 native capture: %s"
        % last_error
    )

# PySVB's get_video_data() returns a temporary RGBA8 object for each frame.
# Those large temporary allocations can remain in the worker's RSS even after
# Python has released the objects. malloc_trim() can help, but is not the main
# solution. The reliable memory boundary is the short-lived worker process.
# The STVID shared buffers themselves are not affected by malloc_trim().
try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int
except Exception:
    _libc = None

def _v22_malloc_trim():
    if _libc is not None:
        try:
            return _libc.malloc_trim(0)
        except Exception:
            pass
    return 0
import multiprocessing
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
from stvid.utils import observe_logic
import logging
import configparser
import argparse
import traceback
import gc

try:
    import resource
except ImportError:
    resource = None

def setup_logging(path):
    logFormatter = logging.Formatter(
        "%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] %(message)s"
    )
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(os.path.join(path, "acquire.log"))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger

# Pi camera capture path retained from the original STVID acquisition code.
def capture_pi(image_queue, z1base, t1base, z2base, t2base, nx, ny, nz, tend, device_id, live, conf_file):
    global logger
    logger = setup_logging(os.getcwd())

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(conf_file)
    
    from picamerax.array import PiRGBArray
    from picamerax import PiCamera

    z1 = np.ctypeslib.as_array(z1base.get_obj()).reshape(ny, nx, nz)
    t1 = np.ctypeslib.as_array(t1base.get_obj())
    z2 = np.ctypeslib.as_array(z2base.get_obj()).reshape(ny, nx, nz)
    t2 = np.ctypeslib.as_array(t2base.get_obj())
    
    # Initialize camera configuration.
    first = True
    slow_CPU = False

    # Configure the Pi camera.
    camera = PiCamera(sensor_mode=2)
    camera.resolution = (nx, ny)    
    # Disable automatic exposure and white-balance control.
    camera.exposure_mode = "off"        
    camera.awb_mode = "off"
    # ISO must be zero for the configured analogue/digital gains to take effect.
    camera.iso = 0
    # Apply camera settings from the selected configuration file.
    camera.framerate = cfg.getfloat(camera_type, "framerate")
    camera.awb_gains = (cfg.getfloat(camera_type, "awb_gain_red"), cfg.getfloat(camera_type, "awb_gain_blue"))    
    camera.analog_gain = cfg.getfloat(camera_type, "analog_gain")
    camera.digital_gain = cfg.getfloat(camera_type, "digital_gain")
    camera.shutter_speed = cfg.getint(camera_type, "exposure")

    rawCapture = PiRGBArray(camera, size=(nx, ny))
    # Give the camera a short warm-up period.
    time.sleep(0.1)

    try:
        # Capture until the configured end time.
        while float(time.time()) < tend:
            # Read a sequence of frames into one of the two STVID buffers.
            i = 0
            for frameA in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                            
                # Record the frame start time.
                t0 = float(time.time())                
                # Get the raw NumPy image returned by the camera.
                frame = frameA.array
                                    
                # Use the midpoint of the capture interval as the frame timestamp.
                t = (float(time.time()) + t0) / 2
                
                # Ignore frames that could not be captured.
                if frame is not None:
                    # STVID expects a monochrome image for this path.
                    z = np.asarray(cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY)).astype(np.uint8)
                    # Optional 180-degree rotation; normally disabled.
                    # z = np.rot90(z, 2)
                
                    # Optional live display.
                    if live is True:                            
                        cv2.imshow("Capture", z)    
                        cv2.waitKey(1)
                    
                    # Store the frame and timestamp in the active shared buffer.
                    if first:
                        z1[:, :, i] = z
                        t1[i] = t
                    else:
                        z2[:, :, i] = z
                        t2[i] = t
                        
                # Reset the capture stream before requesting the next frame.
                rawCapture.truncate(0)
                # Stop after exactly nz frames, producing one complete STVID buffer.
                i += 1
                if i >= nz:
                    break
                
            if first: 
                buf = 1
            else:
                buf = 2
            image_queue.put(buf)
            logger.debug("Captured buffer %d" % buf)

            # Switch to the other shared buffer for the next batch.
            first = not first
        reason = "Session complete"
    except KeyboardInterrupt:
        print()
        reason = "Keyboard interrupt"
    except ValueError as e:
        logger.error("%s" % e)
        reason = "Wrong image dimensions? Fix nx, ny in config."
    finally:
        # Capture process finished.
        logger.info("Capture: %s - Exiting" % reason)
        camera.close()



# OpenCV camera capture path retained for compatibility with STVID.
def capture_cv2(image_queue, z1base, t1base, z2base, t2base, nx, ny, nz, tend, device_id, live, conf_file):
    global logger
    logger = setup_logging(os.getcwd())

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(conf_file)

    z1 = np.ctypeslib.as_array(z1base.get_obj()).reshape(ny, nx, nz)
    t1 = np.ctypeslib.as_array(t1base.get_obj())
    z2 = np.ctypeslib.as_array(z2base.get_obj()).reshape(ny, nx, nz)
    t2 = np.ctypeslib.as_array(t2base.get_obj())
    
    # Initialize camera configuration.
    camera_type  = "CV2"
    first = True
    slow_CPU = False

    # Open the configured OpenCV device.
    if cfg.has_option(camera_type, "device_string"):
        device = cv2.VideoCapture(cfg.get(camera_type, "device_string"))
    else:
        device = cv2.VideoCapture(device_id)

    # Optional software binning.
    try:
        software_bin = cfg.getint(camera_type, "software_bin")
    except configparser.Error:
        software_bin = 1
    
    # Apply the requested capture dimensions.
    device.set(3, nx * software_bin)
    device.set(4, ny * software_bin)
   
    try:
        # Capture until the configured end time.
        while float(time.time()) < tend:
            # The same two-buffer concept is used here: capture reports a buffer
            # only after all nz frames have been written, allowing the compressor
            # to process the other shared buffer safely.

            # Read nz frames for the current STVID buffer.
            for i in range(nz):
                # Record the frame start time.
                t0 = float(time.time())

                # Read one frame from OpenCV.
                res, frame = device.read()

                # Use the midpoint of the capture interval as the frame timestamp.
                t = (float(time.time()) + t0) / 2

                # Ignore frames that could not be captured.
                if res is True:
                    # Convert the OpenCV image to monochrome.
                    z = np.asarray(cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY)).astype(np.uint8)

                    # Resize only when the camera returns dimensions different from STVID.
                # The SVBONY path normally configures the requested ROI directly.
                if z is not None and z.shape != (ny, nx):
                    z = cv2.resize(z, (nx, ny), interpolation=cv2.INTER_AREA)

                # Apply optional software binning.
                    if software_bin > 1:
                        my, mx = z.shape
                        z = cv2.resize(z, (mx // software_bin, my // software_bin))
                    
                    # Optional live display.
                    if live is True:
                        cv2.imshow("Capture", z)
                        cv2.waitKey(1)

                    # Store the frame and timestamp in the active shared buffer.
                    if first:
                        z1[:, :, i] = z
                        t1[i] = t
                    else:
                        z2[:, :, i] = z
                        t2[i] = t

            if first: 
                buf = 1
            else:
                buf = 2
            image_queue.put(buf)
            logger.debug("Captured z%d" % buf)

            # Switch to the other shared buffer for the next batch.
            first = not first
        reason = "Session complete"
    except KeyboardInterrupt:
        print()
        reason = "Keyboard interrupt"
    except ValueError as e:
        logger.error("%s" % e)
        reason = "Wrong image dimensions? Fix nx, ny in config."
    finally:
        # Capture process finished.
        logger.info("Capture: %s - Exiting" % reason)
        device.release()


# SVBONY bridge: PySVB is the single owner of the camera lifecycle.
def _v23_mem_mb():
    """Return current and peak RSS in MiB on Linux."""
    current = -1.0
    peak = -1.0
    try:
        with open("/proc/self/status", "r") as fp:
            for line in fp:
                if line.startswith("VmRSS:"):
                    current = float(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    if resource is not None:
        try:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass
    return current, peak

class SVBToASIBridge:
    """SVBONY bridge with native SDK camera lifecycle.

    V10 uses PySVB as the SOLE owner of the camera lifecycle. Native ctypes
    is used only for SVBGetVideoData() into one persistent caller-owned buffer.
    """

    def __init__(self, requested_width, requested_height, exposure_us, gain, camera_serial):
        from pysvb.camera import (
            PySVBCameraSDK,
            SVB_CAMERA_MODE,
            SVB_ROI_FORMAT,
            SVB_CONTROL_TYPE,
        )

        self.sdk = PySVBCameraSDK()
        print("SVBONY V10.2: PySVB is sole owner of camera lifecycle and frame delivery", flush=True)
        self.SVB_CAMERA_MODE = SVB_CAMERA_MODE
        self.SVB_ROI_FORMAT = SVB_ROI_FORMAT
        self.SVB_CONTROL_TYPE = SVB_CONTROL_TYPE

        connected = self.sdk.get_num_of_connected_cameras()
        print("SVBONY SDK version:", self.sdk.sdk_version)
        print("SVBONY connected cameras:", connected)
        if connected <= 0:
            raise RuntimeError("Geen SVBONY-camera gevonden")

        # Select the physical SVBONY camera by serial number, never by SDK index.
        # Two SVBONY cameras are connected: index 0 is the SV905C2 (1280x960),
        # while the required SV305M PRO is the configured serial.
        requested_serial = str(camera_serial).strip()
        if not requested_serial:
            raise ValueError("V10.2: SVBONY serial selector must not be empty")
        info = None
        auto_select = requested_serial.upper().startswith("AUTO:")
        for index in range(connected):
            candidate = self.sdk.get_camera_info(index)
            candidate_serial = str(getattr(candidate, "CameraSN", "")).strip()
            candidate_name = str(getattr(candidate, "FriendlyName", "SVBONY")).strip()
            print("SVBONY detected camera %d: %s serial=%s" % (
                index, candidate_name, candidate_serial
            ), flush=True)
            if (not auto_select and candidate_serial == requested_serial) or \
               (auto_select and "SV305M PRO" in candidate_name.upper()):
                info = candidate
                break
        if info is None:
            selector = "SV305M PRO" if auto_select else requested_serial
            raise RuntimeError(
                "V10.2: SVBONY camera %s was not found among %d connected camera(s)"
                % (selector, connected)
            )
        self.device_id = info.CameraID
        print("SVBONY selected camera:", info.FriendlyName)
        print("SVBONY camera ID:", self.device_id)
        print("SVBONY serial:", info.CameraSN)

        # IMPORTANT CAMERA OWNERSHIP RULE:
        # Do not call native SVBOpenCamera/SVBStartVideoCapture here.
        # Earlier mixed PySVB/native experiments produced status=2 (INVALID_ID).
        self.sdk.open_camera(self.device_id)
        print("SVBONY: PySVB camera opened for setup", flush=True)
        self.props = self.sdk.get_camera_property(self.device_id)

        print("SVBONY maximum:", self.props.MaxWidth, "x", self.props.MaxHeight)
        print("SVBONY SDK IsColorCam flag:", self.props.IsColorCam)
        print("SVBONY NOTE: SV305M PRO is treated as MONOCHROME for STVID; RGBA R=G=B is SDK packaging, not a color sensor.")
        print("SVBONY max bit depth:", self.props.MaxBitDepth)

        mode = self.sdk.get_camera_mode(self.device_id)
        if mode != self.SVB_CAMERA_MODE.SVB_MODE_NORMAL:
            self.sdk.set_camera_mode(
                self.device_id, self.SVB_CAMERA_MODE.SVB_MODE_NORMAL
            )

        self.width = int(self.props.MaxWidth)
        self.height = int(self.props.MaxHeight)

        # Use the frame dimensions requested by STVID/svbconfiguration.ini.
        # Do not automatically force the SV305M PRO to its maximum 1920x1080 ROI.
        # A smaller configured ROI reduces USB traffic, SDK memory traffic and
        # conversion work before the data reaches the STVID shared buffer.
        # The actual ROI is read back and verified below.
        # If the SDK does not accept the requested dimensions, fail explicitly
        # instead of silently processing a different image size.
        requested_width = int(requested_width)
        requested_height = int(requested_height)

        if requested_width <= 0 or requested_height <= 0:
            raise ValueError(
                "Invalid configured frame size: %dx%d"
                % (requested_width, requested_height)
            )

        if requested_width > self.width or requested_height > self.height:
            raise ValueError(
                "V10.2: configured frame size %dx%d exceeds selected SVBONY camera %dx%d"
                % (requested_width, requested_height,
                   self.width, self.height)
            )

        roi = self.SVB_ROI_FORMAT(
            0, 0, requested_width, requested_height, 1
        )
        self.sdk.set_roi_format(self.device_id, roi)
        self.roi = self.sdk.get_roi_format(self.device_id)

        print("SVBONY configured ROI:", requested_width, "x", requested_height)
        print("SVBONY actual ROI:", self.roi.width, "x", self.roi.height)

        if (int(self.roi.width), int(self.roi.height)) != (
            requested_width, requested_height
        ):
            raise RuntimeError(
                "SVBONY SDK did not accept configured ROI %dx%d; "
                "actual ROI is %dx%d"
                % (requested_width, requested_height,
                   self.roi.width, self.roi.height)
            )

        # Exposure and gain come from the same svbconfiguration.ini used by STVID.
        self.exposure_us = int(exposure_us)
        self.gain = int(gain)

        if self.exposure_us <= 0:
            raise ValueError("Invalid SVBONY exposure: %d us" % self.exposure_us)
        if self.gain < 0:
            raise ValueError("Invalid SVBONY gain: %d" % self.gain)

        # Apply camera controls through the already-open PySVB handle.
        self.sdk.set_control_value(
            self.device_id, self.SVB_CONTROL_TYPE.SVB_EXPOSURE,
            self.exposure_us, False
        )
        self.sdk.set_control_value(
            self.device_id, self.SVB_CONTROL_TYPE.SVB_GAIN,
            self.gain, False
        )
        self.sdk.set_autosave_param(self.device_id, False)

        print("SVBONY: PySVB setup exposure =", self.exposure_us,
              "us, gain =", self.gain)

        # Keep the PySVB camera handle open for the entire worker session.
        # This is intentional: PySVB is the only owner of open/start/capture/stop/close.
        # A second native open/close sequence was unreliable during development and
        # could produce status=2 (INVALID_ID).
        # lifecycle on the same device and produces status=2.

        # Do not open the camera through the native ctypes API.
        # Native ctypes is not a second camera driver in V10.
        # It is retained only because the SDK library must be available in this
        # experimental bridge and because the native API was useful during diagnosis.
        # Actual frame delivery is through PySVB get_video_data().
        # The frame data are copied into one persistent caller-owned RGBA8 buffer.
        # Keeping one buffer also avoids allocating another ctypes buffer per frame.
        #
        # The PySVB handle remains open until close().
        self.native_open = False

        # get_video_data() uses the same camera handle that PySVB opened and started.
        # No second native camera handle is created.
        # Diagnostic testing established that the returned data are RGBA8:
        # four bytes per pixel, even though the camera reports a 16-bit maximum.
        # return RGBA8 pixels (4 bytes/pixel), despite MaxBitDepth reporting 16.
        self.bytes_per_pixel = 4
        self.real_frame_size = int(
            self.bytes_per_pixel * self.roi.width * self.roi.height
        )
        # The SDK buffer is exactly one RGBA8 frame; this matches the working test path.
        self.sdk_buffer_size = self.real_frame_size  # Exactly one RGBA8 frame per SDK buffer
        self.wait_ms = 5000

        # Persistent caller-owned RGBA8 buffer.
        self.raw_buffer = (ctypes.c_ubyte * self.real_frame_size)()
        self.raw_ptr = ctypes.cast(
            self.raw_buffer, ctypes.POINTER(ctypes.c_ubyte)
        )

        # Print detailed diagnostics for only the first four frames.
        # These diagnostics are useful for detecting connected-but-black frames and
        self.debug_frame_count = 0
        # Reusable 2-D monochrome output frame. Reusing it limits per-frame allocation.
        # into this array instead of allocating a new image every frame.
        self.output_frame = np.empty(
            (int(self.roi.height), int(self.roi.width)), dtype=np.uint8
        )
        self.frame_counter = 0
        self.native_started = False

        print("SVBONY data format: RGBA8, bytes/pixel:", self.bytes_per_pixel)
        print("SVBONY real frame size:", self.real_frame_size)
        print("SVBONY SDK buffer size:", self.sdk_buffer_size)

    def native_set_control(self, ctrl_type, value, auto=False):
        # STVID compatibility helper: route control changes through PySVB.
        # but route the actual control operation through PySVB.
        return self.sdk.set_control_value(
            self.device_id, ctrl_type, int(value), bool(auto)
        )

    def init(self, sdk=None):
        # STVID calls asi.init() because the original backend API is ASI-shaped.
        # The PySVB SDK is already initialized by this bridge constructor.
        return True

    def get_num_cameras(self):
        return 1

    def list_cameras(self):
        return [getattr(self.props, "FriendlyName", "SVBONY")]

    def Camera(self, device_id):
        # Represent the single SVBONY camera through the ASI-shaped STVID interface.
        return self

    def get_camera_property(self):
        return {
            "FriendlyName": getattr(self.props, "FriendlyName", "SVBONY"),
            "MaxWidth": int(self.props.MaxWidth),
            "MaxHeight": int(self.props.MaxHeight),
            "MaxBitDepth": int(self.props.MaxBitDepth),
            "IsColorCam": bool(self.props.IsColorCam),
        }

    def disable_dark_subtract(self):
        # Dark subtraction is an ASI-specific feature and is not used here.
        return True

    def set_control_value(self, ctrl_type, value, auto=False):
        # Translate the small set of ASI-style controls that STVID uses into SVBONY controls.
        if ctrl_type == 0:
            ctrl_type = self.SVB_CONTROL_TYPE.SVB_GAIN
        elif ctrl_type == 1:
            ctrl_type = self.SVB_CONTROL_TYPE.SVB_EXPOSURE
        else:
            # Other ASI-only controls, such as white balance or gamma, have no direct
            # equivalent in this experimental SVBONY bridge.
            return True
        if ctrl_type == self.SVB_CONTROL_TYPE.SVB_GAIN:
            self.gain = int(value)
        elif ctrl_type == self.SVB_CONTROL_TYPE.SVB_EXPOSURE:
            self.exposure_us = int(value)
        return self.native_set_control(ctrl_type, value, auto)

    def get_control_values(self):
        # STVID expects Gain and Temperature to be available.
        # Gain is the configured bridge value; temperature is not provided here.
        return {"Gain": self.gain, "Temperature": 0}

    def set_roi(self, bins=1):
        # ROI is configured during bridge initialization.
        # Keep this method for compatibility with the existing STVID camera interface.
        return True

    def set_image_type(self, image_type):
        # Image type is fixed by the SVBONY/PySVB data path.
        return True

    def start_video_capture(self):
        # PySVB owns the complete camera lifecycle.
        print("SVBONY: restoring test exposure/gain immediately before PySVB start:",
              self.exposure_us, "us / gain", self.gain, flush=True)
        self.sdk.set_control_value(
            self.device_id, self.SVB_CONTROL_TYPE.SVB_EXPOSURE,
            self.exposure_us, False
        )
        self.sdk.set_control_value(
            self.device_id, self.SVB_CONTROL_TYPE.SVB_GAIN,
            self.gain, False
        )
        result = self.sdk.start_video_capture(self.device_id)
        print("SVBONY V10 PySVB start_video_capture result:", result, flush=True)
        self.native_started = True
        return result

    def stop_video_capture(self):
        try:
            result = self.sdk.stop_video_capture(self.device_id)
            print("SVBONY V10 PySVB stop_video_capture result:", result, flush=True)
            self.native_started = False
            return result
        except Exception:
            return None

    def capture_video_frame(self):
        """Capture one frame through the SAME PySVB camera handle.

        V17/V10 established that the PySVB get_video_data() path is the
        working acquisition path on this SV305M PRO. V10/V10 proved that
        calling SVBGetVideoData() directly through a second ctypes entry
        point returns status=2 even while PySVB is streaming.

        The short-lived worker remains intentional: PySVB may retain a large
        temporary RGBA allocation per frame, but the worker exits after one
        100-frame STVID buffer, so the OS can reclaim the complete process
        address space.
        """
        z = self.sdk.get_video_data(
            self.device_id, self.sdk_buffer_size, self.wait_ms
        )
        if z is None:
            raise RuntimeError("V10 PySVB get_video_data returned no frame")

        self.frame_counter += 1
        frame_no = self.debug_frame_count + 1

        try:
            raw = np.frombuffer(z, dtype=np.uint8)
        except TypeError:
            raw = np.asarray(z, dtype=np.uint8).reshape(-1)

        expected = int(self.roi.width) * int(self.roi.height) * 4
        if raw.size < expected:
            raise RuntimeError(
                "V10 PySVB frame too small: got %d bytes, expected %d"
                % (raw.size, expected)
            )

        rgba = raw[:expected].reshape(
            (int(self.roi.height), int(self.roi.width), 4)
        )

        if self.debug_frame_count < 4:
            print(
                "SVBONY V10 RAW frame", frame_no,
                ": returned_len =", raw.size,
                "expected_1frame_buffer =", expected,
                "one_RGBA_frame =", expected, flush=True
            )
            print(
                "SVBONY V10 frame", frame_no,
                "first64 HEX =", bytes(raw[:64]).hex(" "), flush=True
            )
            r0, g0, b0, a0 = (rgba[:, :, 0], rgba[:, :, 1],
                              rgba[:, :, 2], rgba[:, :, 3])
            print(
                "SVBONY V10 block1: R[%d,%d,%.3f] G[%d,%d,%.3f] "
                "B[%d,%d,%.3f] A[%d,%d]" % (
                    int(r0.min()), int(r0.max()), float(r0.mean()),
                    int(g0.min()), int(g0.max()), float(g0.mean()),
                    int(b0.min()), int(b0.max()), float(b0.mean()),
                    int(a0.min()), int(a0.max())
                ), flush=True
            )
            print(
                "SVBONY V10 block1 channel differences: "
                "R-G max =", int(np.max(np.abs(
                    r0.astype(np.int16) - g0.astype(np.int16)
                ))),
                "R-B max =", int(np.max(np.abs(
                    r0.astype(np.int16) - b0.astype(np.int16)
                ))),
                "A unique =", np.unique(a0)[:10].tolist(), flush=True
            )
            self.debug_frame_count += 1

        # Copy the monochrome R channel into the reusable STVID frame.
        # For this SV305M PRO, R=G=B was observed in the SDK's RGBA packaging.
        self.output_frame[:, :] = rgba[:, :, 0]

        del rgba
        del raw
        del z

        if self.frame_counter % 10 == 0:
            collected = gc.collect()
            trimmed = _v22_malloc_trim()
            rss_now, rss_peak = _v23_mem_mb()
            logger.debug(
                "V10 MEMORY CHECK: frame=%d rss_now=%.1f MiB "
                "rss_peak=%.1f MiB gc_collected=%d malloc_trim=%d",
                self.frame_counter, rss_now, rss_peak, collected, trimmed
            )

        return self.output_frame

    def close(self):
        # Close the camera through PySVB only; never call the native close function.
        # The native lifecycle API is intentionally not part of V10 camera ownership.
        try:
            if getattr(self, "native_started", False):
                self.stop_video_capture()
        finally:
            try:
                self.sdk.close_camera(self.device_id)
                print("SVBONY V10 PySVB camera closed", flush=True)
            except Exception as exc:
                logger.error("V10 PySVB close_camera exception: %s", exc)

    def close_camera(self):
        return self.close()


def _v24_capture_one_chunk(buf, zbase, tbase, nx, ny, nz, frame_start, frame_count, conf_file, sequence, camera_serial):
    """Capture a small chunk of one STVID buffer in a short-lived worker.

    V10 showed that PySVB can grow worker RSS by roughly one frame (~8 MiB)
    repeatedly during a long SDK session.  A 100-frame worker was therefore
    still able to reach the system OOM limit before it exited.

    V24 deliberately restarts the SDK worker every small chunk (default 10
    frames) while continuing to fill the SAME shared STVID buffer.  The OS
    then reclaims every SDK allocation at worker exit, without changing the
    STVID two-buffer protocol or FITS format.
    """
    global logger
    logger = setup_logging(os.getcwd())
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not cfg.read(conf_file):
        raise RuntimeError("V24: could not read config file: %s" % conf_file)

    try:
        ctypes.CDLL("libusb-1.0.so", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        ctypes.CDLL("/home/hacker3/svbony/SVBCameraSDK/lib/x64/libusb-1.0.so", mode=ctypes.RTLD_GLOBAL)

    z = np.ctypeslib.as_array(zbase.get_obj()).reshape(ny, nx, nz)
    t = np.ctypeslib.as_array(tbase.get_obj())

    camera_type = cfg.get("Setup", "camera_type")
    configured_exposure_us = cfg.getint(camera_type, "exposure")
    configured_gain = cfg.getint(camera_type, "gain")
    logger.info("V24 CONFIG: %s exposure=%d us gain=%d", camera_type, configured_exposure_us, configured_gain)

    camera = None
    batch_start = time.time()
    try:
        bridge = SVBToASIBridge(
            nx, ny,
            exposure_us=configured_exposure_us,
            gain=configured_gain,
            camera_serial=camera_serial,
        )
        camera = bridge.Camera(0)
        camera.start_video_capture()

        logger.info(
            "V24 WORKER START: buffer=%d sequence=%d frames=%d..%d count=%d",
            buf, sequence, frame_start + 1, frame_start + frame_count, frame_count,
        )

        for local_i in range(frame_count):
            i = frame_start + local_i
            t0 = time.time()
            frame_ok = False
            for attempt in range(6):
                frame = camera.capture_video_frame()
                if frame is None:
                    continue
                if int(frame.max()) == 0:
                    continue
                frame_ok = True
                if attempt:
                    logger.info(
                        "V24 worker buffer %d: recovered frame %d after %d retries",
                        buf, i + 1, attempt,
                    )
                break
            if not frame_ok:
                raise RuntimeError(
                    "V24: no valid non-black frame after 6 attempts at frame %d" % (i + 1)
                )

            if frame.shape != (ny, nx):
                frame = cv2.resize(frame, (nx, ny), interpolation=cv2.INTER_AREA)

            z[:, :, i] = frame
            t[i] = (time.time() + t0) / 2.0

            if (local_i + 1) % 5 == 0 or local_i + 1 == frame_count:
                rss_now, rss_peak = _v23_mem_mb()
                logger.debug(
                    "V24 WORKER MEMORY: buffer=%d frames=%d..%d rss_now=%.1f MiB rss_peak=%.1f MiB",
                    buf, frame_start + 1, i + 1, rss_now, rss_peak,
                )

        logger.info(
            "V24 WORKER COMPLETE: buffer=%d frames=%d..%d elapsed=%.3f sec",
            buf, frame_start + 1, frame_start + frame_count, time.time() - batch_start,
        )
    except Exception as e:
        logger.error("V24 WORKER ERROR buffer=%d frames=%d..%d: %s: %s",
                     buf, frame_start + 1, frame_start + frame_count, type(e).__name__, e)
        logger.error("V24 WORKER TRACEBACK:\n%s", traceback.format_exc())
        raise
    finally:
        try:
            if camera is not None:
                camera.stop_video_capture()
        except Exception:
            pass
        try:
            if camera is not None:
                camera.close()
        except Exception:
            pass
        gc.collect()
        _v23_malloc_trim()
        rss_now, rss_peak = _v23_mem_mb()
        logger.info(
            "V24 WORKER EXIT: buffer=%d chunk=%d..%d rss_now=%.1f MiB peak=%.1f MiB",
            buf, frame_start + 1, frame_start + frame_count, rss_now, rss_peak,
        )

def _v23_malloc_trim():
    return _v22_malloc_trim()

def capture_asi(free_queue, ready_queue, z1base, t1base, z2base, t2base, nx, ny, nz, tend, device_id, live, conf_file, camera_serial):
    global logger
    logger = setup_logging(os.getcwd())
    capture_start = time.time()
    buffer_sequence = 0
    total_frames = 0
    reason = "Unknown"

    logger.info("V24 CAPTURE START: now=%.3f tend=%.3f remaining=%.3f sec",
                capture_start, tend, tend - capture_start)
    logger.info("STVID configured frame size: %dx%d, %d frames/buffer", nx, ny, nz)

    try:
        while time.time() < tend:
            buf = free_queue.get()
            buffer_sequence += 1
            logger.debug("V24 GOT FREE BUFFER %d; starting isolated SDK worker sequence=%d", buf, buffer_sequence)

            if buf == 1:
                zbase, tbase = z1base, t1base
            elif buf == 2:
                zbase, tbase = z2base, t2base
            else:
                raise RuntimeError("Invalid free buffer number: %s" % buf)

            # IMPORTANT V24 MEMORY FIX:
            # Do NOT keep one PySVB SDK process alive for all 100 frames.
            # Restart it every 10 frames. The shared STVID buffer remains the
            # same, so FITS/compressor behaviour is unchanged, while the OS
            # reclaims the SDK's per-frame allocations at every worker exit.
            chunk_size = 10
            completed = 0
            while completed < nz:
                count = min(chunk_size, nz - completed)
                chunk_worker = multiprocessing.Process(
                    target=_v24_capture_one_chunk,
                    name="svbony_buffer_%d_chunk_%d" % (buf, completed // chunk_size + 1),
                    args=(buf, zbase, tbase, nx, ny, nz, completed, count,
                          conf_file, buffer_sequence, camera_serial)
                )
                chunk_worker.start()
                chunk_worker.join()

                logger.info(
                    "V24 WORKER JOIN: buffer=%d chunk=%d..%d pid=%s exitcode=%s",
                    buf, completed + 1, completed + count,
                    chunk_worker.pid, chunk_worker.exitcode,
                )
                if chunk_worker.exitcode != 0:
                    raise RuntimeError(
                        "V24 SVBONY buffer worker failed: buffer=%d frames=%d..%d exitcode=%s" %
                        (buf, completed + 1, completed + count, chunk_worker.exitcode)
                    )
                completed += count

            total_frames += nz
            ready_queue.put(buf)
            logger.debug("V24 READY: buffer=%d total_buffers=%d total_frames=%d",
                         buf, buffer_sequence, total_frames)

        reason = "Session time expired"
    except KeyboardInterrupt:
        reason = "Keyboard interrupt"
    except Exception as e:
        reason = "Unhandled exception: %s: %s" % (type(e).__name__, e)
        logger.error("V24 CAPTURE UNHANDLED EXCEPTION: %s", reason)
        logger.error("V24 CAPTURE TRACEBACK:\n%s", traceback.format_exc())
    finally:
        rss_now, rss_peak = _v23_mem_mb()
        logger.info("V24 CAPTURE EXIT: reason=%s elapsed=%.3f sec buffers=%d frames=%d",
                    reason, time.time() - capture_start, buffer_sequence, total_frames)
        logger.info("V24 CAPTURE FINAL RSS: now=%.1f MiB peak=%.1f MiB", rss_now, rss_peak)
        logger.info("Capture V24: %s - Exiting", reason)
def compress(ready_queue, free_queue, z1base, t1base, z2base, t2base, nx, ny, nz, tend, path, device_id, conf_file):
    """ compress: Aggregate nframes of observations into a single FITS file, with statistics.

        ImageHDU[0]: mean pixel value nframes         (zmax)
        ImageHDU[1]: standard deviation of nframes    (zstd)
        ImageHDU[2]: maximum pixel value of nframes   (zmax)
        ImageHDU[3]: maximum pixel value frame number (znum)

    Also updates a [observations_path]/control/state.txt for interfacing with satttools/runsched and sattools/slewto
    """
    global logger
    logger = setup_logging(os.getcwd())

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(conf_file)

    z1 = np.ctypeslib.as_array(z1base.get_obj()).reshape(ny, nx, nz)
    t1 = np.ctypeslib.as_array(t1base.get_obj())
    z2 = np.ctypeslib.as_array(z2base.get_obj()).reshape(ny, nx, nz)
    t2 = np.ctypeslib.as_array(t2base.get_obj())
    
    # Start a new observation when STVID requests a restart.
    controlpath = os.path.join(path, "control")
    if not os.path.exists(controlpath):
        try:
            os.makedirs(controlpath)
        except PermissionError:
            logger.error("Can not create control path directory: %s" % controlpath)
            raise
    if not os.path.exists(os.path.join(controlpath, "position.txt")):
        with open(os.path.join(controlpath, "position.txt"), "w") as fp:
            fp.write("\n")
            
    with open(os.path.join(controlpath, "state.txt"), "w") as fp:
        fp.write("restart\n")

    try:
        # Main compression loop.
        while True:
            # Check whether the observation control file requests a restart.
            restart = False
            with open(os.path.join(controlpath, "state.txt"), "r") as fp:
                line = fp.readline().rstrip()
                if line == "restart":
                    restart = True

            # Start/restart the current observation.
            if restart:
                # Record the observing state.
                with open(os.path.join(controlpath, "state.txt"), "w") as fp:
                    fp.write("observing\n")

                # Build the STVID observation identifier and create its output directory.
                trestart = time.gmtime()
                obsid = "%s_%d/%s" % (time.strftime("%Y%m%d", trestart), device_id, time.strftime("%H%M%S", trestart))
                filepath = os.path.join(path, obsid)
                logger.info("Storing files in %s" % filepath)

                # Create the output directory if necessary.
                if not os.path.exists(filepath):
                    try:
                        os.makedirs(filepath)
                    except PermissionError:
                        logger.error("Can not create output directory: %s" % filepath)
                        raise

                # Copy the current mount position into the new observation directory.
                with open(os.path.join(controlpath, "position.txt"), "r") as fp:
                    line = fp.readline()
                with open(os.path.join(filepath, "position.txt"), "w") as fp:
                    fp.write(line)

            # Wait for a buffer that capture has completely filled.
            # The capture process sends no end marker itself.
            # The main process sends None only after capture has really exited, so the
            # compressor can always finish processing the last ready buffer first.
            proc_buffer = ready_queue.get()

            # None is the explicit end-of-capture marker; 1 and 2 are the only buffers.
            if proc_buffer is None:
                logger.debug("Received end-of-capture marker")
                break

            logger.debug("Processing buffer %d" % proc_buffer)

            # Record processing start time for this buffer.
            tstart = time.time()

            # Select the shared arrays belonging to the completed buffer.
            if proc_buffer == 1:
                t = t1                
                z = z1
            elif proc_buffer == 2:
                t = t2
                z = z2

            # Convert the first frame timestamp into the FITS DATE-OBS time format.
            nfd = "%s.%03d" % (time.strftime("%Y-%m-%dT%T",
                                             time.gmtime(t[0])), int((t[0] - np.floor(t[0])) * 1000))
            t0 = Time(nfd, format="isot")
            dt = t - t[0]

            # Measure statistics and FITS I/O separately so timing problems can be diagnosed.
            # This makes it clear whether delays come from NumPy processing or FITS writing.
            # NumPy statistics or by FITS file creation/writing.
            t_stats = time.time()

            # Input pixels are uint8. Integer accumulators avoid unnecessary float memory use.
            # The final 2-D products are converted to float32 only after accumulation.
            #
            # uint32 is sufficient for 100 frames because the maximum possible sums are small:
            #   sum   <= 100 * 255       = 25,500
            #   sumsq <= 100 * 255^2     = 6,502,500
            zmax = np.zeros((ny, nx), dtype=np.uint8)
            znum = np.zeros((ny, nx), dtype=np.int16)
            zsum = np.zeros((ny, nx), dtype=np.uint32)
            zsum2 = np.zeros((ny, nx), dtype=np.uint32)

            for j in range(nz):
                frame = z[:, :, j]

                # Track both the maximum pixel value and the frame containing that maximum.
                mask = frame > zmax
                zmax[mask] = frame[mask]
                znum[mask] = j

                # Accumulate integer sums. Cast before squaring so uint8 cannot overflow.
                # square cannot overflow uint8.
                frame32 = frame.astype(np.uint32)
                zsum += frame32
                zsum2 += frame32 * frame32

            t_stats_done = time.time()

            # Remove the maximum frame exactly as the original STVID algorithm does.
            zmax32 = zmax.astype(np.uint32)
            zs1 = zsum - zmax32
            zs2 = zsum2 - zmax32 * zmax32

            zavg = zs1.astype(np.float32) / float(nz - 1)
            zstd = np.sqrt(
                (zs2.astype(np.float32) -
                 zs1.astype(np.float32) * zavg) / float(nz - 2)
            ).astype(np.float32)

            t_stats_final = time.time()

            logger.info(
                "Timing buffer %d: statistics accumulation %.3f sec, "
                "statistics finalization %.3f sec",
                proc_buffer,
                t_stats_done - t_stats,
                t_stats_final - t_stats_done
            )

            # Convert the calculated products to FITS output types and flip them vertically.
            zmax = np.flipud(zmax.astype("float32"))
            znum = np.flipud(znum.astype("float32"))
            zavg = np.flipud(zavg.astype("float32"))
            zstd = np.flipud(zstd.astype("float32"))

            # Prepare temporary and final FITS filenames.
            t_fits = time.time()
            ftemp = "%s.temp" % nfd.replace(":", "-")
            fname = "%s.fits" % nfd.replace(":", "-")

            # Build the FITS header with observation timing, geometry and observer metadata.
            hdr = fits.Header()
            hdr["DATE-OBS"] = "%s" % nfd
            hdr["MJD-OBS"]  = t0.mjd
            hdr["EXPTIME"]  = dt[-1] - dt[0]
            hdr["NFRAMES"]  = nz
            hdr["CRPIX1"]   = float(nx) / 2
            hdr["CRPIX2"]   = float(ny) / 2
            hdr["CRVAL1"]   = 0.0
            hdr["CRVAL2"]   = 0.0
            hdr["CD1_1"]    = 1 / 3600
            hdr["CD1_2"]    = 0.0
            hdr["CD2_1"]    = 0.0
            hdr["CD2_2"]    = 1 / 3600
            hdr["CTYPE1"]   = "RA---TAN"
            hdr["CTYPE2"]   = "DEC--TAN"
            hdr["CUNIT1"]   = "deg"
            hdr["CUNIT2"]   = "deg"
            hdr["CRRES1"]   = 0.0
            hdr["CRRES2"]   = 0.0
            hdr["EQUINOX"]  = 2000.0
            hdr["RADECSYS"] = "ICRS"
            hdr["COSPAR"]   = cfg.getint("Observer", "cospar")
            hdr["OBSERVER"] = cfg.get("Observer", "name")
            hdr["SITELONG"] = cfg.getfloat("Observer", "longitude")
            hdr["SITELAT"] = cfg.getfloat("Observer", "latitude")
            hdr["ELEVATIO"] = cfg.getfloat("Observer", "height")
            if cfg.getboolean("Setup", "tracking_mount"):
                hdr["TRACKED"] = 1
            else:
                hdr["TRACKED"] = 0
            for i in range(nz):
                hdr["DT%04d" % i] = dt[i]
            for i in range(10):
                hdr["DUMY%03d" % i] = 0.0

            # Write to a temporary file first, then rename it to the final FITS filename.
            hdu = fits.PrimaryHDU(data=np.array([zavg, zstd, zmax, znum]),
                                  header=hdr)
            hdu.writeto(os.path.join(filepath, ftemp))
            os.rename(os.path.join(filepath, ftemp), os.path.join(filepath, fname))

            t_fits_done = time.time()
            logger.info(
                "Timing buffer %d: FITS create/write %.3f sec",
                proc_buffer,
                t_fits_done - t_fits
            )
            logger.info("Compressed %s in %.2f sec" % (fname, t_fits_done - tstart))

            # The compressor has completely finished reading this shared buffer.
            # Only now is it safe for capture to reuse that buffer.
            free_queue.put(proc_buffer)
            logger.debug("Released buffer %d back to capture", proc_buffer)

            # Do not shut down because the last frame timestamp is old.
            # The main process sends None only after capture has actually exited.
            # That explicit marker is the compressor's sole shutdown condition.
            # shutdown condition for this process.
            logger.debug("V10 processed buffer %d; waiting for next ready buffer", proc_buffer)
            

    except KeyboardInterrupt:
        logger.info("V10 COMPRESS: KeyboardInterrupt")
    except MemoryError as e:
        logger.error("V10 COMPRESS MEMORYERROR: %s", e)
        logger.error("V10 COMPRESS TRACEBACK:\n%s", traceback.format_exc())
    except Exception as e:
        logger.error(
            "V10 COMPRESS UNHANDLED EXCEPTION: %s: %s",
            type(e).__name__, e
        )
        logger.error("V10 COMPRESS TRACEBACK:\n%s", traceback.format_exc())
        raise
    finally:
        # Compression process finished.
        logger.info("Exiting compress")


# Main program entry point.
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    
    # Define the command-line options.
    conf_parser = argparse.ArgumentParser(description="SVBONY V10 debug capture: isolated SDK worker per STVID buffer.")
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify configuration file(s). If no file" +
                             " is specified 'svbconfiguration.ini' is used.",
                             action="append",
                             nargs="?",
                             metavar="FILE")
    conf_parser.add_argument("-t", "--test", 
                             nargs="?",
                             action="store", 
                             default=False,
                             help="Testing mode - start immediately for (optional) seconds; omit -t for normal night operation",
                             metavar="s")
    conf_parser.add_argument("-l", "--live", action="store_true",
                             help="Display live image while capturing")

    args = conf_parser.parse_args()

    # Read the selected configuration file and validate that it can be opened.
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    
    conf_file = args.conf_file if args.conf_file else "svbconfiguration.ini"
    result = cfg.read(conf_file)

    if not result:
        print("Could not read config file: %s\nExiting..." % conf_file)
        sys.exit()

    # Configure the main log file and console logging.
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] " +
                                     "[%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    # Create the configured observations directory if it does not exist.
    path = os.path.abspath(cfg.get("Setup", "observations_path"))
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except PermissionError:
            logger.error("Can not create observations_path: %s" % path)
            sys.exit()

    fileHandler = logging.FileHandler(os.path.join(path, "acquire.log"))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)

    logger.info("Using config: %s" % conf_file)

    # Interpret test-mode options. Without -t, normal sunset/sunrise scheduling is used.
    if args.test is None:
        test_duration = 31
        testing = True
    elif args.test is not False:
        test_duration = int(args.test)
        testing = True
    else:
        testing = False
    logger.info("Test mode: %s" % testing)
    if (testing):
        logger.info("Test duration: %ds" % test_duration)

    # Interpret the optional live-display flag.
    live = True if args.live else False
    logger.info("Live mode: %s" % live)
    if not testing:
        logger.info("SVBONY night mode: normal sunset/sunrise scheduling; FITS capture will run through the night")

    # Select the camera backend named by Setup.camera_type.
    camera_type = cfg.get("Setup", "camera_type")

    # V10.2 SVBONY backend: all SVBONY acquisition settings come from [SVB].
    # The physical camera is identified by its serial number, never by SDK index.
    # device_id is only the historical STVID observation-directory number; the
    # actual SVBONY SDK CameraID is obtained after selecting the configured serial.
    if camera_type != "SVB":
        raise RuntimeError(
            "V10.2 SVBONY backend requires Setup.camera_type=SVB; got %s"
            % camera_type
        )
    if not cfg.has_section("SVB"):
        raise RuntimeError("Setup.camera_type=SVB but [SVB] section is missing")
    if not cfg.has_option("SVB", "serial"):
        raise RuntimeError("Setup.camera_type=SVB requires [SVB] serial")

    camera_serial = cfg.get("SVB", "serial").strip()
    if not camera_serial:
        raise RuntimeError("[SVB] serial must not be empty")

    logger.info("V10.2 SVBONY camera selector: %s", camera_serial)

    # Keep STVID output-directory numbering independent of the SVBONY SDK CameraID.
    device_id = 0

    # Record the current time for scheduling and FITS timestamps.
    tnow = Time.now()

    # Configure the observing location used by STVID's day/night scheduling.
    loc = EarthLocation(lat=cfg.getfloat("Observer", "latitude") * u.deg,
                        lon=cfg.getfloat("Observer", "longitude") * u.deg,
                        height=cfg.getfloat("Observer", "height") * u.m)

    if not testing:
        # Read the configured sunset and sunrise reference altitudes.
        refalt_set  = cfg.getfloat("Setup", "alt_sunset") * u.deg
        refalt_rise = cfg.getfloat("Setup", "alt_sunrise") * u.deg

        # Optional aimpoint used by STVID's observing scheduler.
        if cfg.has_section("Aimpoint"):
            aimpoint_az = cfg.getfloat("Aimpoint", "az_deg") * u.deg
            aimpoint_alt = cfg.getfloat("Aimpoint", "alt_deg") * u.deg
            aimpoint_height = cfg.getfloat("Aimpoint", "height_km") * u.km
        else:
            aimpoint_az, aimpoint_alt, aimpoint_height = None, None, None

        # Ask STVID whether acquisition should start now or wait for the next observing window.
        action, wait_time, tend, state = observe_logic(tnow, loc, refalt_set, refalt_rise,
                                                       aimpoint_az, aimpoint_alt, aimpoint_height)

        # In normal mode, wait until the scheduler says the observation window begins.
        logger.info(state)
        if action == "wait":
            logger.info(f"Waiting for {wait_time:.0f} seconds.")
            try:
                time.sleep(wait_time)
            except KeyboardInterrupt:
                sys.exit()
    else:
        tend = tnow + test_duration * u.s

    # Optional GPIO-controlled shutter.
    if cfg.has_section("Shutter"):
        from stvid.shutter import Shutter
        shutter = Shutter(cfg.getint("Shutter", "pin"))
    else:
        shutter = None
        
    logger.info("Starting data acquisition")
    logger.info("Acquisition will end after "+tend.isot)

    # Read the STVID frame dimensions and number of frames per buffer.
    nx = cfg.getint(camera_type, "nx")
    ny = cfg.getint(camera_type, "ny")
    nz = cfg.getint(camera_type, "nframes")

    # Allocate the two shared image buffers and their timestamp arrays.
    z1base = multiprocessing.Array(ctypes.c_uint8, nx * ny * nz)
    t1base = multiprocessing.Array(ctypes.c_double, nz)
    z2base = multiprocessing.Array(ctypes.c_uint8, nx * ny * nz)
    t2base = multiprocessing.Array(ctypes.c_double, nz)

    # Two-buffer handshake:
    #   free_queue  = buffers capture may write into.
    #   ready_queue = buffers that capture has completely filled.
    # The compressor releases a buffer only after FITS processing is finished.
    # Both buffers start as free, so capture can immediately use either one.
    # Because ready_queue has capacity 1, capture cannot build an unbounded backlog.
    free_queue = multiprocessing.Queue(maxsize=2)
    ready_queue = multiprocessing.Queue(maxsize=1)
    free_queue.put(1)
    free_queue.put(2)

    # Create the compressor and capture processes.
    pcompress = multiprocessing.Process(target=compress,
                                        name="compress",
                                        args=(ready_queue, free_queue,
                                              z1base, t1base, z2base, t2base,
                                              nx, ny, nz, tend.unix,
                                              path, device_id, conf_file))
    if camera_type == "PI":
        pcapture = multiprocessing.Process(target=capture_pi,
                                           name="capture_pi",
                                           args=(image_queue,
                                                 z1base, t1base, z2base, t2base,
                                                 nx, ny, nz, tend.unix,
                                                 device_id, live, conf_file, camera_serial))
    elif camera_type == "CV2":
        pcapture = multiprocessing.Process(target=capture_cv2,
                                           name="capture_cv2",
                                           args=(image_queue,
                                                 z1base, t1base, z2base, t2base,
                                                 nx, ny, nz, tend.unix,
                                                 device_id, live, conf_file))
    elif camera_type == "SVB":
        pcapture = multiprocessing.Process(target=capture_asi,
                                           name="capture_svb",
                                           args=(free_queue, ready_queue,
                                                 z1base, t1base, z2base, t2base,
                                                 nx, ny, nz, tend.unix,
                                                 device_id, live, conf_file, camera_serial))

    try:
        # Open the optional shutter before starting acquisition.
        if shutter:
            shutter.open()
        
        # Start capture and compression.
        pcapture.start()
        pcompress.start()
        logger.info(
            "V10 MAIN: started capture pid=%s compressor pid=%s",
            pcapture.pid, pcompress.pid
        )

        # Shutdown sequence: capture must finish before the compressor is told to stop.
        # This guarantees that the final completed buffer is processed.
        # It also replaces the older timeout-based shutdown behaviour, which could
        # leave the compressor waiting on a queue timeout at the end of a session.
        # "Ready-buffer queue timed out" shutdown.
        try:
            pcapture.join()
            logger.info(
                "V10 MAIN: capture process joined; exitcode=%s",
                pcapture.exitcode
            )

            # None is never a valid buffer number; it is reserved as the shutdown marker.
            logger.debug("V10 MAIN: sending end-of-capture marker to compressor")
            ready_queue.put(None)
            logger.debug("V10 MAIN: end-of-capture marker sent")

            pcompress.join()
            logger.info(
                "V10 MAIN: compressor process joined; exitcode=%s",
                pcompress.exitcode
            )
        except (KeyboardInterrupt, ValueError):
            time.sleep(0.1) # Allow a short grace period for a clean shutdown
        except MemoryError as e:
            logger.error("Memory error %s" % e)
        finally:
            if pcapture.is_alive():
                pcapture.terminate()
            if pcompress.is_alive():
                pcompress.terminate()

        # Close any optional live-display window and finish shutdown.
        if live is True:
            cv2.destroyAllWindows()
    finally:
        if shutter:
            shutter.close()
