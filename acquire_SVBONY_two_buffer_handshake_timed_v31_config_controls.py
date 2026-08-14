#!/usr/bin/env python3
# SVBONY two-buffer handshake timed v28 DEBUG: PySVB owner + native GetVideoData
# Each STVID buffer gets its own short-lived worker. PySVB owns the COMPLETE
# camera lifecycle AND frame delivery: OPEN/START/GET/STOP/CLOSE.
# This deliberately abandons the native ctypes GetVideoData experiment because
# the SVB SDK returns status=2 (INVALID_ID) when that entry point is called
# separately from the proven PySVB capture path.
# Based directly on V6; image/data path intentionally unchanged.
import sys
import os
import numpy as np
import cv2
import time
import ctypes

# V29: use the ACTUAL SVBONY native SDK for camera lifecycle AND frame delivery.
# PySVB is retained only for discovery/property/ROI compatibility; native SDK owns open/controls/start/stop/close.
# We only call SVBGetVideoData() to write frames into one caller-owned buffer.
# V11 accidentally loaded the ZWO ASI library; V12+ established that the
# correct library is libSVBCameraSDK.so.
# IMPORTANT V25 FIX:
# multiprocessing uses spawn, so the child re-imports this module.  The old
# V25 tried to load libSVBCameraSDK.so at module import time, BEFORE libusb was
# made RTLD_GLOBAL in the worker.  In the spawned worker that load could fail,
# leaving _svb_native=None even though the same SDK works normally.
# Load the SVBONY native library only after libusb has been loaded globally.
_svb_native = None
_svb_native_path = None

# V29: make libusb globally visible before loading the SVBONY native SDK.
# This is important in multiprocessing/spawn workers because the SVBONY
# shared library depends on libusb symbols.
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

            # V29: native SDK owns the camera lifecycle used by GetVideoData.
            for _fname in ("SVBOpenCamera", "SVBStartVideoCapture",
                           "SVBStopVideoCapture", "SVBCloseCamera"):
                _fn = getattr(_svb_native, _fname)
                _fn.argtypes = [ctypes.c_int]
                _fn.restype = ctypes.c_int

            _svb_native.SVBSetControlValue.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_int
            ]
            _svb_native.SVBSetControlValue.restype = ctypes.c_int

            print("SVBONY V31 native library:", _svb_native_path, flush=True)
            return _svb_native
        except OSError as exc:
            last_error = exc

    raise OSError(
        "Could not load libSVBCameraSDK.so for V31 native capture: %s"
        % last_error
    )

# V22: PySVB is the only proven-working camera access path.  Its
# get_video_data() allocates a large temporary RGBA object per frame.
# glibc can keep those freed multi-megabyte blocks in the process RSS.
# Ask glibc to return free heap pages periodically; this does NOT touch
# the SVBONY camera buffer or the STVID shared buffers.
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

# Capture images from pi
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
    
    # Intialization
    first = True
    slow_CPU = False

    # Initialize cv2 device
    camera = PiCamera(sensor_mode=2)
    camera.resolution = (nx, ny)    
    # Turn off any thing automatic.
    camera.exposure_mode = "off"        
    camera.awb_mode = "off"
    # ISO needs to be 0 otherwise analog and digital gain won't work.
    camera.iso = 0
    # set the camea settings
    camera.framerate = cfg.getfloat(camera_type, "framerate")
    camera.awb_gains = (cfg.getfloat(camera_type, "awb_gain_red"), cfg.getfloat(camera_type, "awb_gain_blue"))    
    camera.analog_gain = cfg.getfloat(camera_type, "analog_gain")
    camera.digital_gain = cfg.getfloat(camera_type, "digital_gain")
    camera.shutter_speed = cfg.getint(camera_type, "exposure")

    rawCapture = PiRGBArray(camera, size=(nx, ny))
    # allow the camera to warmup
    time.sleep(0.1)

    try:
        # Loop until reaching end time
        while float(time.time()) < tend:
            # Get frames
            i = 0
            for frameA in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                            
                # Store start time
                t0 = float(time.time())                
                # grab the raw NumPy array representing the image, then initialize the timestamp                
                frame = frameA.array
                                    
                # Compute mid time
                t = (float(time.time()) + t0) / 2
                
                # Skip lost frames
                if frame is not None:
                    # Convert image to grayscale
                    z = np.asarray(cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY)).astype(np.uint8)
                    # optionally rotate the frame by 2 * 90 degrees.    
                    # z = np.rot90(z, 2)
                
                    # Display Frame
                    if live is True:                            
                        cv2.imshow("Capture", z)    
                        cv2.waitKey(1)
                    
                    # Store results
                    if first:
                        z1[:, :, i] = z
                        t1[i] = t
                    else:
                        z2[:, :, i] = z
                        t2[i] = t
                        
                # clear the stream in preparation for the next frame
                rawCapture.truncate(0)
                # count up to nz frames, then break out of the for loop.
                i += 1
                if i >= nz:
                    break
                
            if first: 
                buf = 1
            else:
                buf = 2
            image_queue.put(buf)
            logger.debug("Captured buffer %d" % buf)

            # Swap flag
            first = not first
        reason = "Session complete"
    except KeyboardInterrupt:
        print()
        reason = "Keyboard interrupt"
    except ValueError as e:
        logger.error("%s" % e)
        reason = "Wrong image dimensions? Fix nx, ny in config."
    finally:
        # End capture
        logger.info("Capture: %s - Exiting" % reason)
        camera.close()



# Capture images from cv2
def capture_cv2(image_queue, z1base, t1base, z2base, t2base, nx, ny, nz, tend, device_id, live, conf_file):
    global logger
    logger = setup_logging(os.getcwd())

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(conf_file)

    z1 = np.ctypeslib.as_array(z1base.get_obj()).reshape(ny, nx, nz)
    t1 = np.ctypeslib.as_array(t1base.get_obj())
    z2 = np.ctypeslib.as_array(z2base.get_obj()).reshape(ny, nx, nz)
    t2 = np.ctypeslib.as_array(t2base.get_obj())
    
    # Intialization
    camera_type  = "CV2"
    first = True
    slow_CPU = False

    # Initialize cv2 device
    if cfg.has_option(camera_type, "device_string"):
        device = cv2.VideoCapture(cfg.get(camera_type, "device_string"))
    else:
        device = cv2.VideoCapture(device_id)

    # Test for software binning
    try:
        software_bin = cfg.getint(camera_type, "software_bin")
    except configparser.Error:
        software_bin = 1
    
    # Set properties
    device.set(3, nx * software_bin)
    device.set(4, ny * software_bin)
   
    try:
        # Loop until reaching end time
        while float(time.time()) < tend:
            # The queue is deliberately limited to one completed buffer.
            # This prevents the capture process from getting ahead of the
            # FITS writer while the two shared image buffers are reused.

            # Get frames
            for i in range(nz):
                # Store start time
                t0 = float(time.time())

                # Get frame
                res, frame = device.read()

                # Compute mid time
                t = (float(time.time()) + t0) / 2

                # Skip lost frames
                if res is True:
                    # Convert image to grayscale
                    z = np.asarray(cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY)).astype(np.uint8)

                    # Match the STVID configured frame size if the full-sensor
                # SVBONY frame has a different size.
                if z is not None and z.shape != (ny, nx):
                    z = cv2.resize(z, (nx, ny), interpolation=cv2.INTER_AREA)

                # Apply software binning
                    if software_bin > 1:
                        my, mx = z.shape
                        z = cv2.resize(z, (mx // software_bin, my // software_bin))
                    
                    # Display Frame
                    if live is True:
                        cv2.imshow("Capture", z)
                        cv2.waitKey(1)

                    # Store results
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

            # Swap flag
            first = not first
        reason = "Session complete"
    except KeyboardInterrupt:
        print()
        reason = "Keyboard interrupt"
    except ValueError as e:
        logger.error("%s" % e)
        reason = "Wrong image dimensions? Fix nx, ny in config."
    finally:
        # End capture
        logger.info("Capture: %s - Exiting" % reason)
        device.release()


# Capture images
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

    V31 uses PySVB as the SOLE owner of the camera lifecycle. Native ctypes
    is used only for SVBGetVideoData() into one persistent caller-owned buffer.
    """

    def __init__(self, requested_width, requested_height, exposure_us, gain):
        from pysvb.camera import (
            PySVBCameraSDK,
            SVB_CAMERA_MODE,
            SVB_ROI_FORMAT,
            SVB_CONTROL_TYPE,
        )

        self.sdk = PySVBCameraSDK()
        print("SVBONY V29: PySVB is sole owner of camera lifecycle and frame delivery", flush=True)
        self.SVB_CAMERA_MODE = SVB_CAMERA_MODE
        self.SVB_ROI_FORMAT = SVB_ROI_FORMAT
        self.SVB_CONTROL_TYPE = SVB_CONTROL_TYPE

        connected = self.sdk.get_num_of_connected_cameras()
        print("SVBONY SDK version:", self.sdk.sdk_version)
        print("SVBONY connected cameras:", connected)
        if connected <= 0:
            raise RuntimeError("Geen SVBONY-camera gevonden")

        info = self.sdk.get_camera_info(0)
        self.device_id = info.CameraID
        print("SVBONY camera:", info.FriendlyName)
        print("SVBONY camera ID:", self.device_id)
        print("SVBONY serial:", info.CameraSN)

        # V31 CRITICAL LIFECYCLE EXPERIMENT:
        # There is deliberately no native SVBOpenCamera/SVBStartVideoCapture.
        # Earlier experiments showed status=2 from the second/native path.
        self.sdk.open_camera(self.device_id)
        print("SVBONY V29: PySVB camera opened for setup", flush=True)
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

        # IMPORTANT:
        # Use the frame dimensions selected by STVID/configuration.ini.
        # Do not force the SV305M PRO to stream its full 1920x1080 sensor
        # when STVID is configured for a smaller observation frame.
        #
        # This reduces USB traffic, SDK memory traffic and the amount of
        # data that has to be converted before it reaches the STVID buffer.
        requested_width = int(requested_width)
        requested_height = int(requested_height)

        if requested_width <= 0 or requested_height <= 0:
            raise ValueError(
                "Invalid configured frame size: %dx%d"
                % (requested_width, requested_height)
            )

        if requested_width > self.width or requested_height > self.height:
            raise ValueError(
                "Configured frame size %dx%d exceeds SVBONY sensor %dx%d"
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

        # V31: exposure and gain are supplied from configuration.ini.
        self.exposure_us = int(exposure_us)
        self.gain = int(gain)

        if self.exposure_us <= 0:
            raise ValueError("Invalid SVBONY exposure: %d us" % self.exposure_us)
        if self.gain < 0:
            raise ValueError("Invalid SVBONY gain: %d" % self.gain)

        # Configure everything that requires the PySVB handle first.
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

        # V29: KEEP THIS PySVB HANDLE OPEN.
        # V17 showed the correct ownership model for the GetVideoData test:
        # PySVB remains the sole camera owner. Native ctypes must NOT call
        # SVBOpenCamera/SVBCloseCamera, because that creates a second camera
        # lifecycle on the same device and produces status=2.

        # V29: DO NOT call native SVBOpenCamera here.
        #
        # The experiments showed that SVBOpenCamera(1) returns status=2 when
        # mixed with the PySVB ownership model.  V17 deliberately solved this
        # by making PySVB the SOLE owner of the camera lifecycle.  Native ctypes
        # is used only for SVBGetVideoData(), writing into our persistent caller
        # owned buffer.
        #
        # Keep the PySVB handle open from here until close().
        self.native_open = False

        # V29: native GetVideoData is allowed to use the SAME camera handle
        # already opened and started by PySVB.  No second native handle is
        # created.
        # V6: get_video_data() has been verified from the raw hex dump to
        # return RGBA8 pixels (4 bytes/pixel), despite MaxBitDepth reporting 16.
        self.bytes_per_pixel = 4
        self.real_frame_size = int(
            self.bytes_per_pixel * self.roi.width * self.roi.height
        )
        # Same safety margin as the working demo.
        self.sdk_buffer_size = self.real_frame_size  # V19: request exactly one RGBA8 frame
        self.wait_ms = 5000

        # V25: one persistent caller-owned RGBA8 buffer.
        self.raw_buffer = (ctypes.c_ubyte * self.real_frame_size)()
        self.raw_ptr = ctypes.cast(
            self.raw_buffer, ctypes.POINTER(ctypes.c_ubyte)
        )

        # DEBUG: print statistics for only the first four raw frames.
        # Comment out this assignment to disable the diagnostic below.
        self.debug_frame_count = 0
        # Reusable 2-D monochrome output.  capture_video_frame() writes
        # into this array instead of allocating a new image every frame.
        self.output_frame = np.empty(
            (int(self.roi.height), int(self.roi.width)), dtype=np.uint8
        )
        self.frame_counter = 0
        self.native_started = False

        print("SVBONY V6 data format: RGBA8, bytes/pixel:", self.bytes_per_pixel)
        print("SVBONY real frame size:", self.real_frame_size)
        print("SVBONY SDK buffer size:", self.sdk_buffer_size)

    def native_set_control(self, ctrl_type, value, auto=False):
        # V29: PySVB owns camera controls. Keep this compatibility helper,
        # but route the actual control operation through PySVB.
        return self.sdk.set_control_value(
            self.device_id, ctrl_type, int(value), bool(auto)
        )

    def init(self, sdk=None):
        # STVID calls asi.init() because the original backend is ASI.
        # The PySVB SDK is already initialized in __init__.
        return True

    def get_num_cameras(self):
        return 1

    def list_cameras(self):
        return [getattr(self.props, "FriendlyName", "SVBONY")]

    def Camera(self, device_id):
        # The bridge represents the single opened SVBONY camera itself.
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
        # Not applicable to the SVBONY SDK path.
        return True

    def set_control_value(self, ctrl_type, value, auto=False):
        # Map the old ASI-style control IDs to the real SVBONY enums.
        if ctrl_type == 0:
            ctrl_type = self.SVB_CONTROL_TYPE.SVB_GAIN
        elif ctrl_type == 1:
            ctrl_type = self.SVB_CONTROL_TYPE.SVB_EXPOSURE
        else:
            # ASI-only controls (white balance, gamma, bandwidth, etc.)
            # have no direct equivalent in this SVBONY test bridge.
            return True
        if ctrl_type == self.SVB_CONTROL_TYPE.SVB_GAIN:
            self.gain = int(value)
        elif ctrl_type == self.SVB_CONTROL_TYPE.SVB_EXPOSURE:
            self.exposure_us = int(value)
        return self.native_set_control(ctrl_type, value, auto)

    def get_control_values(self):
        # The existing STVID code needs Gain and Temperature.
        # Keep these from the values explicitly set by this bridge.
        return {"Gain": self.gain, "Temperature": 0}

    def set_roi(self, bins=1):
        # The bridge already configured the full sensor ROI. Keep this
        # method for compatibility with the original STVID code.
        return True

    def set_image_type(self, image_type):
        # The PySVB wrapper supplies the native camera data format.
        return True

    def start_video_capture(self):
        # V29: PySVB is the sole owner of OPEN/START/STOP/CLOSE.
        print("SVBONY V29: restoring test exposure/gain immediately before PySVB start:",
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
        print("SVBONY V31 PySVB start_video_capture result:", result, flush=True)
        self.native_started = True
        return result

    def stop_video_capture(self):
        try:
            result = self.sdk.stop_video_capture(self.device_id)
            print("SVBONY V31 PySVB stop_video_capture result:", result, flush=True)
            self.native_started = False
            return result
        except Exception:
            return None

    def capture_video_frame(self):
        """Capture one frame through the SAME PySVB camera handle.

        V17/V23 established that the PySVB get_video_data() path is the
        working acquisition path on this SV305M PRO. V25/V31 proved that
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
            raise RuntimeError("V31 PySVB get_video_data returned no frame")

        self.frame_counter += 1
        frame_no = self.debug_frame_count + 1

        try:
            raw = np.frombuffer(z, dtype=np.uint8)
        except TypeError:
            raw = np.asarray(z, dtype=np.uint8).reshape(-1)

        expected = int(self.roi.width) * int(self.roi.height) * 4
        if raw.size < expected:
            raise RuntimeError(
                "V31 PySVB frame too small: got %d bytes, expected %d"
                % (raw.size, expected)
            )

        rgba = raw[:expected].reshape(
            (int(self.roi.height), int(self.roi.width), 4)
        )

        if self.debug_frame_count < 4:
            print(
                "SVBONY V31 RAW frame", frame_no,
                ": returned_len =", raw.size,
                "expected_1frame_buffer =", expected,
                "one_RGBA_frame =", expected, flush=True
            )
            print(
                "SVBONY V31 frame", frame_no,
                "first64 HEX =", bytes(raw[:64]).hex(" "), flush=True
            )
            r0, g0, b0, a0 = (rgba[:, :, 0], rgba[:, :, 1],
                              rgba[:, :, 2], rgba[:, :, 3])
            print(
                "SVBONY V31 block1: R[%d,%d,%.3f] G[%d,%d,%.3f] "
                "B[%d,%d,%.3f] A[%d,%d]" % (
                    int(r0.min()), int(r0.max()), float(r0.mean()),
                    int(g0.min()), int(g0.max()), float(g0.mean()),
                    int(b0.min()), int(b0.max()), float(b0.mean()),
                    int(a0.min()), int(a0.max())
                ), flush=True
            )
            print(
                "SVBONY V31 block1 channel differences: "
                "R-G max =", int(np.max(np.abs(
                    r0.astype(np.int16) - g0.astype(np.int16)
                ))),
                "R-B max =", int(np.max(np.abs(
                    r0.astype(np.int16) - b0.astype(np.int16)
                ))),
                "A unique =", np.unique(a0)[:10].tolist(), flush=True
            )
            self.debug_frame_count += 1

        # Copy only the monochrome R channel into the reusable STVID frame.
        # Do not retain z/raw/rgba beyond this point.
        self.output_frame[:, :] = rgba[:, :, 0]

        del rgba
        del raw
        del z

        if self.frame_counter % 10 == 0:
            collected = gc.collect()
            trimmed = _v22_malloc_trim()
            rss_now, rss_peak = _v23_mem_mb()
            logger.debug(
                "V31 MEMORY CHECK: frame=%d rss_now=%.1f MiB "
                "rss_peak=%.1f MiB gc_collected=%d malloc_trim=%d",
                self.frame_counter, rss_now, rss_peak, collected, trimmed
            )

        return self.output_frame

    def close(self):
        # V29: PySVB alone closes the camera. There is deliberately NO
        # SVBCloseCamera() call because V31 never called SVBOpenCamera().
        try:
            if getattr(self, "native_started", False):
                self.stop_video_capture()
        finally:
            try:
                self.sdk.close_camera(self.device_id)
                print("SVBONY V31 PySVB camera closed", flush=True)
            except Exception as exc:
                logger.error("V31 PySVB close_camera exception: %s", exc)

    def close_camera(self):
        return self.close()


def _v23_capture_one_buffer(buf, zbase, tbase, nx, ny, nz, conf_file, sequence):
    """Capture exactly one STVID buffer in a short-lived worker process.

    The SVBONY Python wrapper retains/allocates roughly 8 MiB per returned RGBA
    frame.  malloc_trim() cannot reclaim that growth reliably.  The operating
    system *does* reclaim it when this worker exits, so one SDK session lives
    for exactly one 100-frame STVID buffer.
    """
    global logger
    logger = setup_logging(os.getcwd())
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(conf_file)

    # Make libusb globally visible BEFORE loading either PySVB or the native
    # SVBONY library.  This ordering is required with multiprocessing spawn.
    try:
        ctypes.CDLL("libusb-1.0.so", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        ctypes.CDLL("/home/hacker3/svbony/SVBCameraSDK/lib/x64/libusb-1.0.so", mode=ctypes.RTLD_GLOBAL)

    # V29: DO NOT load/use the native SVB capture API in this worker.
    # PySVB is the sole owner of the SDK lifecycle and GetVideoData call.

    z = np.ctypeslib.as_array(zbase.get_obj()).reshape(ny, nx, nz)
    t = np.ctypeslib.as_array(tbase.get_obj())

    # V31: read the camera controls from the same configuration.ini that
    # STVID uses. This makes exposure/gain easy to change without editing
    # this Python program. Exposure is in microseconds, as in configuration.ini.
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not cfg.read(conf_file):
        raise RuntimeError("V31: could not read config file: %s" % conf_file)
    camera_type = cfg.get("Setup", "camera_type")
    configured_exposure_us = cfg.getint(camera_type, "exposure")
    configured_gain = cfg.getint(camera_type, "gain")

    logger.info(
        "V31 CONFIG: %s exposure=%d us gain=%d",
        camera_type, configured_exposure_us, configured_gain
    )

    camera = None
    try:
        bridge = SVBToASIBridge(
            nx, ny,
            exposure_us=configured_exposure_us,
            gain=configured_gain
        )
        camera = bridge.Camera(0)
        # The bridge does not define ASI_* constants. The constructor and
        # start_video_capture() apply the values read from configuration.ini.
        # Do not call ASI_GAIN / ASI_EXPOSURE / ASI_IMG_RAW8 here: those are
        # constants from the real ASI backend, not attributes of this bridge.
        camera.start_video_capture()

        logger.info("V31 WORKER START: buffer=%d sequence=%d", buf, sequence)
        batch_start = time.time()

        for i in range(nz):
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
                    logger.info("V31 worker buffer %d: recovered frame %d after %d retries", buf, i, attempt)
                break
            if not frame_ok:
                raise RuntimeError("V29: no valid non-black frame after 6 attempts at frame %d" % i)

            if frame.shape != (ny, nx):
                frame = cv2.resize(frame, (nx, ny), interpolation=cv2.INTER_AREA)

            z[:, :, i] = frame
            t[i] = (time.time() + t0) / 2.0

            if (i + 1) % 10 == 0:
                rss_now, rss_peak = _v23_mem_mb()
                logger.debug("V31 WORKER MEMORY: buffer=%d frame=%d rss_now=%.1f MiB rss_peak=%.1f MiB",
                             buf, i + 1, rss_now, rss_peak)

        logger.info("V31 WORKER COMPLETE: buffer=%d frames=%d elapsed=%.3f sec",
                    buf, nz, time.time() - batch_start)
    except Exception as e:
        logger.error("V31 WORKER ERROR buffer=%d: %s: %s", buf, type(e).__name__, e)
        logger.error("V31 WORKER TRACEBACK:\n%s", traceback.format_exc())
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
        logger.info("V31 WORKER EXIT: buffer=%d rss_now=%.1f MiB peak=%.1f MiB",
                    buf, rss_now, rss_peak)


def _v23_malloc_trim():
    return _v22_malloc_trim()

def capture_asi(free_queue, ready_queue, z1base, t1base, z2base, t2base, nx, ny, nz, tend, device_id, live, conf_file):
    global logger
    logger = setup_logging(os.getcwd())
    capture_start = time.time()
    buffer_sequence = 0
    total_frames = 0
    reason = "Unknown"

    logger.info("V31 CAPTURE START: now=%.3f tend=%.3f remaining=%.3f sec",
                capture_start, tend, tend - capture_start)
    logger.info("STVID configured frame size: %dx%d, %d frames/buffer", nx, ny, nz)

    try:
        while time.time() < tend:
            buf = free_queue.get()
            buffer_sequence += 1
            logger.debug("V31 GOT FREE BUFFER %d; starting isolated SDK worker sequence=%d", buf, buffer_sequence)

            if buf == 1:
                zbase, tbase = z1base, t1base
            elif buf == 2:
                zbase, tbase = z2base, t2base
            else:
                raise RuntimeError("Invalid free buffer number: %s" % buf)

            worker = multiprocessing.Process(
                target=_v23_capture_one_buffer,
                name="svbony_buffer_%d" % buf,
                args=(buf, zbase, tbase, nx, ny, nz, conf_file, buffer_sequence)
            )
            worker.start()
            worker.join()

            logger.info("V31 WORKER JOIN: buffer=%d pid=%s exitcode=%s",
                        buf, worker.pid, worker.exitcode)
            if worker.exitcode != 0:
                raise RuntimeError("V31 SVBONY buffer worker failed: buffer=%d exitcode=%s" %
                                   (buf, worker.exitcode))

            total_frames += nz
            ready_queue.put(buf)
            logger.debug("V31 READY: buffer=%d total_buffers=%d total_frames=%d",
                         buf, buffer_sequence, total_frames)

        reason = "Session time expired"
    except KeyboardInterrupt:
        reason = "Keyboard interrupt"
    except Exception as e:
        reason = "Unhandled exception: %s: %s" % (type(e).__name__, e)
        logger.error("V31 CAPTURE UNHANDLED EXCEPTION: %s", reason)
        logger.error("V31 CAPTURE TRACEBACK:\n%s", traceback.format_exc())
    finally:
        rss_now, rss_peak = _v23_mem_mb()
        logger.info("V31 CAPTURE EXIT: reason=%s elapsed=%.3f sec buffers=%d frames=%d",
                    reason, time.time() - capture_start, buffer_sequence, total_frames)
        logger.info("V31 CAPTURE FINAL RSS: now=%.1f MiB peak=%.1f MiB", rss_now, rss_peak)
        logger.info("Capture: %s - Exiting", reason)
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
    
    # Force a restart
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
        # Start processing
        while True:
            # Check mount state
            restart = False
            with open(os.path.join(controlpath, "state.txt"), "r") as fp:
                line = fp.readline().rstrip()
                if line == "restart":
                    restart = True

            # Restart
            if restart:
                # Log state
                with open(os.path.join(controlpath, "state.txt"), "w") as fp:
                    fp.write("observing\n")

                # Get obsid
                trestart = time.gmtime()
                obsid = "%s_%d/%s" % (time.strftime("%Y%m%d", trestart), device_id, time.strftime("%H%M%S", trestart))
                filepath = os.path.join(path, obsid)
                logger.info("Storing files in %s" % filepath)

                # Create output directory
                if not os.path.exists(filepath):
                    try:
                        os.makedirs(filepath)
                    except PermissionError:
                        logger.error("Can not create output directory: %s" % filepath)
                        raise

                # Get mount position
                with open(os.path.join(controlpath, "position.txt"), "r") as fp:
                    line = fp.readline()
                with open(os.path.join(filepath, "position.txt"), "w") as fp:
                    fp.write(line)

            # Wait for a completed capture buffer.
            # The main process sends None after capture has really exited.
            # Therefore we do not use a timeout here: a timeout can make the
            # compressor exit while capture is still running slowly.
            proc_buffer = ready_queue.get()

            # None is the explicit end-of-capture marker.
            if proc_buffer is None:
                logger.debug("Received end-of-capture marker")
                break

            logger.debug("Processing buffer %d" % proc_buffer)

            # Log start time
            tstart = time.time()

            # Process first buffer
            if proc_buffer == 1:
                t = t1                
                z = z1
            elif proc_buffer == 2:
                t = t2
                z = z2

            # Format time
            nfd = "%s.%03d" % (time.strftime("%Y-%m-%dT%T",
                                             time.gmtime(t[0])), int((t[0] - np.floor(t[0])) * 1000))
            t0 = Time(nfd, format="isot")
            dt = t - t[0]

            # Detailed timing: separate statistics calculation from FITS I/O.
            # This lets us identify whether the increasing delay is caused by
            # NumPy statistics or by FITS file creation/writing.
            t_stats = time.time()

            # Input is uint8. Keep the accumulators integer while processing
            # frames; only convert the final 2-D results to float32.
            #
            # uint32 is sufficient for 100 frames:
            #   sum       <= 100 * 255       = 25,500
            #   sumsq     <= 100 * 255^2     = 6,502,500
            zmax = np.zeros((ny, nx), dtype=np.uint8)
            znum = np.zeros((ny, nx), dtype=np.int16)
            zsum = np.zeros((ny, nx), dtype=np.uint32)
            zsum2 = np.zeros((ny, nx), dtype=np.uint32)

            for j in range(nz):
                frame = z[:, :, j]

                # Maximum and frame number of the maximum.
                mask = frame > zmax
                zmax[mask] = frame[mask]
                znum[mask] = j

                # Integer accumulation. Cast before multiplication so the
                # square cannot overflow uint8.
                frame32 = frame.astype(np.uint32)
                zsum += frame32
                zsum2 += frame32 * frame32

            t_stats_done = time.time()

            # Exclude the maximum frame exactly as the original algorithm did.
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

            # Convert to FITS output types and flip vertically.
            zmax = np.flipud(zmax.astype("float32"))
            znum = np.flipud(znum.astype("float32"))
            zavg = np.flipud(zavg.astype("float32"))
            zstd = np.flipud(zstd.astype("float32"))

            # Generate fits
            t_fits = time.time()
            ftemp = "%s.temp" % nfd.replace(":", "-")
            fname = "%s.fits" % nfd.replace(":", "-")

            # Format header
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

            # Write fits file
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

            # The compressor is now completely finished with this buffer.
            # Only NOW may capture reuse it.
            free_queue.put(proc_buffer)
            logger.debug("Released buffer %d back to capture", proc_buffer)

            # V19: do NOT stop the compressor based on the last frame time.
            # The main process sends an explicit None marker only after the
            # capture child has really exited.  That marker is the sole
            # shutdown condition for this process.
            logger.debug("V25 processed buffer %d; waiting for next ready buffer", proc_buffer)
            

    except KeyboardInterrupt:
        logger.info("V25 COMPRESS: KeyboardInterrupt")
    except MemoryError as e:
        logger.error("V25 COMPRESS MEMORYERROR: %s", e)
        logger.error("V25 COMPRESS TRACEBACK:\n%s", traceback.format_exc())
    except Exception as e:
        logger.error(
            "V25 COMPRESS UNHANDLED EXCEPTION: %s: %s",
            type(e).__name__, e
        )
        logger.error("V25 COMPRESS TRACEBACK:\n%s", traceback.format_exc())
        raise
    finally:
        # Exiting
        logger.info("Exiting compress")


# Main function
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    
    # Read commandline options
    conf_parser = argparse.ArgumentParser(description="SVBONY V31 debug capture: isolated SDK worker per STVID buffer.")
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify configuration file(s). If no file" +
                             " is specified 'configuration.ini' is used.",
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

    # Process commandline options and parse configuration
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    
    conf_file = args.conf_file if args.conf_file else "configuration.ini"
    result = cfg.read(conf_file)

    if not result:
        print("Could not read config file: %s\nExiting..." % conf_file)
        sys.exit()

    # Setup logging
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] " +
                                     "[%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    # Generate directory
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

    # Testing mode
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

    # Live mode
    live = True if args.live else False
    logger.info("Live mode: %s" % live)
    if not testing:
        logger.info("SVBONY night mode: normal sunset/sunrise scheduling; FITS capture will run through the night")

    # Get camera type
    camera_type = cfg.get("Setup", "camera_type")

    # Get device id
    device_id = cfg.getint(camera_type, "device_id")

    # Current time
    tnow = Time.now()

    # Set location
    loc = EarthLocation(lat=cfg.getfloat("Observer", "latitude") * u.deg,
                        lon=cfg.getfloat("Observer", "longitude") * u.deg,
                        height=cfg.getfloat("Observer", "height") * u.m)

    if not testing:
        # Reference altitudes
        refalt_set  = cfg.getfloat("Setup", "alt_sunset") * u.deg
        refalt_rise = cfg.getfloat("Setup", "alt_sunrise") * u.deg

        # Aimpoint configuration
        if cfg.has_section("Aimpoint"):
            aimpoint_az = cfg.getfloat("Aimpoint", "az_deg") * u.deg
            aimpoint_alt = cfg.getfloat("Aimpoint", "alt_deg") * u.deg
            aimpoint_height = cfg.getfloat("Aimpoint", "height_km") * u.km
        else:
            aimpoint_az, aimpoint_alt, aimpoint_height = None, None, None

        # Get logic
        action, wait_time, tend, state = observe_logic(tnow, loc, refalt_set, refalt_rise,
                                                       aimpoint_az, aimpoint_alt, aimpoint_height)

        # Wait for observation start
        logger.info(state)
        if action == "wait":
            logger.info(f"Waiting for {wait_time:.0f} seconds.")
            try:
                time.sleep(wait_time)
            except KeyboardInterrupt:
                sys.exit()
    else:
        tend = tnow + test_duration * u.s

    # Read shutter config
    if cfg.has_section("Shutter"):
        from stvid.shutter import Shutter
        shutter = Shutter(cfg.getint("Shutter", "pin"))
    else:
        shutter = None
        
    logger.info("Starting data acquisition")
    logger.info("Acquisition will end after "+tend.isot)

    # Get settings
    nx = cfg.getint(camera_type, "nx")
    ny = cfg.getint(camera_type, "ny")
    nz = cfg.getint(camera_type, "nframes")

    # Initialize arrays
    z1base = multiprocessing.Array(ctypes.c_uint8, nx * ny * nz)
    t1base = multiprocessing.Array(ctypes.c_double, nz)
    z2base = multiprocessing.Array(ctypes.c_uint8, nx * ny * nz)
    t2base = multiprocessing.Array(ctypes.c_double, nz)

    # Real two-buffer handshake:
    #   free_queue  = buffers that capture is allowed to write
    #   ready_queue = buffers completely filled and ready for compression
    #
    # Buffer 1 is initially available.  Buffer 2 is also available, so
    # capture can fill one while compress processes the other.
    free_queue = multiprocessing.Queue(maxsize=2)
    ready_queue = multiprocessing.Queue(maxsize=1)
    free_queue.put(1)
    free_queue.put(2)

    # Set processes
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
                                                 device_id, live, conf_file))
    elif camera_type == "CV2":
        pcapture = multiprocessing.Process(target=capture_cv2,
                                           name="capture_cv2",
                                           args=(image_queue,
                                                 z1base, t1base, z2base, t2base,
                                                 nx, ny, nz, tend.unix,
                                                 device_id, live, conf_file))
    elif camera_type == "ASI":
        pcapture = multiprocessing.Process(target=capture_asi,
                                           name="capture_asi",
                                           args=(free_queue, ready_queue,
                                                 z1base, t1base, z2base, t2base,
                                                 nx, ny, nz, tend.unix,
                                                 device_id, live, conf_file))

    try:
        # Open shutter
        if shutter:
            shutter.open()
        
        # Start
        pcapture.start()
        pcompress.start()
        logger.info(
            "V31 MAIN: started capture pid=%s compressor pid=%s",
            pcapture.pid, pcompress.pid
        )

        # End
        # First wait until capture has really stopped.  Only then send an
        # explicit end-of-capture marker to compress.  This drains any final
        # ready buffer before compress exits and avoids the old 60-second
        # "Ready-buffer queue timed out" shutdown.
        try:
            pcapture.join()
            logger.info(
                "V31 MAIN: capture process joined; exitcode=%s",
                pcapture.exitcode
            )

            # None is never a valid buffer number; it is our shutdown marker.
            logger.debug("V31 MAIN: sending end-of-capture marker to compressor")
            ready_queue.put(None)
            logger.debug("V31 MAIN: end-of-capture marker sent")

            pcompress.join()
            logger.info(
                "V31 MAIN: compressor process joined; exitcode=%s",
                pcompress.exitcode
            )
        except (KeyboardInterrupt, ValueError):
            time.sleep(0.1) # Allow a little time for a graceful exit
        except MemoryError as e:
            logger.error("Memory error %s" % e)
        finally:
            if pcapture.is_alive():
                pcapture.terminate()
            if pcompress.is_alive():
                pcompress.terminate()

        # Release device
        if live is True:
            cv2.destroyAllWindows()
    finally:
        if shutter:
            shutter.close()
