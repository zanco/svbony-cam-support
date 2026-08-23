#!/usr/bin/env python3

"""
SVBONY camera detection and diagnostics.

This tool uses the SVBONY Python SDK to enumerate connected SVBONY
cameras and display their basic properties.

Requirements
------------
- Python 3
- SVBONY Python SDK (PySVBCameraSDK)
- A supported SVBONY camera connected by USB

The SDK is NOT included with this script. Install the SVBONY SDK
before running this tool.

The script does not start video capture and does not intentionally
change camera settings.

Usage
-----
    python3 detect_svbony.py
"""

import sys


def safe_get(obj, name, default="N/A"):
    """Read an attribute without allowing one missing property to abort detection."""
    try:
        return getattr(obj, name)
    except Exception:
        return default


def main():
    print("=" * 70)
    print("SVBONY CAMERA DETECTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # SDK availability check
    # ------------------------------------------------------------------
    try:
        from pysvb.camera import PySVBCameraSDK
    except Exception as exc:
        print()
        print("ERROR: SVBONY Python SDK is not available.")
        print()
        print("This tool requires the SVBONY Python SDK (PySVBCameraSDK).")
        print("Install the SVBONY SDK before running this script.")
        print()
        print("Python import error:")
        print(f"  {exc}")
        print()
        return 2

    # ------------------------------------------------------------------
    # SDK initialisation check
    # ------------------------------------------------------------------
    try:
        sdk = PySVBCameraSDK()
    except Exception as exc:
        print()
        print("ERROR: The SVBONY Python SDK could not be initialised.")
        print()
        print("The SDK module was found, but initialisation failed.")
        print("Check that the SVBONY SDK installation is complete.")
        print()
        print(f"SDK error:")
        print(f"  {exc}")
        print()
        return 3

    print()
    print("SDK version:")
    print(f"  {safe_get(sdk, 'sdk_version')}")
    print()

    # ------------------------------------------------------------------
    # Camera enumeration
    # ------------------------------------------------------------------
    try:
        count = int(sdk.get_num_of_connected_cameras())
    except Exception as exc:
        print("ERROR: Could not enumerate SVBONY cameras.")
        print()
        print(f"SDK error:")
        print(f"  {exc}")
        print()
        return 4

    print(f"Connected SVBONY cameras: {count}")
    print()

    if count == 0:
        print("No SVBONY cameras detected.")
        print()
        return 0

    cameras = []

    for index in range(count):
        print("-" * 70)
        print(f"CAMERA INDEX {index}")
        print("-" * 70)

        try:
            info = sdk.get_camera_info(index)
        except Exception as exc:
            print(f"ERROR reading camera information for index {index}:")
            print(f"  {exc}")
            print()
            continue

        camera_id = safe_get(info, "CameraID")
        friendly_name = safe_get(info, "FriendlyName")
        serial = safe_get(info, "CameraSN")

        print(f"SDK index       : {index}")
        print(f"Camera ID       : {camera_id}")
        print(f"Friendly name   : {friendly_name}")
        print(f"Serial number   : {serial}")

        opened = False

        # Temporarily open the camera so its properties can be queried.
        # No video capture is started.
        try:
            sdk.open_camera(camera_id)
            opened = True
        except Exception as exc:
            print()
            print("Could not open camera for property inspection:")
            print(f"  {exc}")

        if opened:
            try:
                props = sdk.get_camera_property(camera_id)

                print()
                print("Camera properties:")
                print(
                    f"  Maximum resolution : "
                    f"{safe_get(props, 'MaxWidth')} x "
                    f"{safe_get(props, 'MaxHeight')}"
                )
                print(f"  Maximum bit depth  : {safe_get(props, 'MaxBitDepth')}")
                print(f"  Color camera flag  : {safe_get(props, 'IsColorCam')}")

                prop_name = safe_get(props, "FriendlyName", None)
                if prop_name not in (None, "N/A"):
                    print(f"  Property name      : {prop_name}")

                try:
                    mode = sdk.get_camera_mode(camera_id)
                    print(f"  Camera mode        : {mode}")
                except Exception:
                    print("  Camera mode        : unavailable")

            except Exception as exc:
                print()
                print("Could not read camera properties:")
                print(f"  {exc}")

            finally:
                try:
                    sdk.close_camera(camera_id)
                except Exception:
                    pass

        cameras.append(
            {
                "index": index,
                "camera_id": camera_id,
                "friendly_name": friendly_name,
                "serial": serial,
            }
        )

        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)

    for camera in cameras:
        print(
            f"index={camera['index']}  "
            f"id={camera['camera_id']}  "
            f"name={camera['friendly_name']}  "
            f"serial={camera['serial']}"
        )

    print()
    print("No video capture was started.")
    print("No camera settings were intentionally changed.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
