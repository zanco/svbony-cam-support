# SVBONY SV305M PRO support for STVID — V31 background and development notes

## Overview

This repository contains **V31** of the experimental STVID acquisition program for the **SVBONY SV305M PRO**, using the SVBONY Python SDK wrapper (PySVB) as the camera access path.

V31 is not intended to be a generic replacement for the normal ZWO/ASI camera path. It documents and implements a practical workaround for several problems encountered while trying to make the SV305M PRO behave as a reliable STVID source:

- the SVBONY Python wrapper returns frames in an RGBA representation;
- the SV305M PRO is a monochrome camera even though the SDK reports `IsColorCam = True`;
- a returned frame is much larger than the actual monochrome image because of the RGBA packaging;
- the PySVB/SVBONY wrapper appears to retain substantial temporary memory for every returned frame;
- keeping one camera/SDK worker alive for a long STVID session caused process memory to grow dramatically;
- the original two-buffer approach therefore needed to be reconsidered;
- V31 uses a short-lived worker process for each STVID 100-frame buffer;
- V31 reads exposure and gain from the selected STVID configuration file instead of hard-coding them in the Python source.

The program is the result of an iterative debugging process rather than a clean-room implementation. The version history and comments in the source are therefore deliberately retained: they explain why apparently unusual decisions were made.

---

# 1. The camera and the first surprise: "IsColorCam = True"

When the SV305M PRO is opened through the SVBONY SDK, the SDK reports:

```text
SVBONY SDK IsColorCam flag: True
```

That is potentially misleading for STVID.

The camera is being treated here as a **monochrome sensor**. The fact that the SDK reports the camera as a color camera does not mean that the data delivered to this STVID bridge should be interpreted as RGB astronomical imagery.

In the observed PySVB output, the returned pixels have the form:

```text
R G B A
```

but the three image channels are identical:

```text
R = G = B
```

while alpha is:

```text
A = 255
```

For example, V31 deliberately checks the first frames and observes identical R, G and B values.

This is **RGBA packaging of a monochrome image**, not evidence that the SV305M PRO is producing three independent colour channels.

V31 therefore copies only the R channel into the reusable STVID frame:

```python
self.output_frame[:, :] = rgba[:, :, 0]
```

This is intentional.

---

# 2. Why is a 1920 × 1080 frame 8,294,400 bytes?

The configured camera frame is:

```text
1920 × 1080 pixels
```

That is:

```text
2,073,600 pixels
```

The PySVB data observed for this camera is RGBA8, meaning four 8-bit values per pixel:

```text
R + G + B + A = 4 bytes/pixel
```

Therefore:

```text
1920 × 1080 × 4
= 8,294,400 bytes
```

V31 verifies this size explicitly.

The log reports:

```text
SVBONY V6 data format: RGBA8, bytes/pixel: 4
SVBONY real frame size: 8294400
SVBONY SDK buffer size: 8294400
```

The actual astronomical image information is monochrome, but the SDK delivery format is four bytes per pixel.

The bridge therefore does **not** retain four channels in the STVID shared buffer. It extracts one channel and discards the temporary RGBA representation as soon as possible.

---

# 3. Why the FITS files are much larger than one raw frame

An STVID buffer contains 100 frames.

At 1920 × 1080, one monochrome 8-bit frame contains:

```text
2,073,600 bytes
```

A 100-frame buffer therefore represents approximately:

```text
207,360,000 bytes
```

of 8-bit image samples before STVID statistics are calculated.

The FITS file is not a single raw camera frame.

For each 100-frame buffer, the compressor calculates four statistical products:

- average image;
- standard deviation image;
- maximum image;
- number/count image.

The FITS primary HDU contains:

```python
np.array([zavg, zstd, zmax, znum])
```

and these arrays are stored as `float32`.

So the final FITS data volume is approximately:

```text
4 statistical planes
× 1920 × 1080 pixels
× 4 bytes/pixel
= 33,177,600 bytes
```

before FITS overhead.

Therefore FITS files around **32 MB** are entirely plausible.

This is an important distinction:

> The approximately 8.3 MB number describes one RGBA8 frame returned by PySVB. The approximately 32 MB FITS file describes four float32 statistical images produced from a 100-frame STVID buffer.

These are two different stages of the pipeline.

---

# 4. The original buffer problem

The most difficult problem was not initially the FITS writer. It was the interaction between:

1. PySVB;
2. the SVBONY SDK;
3. large RGBA frame allocations;
4. the long-running Python worker;
5. STVID's two-buffer acquisition/compression model.

The PySVB path returns a large temporary RGBA object for every frame.

At 1920 × 1080:

```text
8,294,400 bytes per returned RGBA frame
```

During testing it became clear that deleting the Python/NumPy objects and periodically calling:

```python
gc.collect()
malloc_trim()
```

did not reliably return the process RSS to its original level.

The V31 design therefore uses the operating system's process boundary as the reliable reclamation mechanism.

---

# 5. Why malloc_trim() alone was not enough

V31 still performs garbage collection and `malloc_trim()` periodically.

That is useful housekeeping, but it is not the fundamental solution.

The critical discovery was that the operating system can reclaim the complete address space when the worker process exits.

That led to the central V31 design decision:

> **One SDK/PySVB worker process captures exactly one 100-frame STVID buffer and then exits.**

The next 100-frame buffer is handled by a new worker process.

This gives the operating system a clean boundary at which all memory associated with that PySVB session can be reclaimed.

---

# 6. The short-lived worker design

The worker is deliberately designed to capture exactly one STVID buffer in a short-lived process.

Conceptually:

```text
open camera
    ↓
configure camera
    ↓
start video capture
    ↓
capture 100 frames
    ↓
copy monochrome data into STVID shared buffer
    ↓
stop video capture
    ↓
close camera
    ↓
exit worker process
```

The parent capture process waits for the worker to finish before considering that buffer ready.

The compressor can process the completed buffer while the capture side starts another isolated worker for the next buffer.

The short-lived worker is not an accidental implementation detail. It is the principal workaround for the observed PySVB memory-retention behaviour.

---

# 7. Memory behaviour observed during V31 testing

A V31 test at:

```text
1920 × 1080
100 frames/buffer
RGBA8
100 ms exposure
gain 300
```

showed substantial RSS growth during a single 100-frame buffer.

Representative measurements were approximately:

```text
frame 10   ~422 MiB
frame 20   ~501 MiB
frame 30   ~580 MiB
frame 40   ~659 MiB
frame 50   ~738 MiB
frame 60   ~817 MiB
frame 70   ~896 MiB
frame 80   ~976 MiB
frame 90  ~1055 MiB
frame 100 ~1134 MiB
```

The worker then completed the buffer and exited.

This is why V31 does not simply keep one PySVB worker alive indefinitely.

These values describe the observed Python/SVBONY acquisition process, not the physical memory usage of the camera.

---

# 8. Why the two-buffer model is still useful

STVID already has a producer/consumer concept:

```text
capture → buffer → compressor
```

V31 retains that concept.

The important change is that the SVBONY SDK interaction is isolated inside a worker process.

Conceptually:

```text
                         ┌──────────────────┐
                         │  STVID buffer 1  │
                         └────────┬─────────┘
                                  │
Camera → PySVB worker 1 ──────────┤
                                  ↓
                              compressor
                                  ↑
Camera → PySVB worker 2 ──────────┤
                                  │
                         ┌────────┴─────────┐
                         │  STVID buffer 2  │
                         └──────────────────┘
```

The implementation uses shared memory for the STVID buffers and separate worker processes for the camera/SDK side.

The compressor is not stopped merely because a frame timestamp becomes old.

Instead, the main process waits for the capture process to really finish and then sends an explicit end-of-capture marker to the compressor.

This avoids the older shutdown problem involving a long:

```text
"Ready-buffer queue timed out"
```

condition.

The explicit `None` marker is the compressor's shutdown condition.

---

# 9. Camera lifecycle: PySVB is deliberately the sole owner

Another major part of the debugging process concerned the camera lifecycle.

Earlier experiments used more than one route into the SVBONY SDK/native library. This produced unreliable behaviour, including camera status problems when a second/native capture path was involved.

V31 deliberately uses:

```text
PySVB
```

as the sole owner of:

- opening the camera;
- starting video capture;
- obtaining video frames;
- stopping video capture;
- closing the camera.

The native library may still be loaded where necessary for the SDK environment, but the actual camera lifecycle and `get_video_data()` path are handled by PySVB.

This separation is intentional.

---

# 10. Exposure and gain are configuration values, not source-code constants

A practical lesson from the experiments was that exposure and gain should not be hard-coded into every experimental Python version.

V31 reads the camera controls from the same configuration file used by STVID.

The relevant logic is:

```python
camera_type = cfg.get("Setup", "camera_type")
configured_exposure_us = cfg.getint(camera_type, "exposure")
configured_gain = cfg.getint(camera_type, "gain")
```

The values are then passed to the SVBONY bridge.

A V31 test configuration containing:

```text
exposure = 100000
gain = 300
```

produced log output confirming:

```text
V31 CONFIG: ASI exposure=100000 us gain=300
```

and:

```text
PySVB setup exposure = 100000 us, gain = 300
```

This was verified in an actual V31 test run.

This means the same Python program can be tested with different exposure/gain combinations by changing the configuration file rather than editing the acquisition program itself.

---

# 11. The `-c` option

V31 retains the STVID-style configuration-file selection:

```bash
python3 acquire_SVBONY_two_buffer_handshake_timed_v31_config_controls.py -c configuration.ini
```

If `-c` is omitted, the program uses:

```text
configuration.ini
```

as the default.

The option is implemented with `-c` / `--conf_file`.

This is useful for experimentation because multiple configuration files can be kept and selected without modifying the Python program.

For example:

```bash
python3 acquire_SVBONY_two_buffer_handshake_timed_v31_config_controls.py -c configuration_gain10.ini
```

or:

```bash
python3 acquire_SVBONY_two_buffer_handshake_timed_v31_config_controls.py -c configuration_gain300.ini
```

provided those files contain the appropriate STVID sections.

---

# 12. STVID frame size and ROI

V31 does not blindly force the camera to stream its maximum sensor size.

The requested dimensions come from the STVID configuration.

For the SV305M PRO, the maximum observed sensor dimensions are:

```text
1920 × 1080
```

V31 requests the configured STVID dimensions from the SVBONY SDK and verifies the actual ROI returned by the camera.

If the camera does not accept the requested ROI, V31 raises an error rather than silently continuing with a different frame size.

This matters because unnecessarily forcing 1920 × 1080 would increase:

- USB traffic;
- SDK memory traffic;
- temporary RGBA allocation;
- conversion work;
- STVID processing load.

---

# 13. RGBA conversion to the STVID monochrome buffer

The returned PySVB object is interpreted as:

```text
height × width × 4
```

with:

```text
R G B A
```

per pixel.

V31 verifies that the returned object contains at least:

```text
width × height × 4
```

bytes.

It then reshapes the returned data into the RGBA image.

For the SV305M PRO, the R, G and B channels were observed to be identical.

V31 therefore copies only:

```python
rgba[:, :, 0]
```

into the STVID output frame.

Immediately afterwards it deletes the temporary objects:

```python
del rgba
del raw
del z
```

and periodically performs garbage collection and `malloc_trim()`.

The process boundary remains the important memory-reclamation mechanism.

---

# 14. Why the first frames are inspected

The first few frames are deliberately inspected in V31.

The program reports:

- returned frame length;
- expected frame length;
- first bytes in hexadecimal;
- minimum pixel value;
- maximum pixel value;
- mean pixel value;
- alpha range;
- maximum R-G difference;
- maximum R-B difference.

This was added because one of the most confusing failures during development was not a simple camera connection failure.

The SDK could successfully:

- find the camera;
- open the camera;
- report the expected resolution;
- start video capture;

while still returning frames that were effectively all zero.

An earlier failing run repeatedly showed:

```text
00 00 00 ff
00 00 00 ff
...
```

with:

```text
R[0,0,0.000]
G[0,0,0.000]
B[0,0,0.000]
A[255,255]
```

That is a structurally valid RGBA packet but contains no useful image signal.

V31 therefore makes the first frames visible in the log instead of allowing a structurally valid but black frame to go unnoticed.

---

# 15. Why a "black frame" is not the same as a camera failure

There are several different failure modes:

### Camera not found

The SDK reports no connected cameras.

### Camera cannot be opened

The SDK lifecycle fails.

### Video capture does not start

The SDK start call fails.

### Frame has the wrong size

The SDK returns data, but it does not match the configured ROI and expected RGBA representation.

### Frame is structurally valid but black

The SDK returns the expected number of bytes and a valid RGBA structure, but the image samples are all zero.

The last case was especially important.

A test that only checks:

```text
returned_len == expected_len
```

would incorrectly conclude that everything is working.

V31 checks actual pixel content as well.

---

# 16. FITS display scaling is not the same as the underlying data

During development, FITS files were inspected with SAOImage DS9.

A FITS image can look very different depending on:

- linear scaling;
- logarithmic scaling;
- automatic minimum/maximum values;
- manually selected display limits;
- which FITS plane is being displayed.

Therefore an image that visually appears black or white is not by itself sufficient to diagnose the camera.

The acquisition program logs numerical properties of the raw frames.

The FITS writer also stores several statistical products that can be examined numerically.

This makes it possible to compare what a viewer displays with the actual image values.

---

# 17. FITS structure produced by V31

For every completed 100-frame STVID buffer, V31 creates a FITS primary HDU containing four planes:

```text
[zavg, zstd, zmax, znum]
```

The header includes, among other things:

```text
DATE-OBS
MJD-OBS
EXPTIME
NFRAMES
CRPIX1
CRPIX2
CRVAL1
CRVAL2
...
```

and one timestamp keyword for every frame:

```text
DT0000
DT0001
...
```

The FITS data are written as `float32` arrays.

This is why the output file size is much larger than the original 8-bit monochrome image.

---

# 18. Exposure time and FITS `EXPTIME`

One subtle point is worth documenting.

The FITS header contains:

```text
EXPTIME = dt[-1] - dt[0]
```

This is the elapsed time represented by the first and last frame timestamps in the STVID buffer.

It is therefore not simply the camera exposure setting multiplied by the number of frames.

For example, with a nominal 100 ms camera exposure and 100 frames, the total elapsed buffer time may differ from exactly:

```text
100 × 0.100 s
```

because the timestamps represent the actual acquisition timing.

The camera exposure setting and the FITS `EXPTIME` field therefore describe different things.

---

# 19. What V31 demonstrated in the controlled test

A V31 test at 100 ms exposure and gain 300 demonstrated:

```text
SVBONY SV305M PRO detected
        ↓
PySVB camera opened
        ↓
1920 × 1080 ROI accepted
        ↓
100 ms exposure / gain 300 read from configuration
        ↓
PySVB video capture started
        ↓
100-frame buffer captured
        ↓
worker exited
        ↓
next isolated worker captured another 100 frames
        ↓
third isolated worker captured another 100 frames
        ↓
compressor created FITS output
```

The test produced three 100-frame buffers / FITS files during a 30-second test.

The raw frame diagnostics showed real non-zero pixel values in the first frames, while later frames at that exposure/gain combination could reach 255 and therefore saturate.

The worker memory measurements also confirmed why the isolated-process architecture is important.

---

# 20. What V31 does not claim

V31 should be considered an experimental SVBONY/STVID bridge.

It does **not** claim that:

- the SV305M PRO is natively supported by STVID;
- every SVBONY camera will behave the same way;
- the SVBONY SDK's `IsColorCam` flag should be trusted for every camera;
- the RGBA representation is the physical sensor format;
- the memory behaviour is guaranteed to be identical on every Linux/Python/SDK combination;
- the tested exposure/gain combination is suitable for every observing condition;
- a FITS image that looks good in DS9 is necessarily numerically correct for every STVID use case.

The code is deliberately instrumented because the project is still experimental.

---

# 21. Why this repository contains V31 rather than all previous versions

The earlier versions were useful during development, but they represent intermediate experiments involving different combinations of:

- native SVBONY SDK access;
- PySVB access;
- camera lifecycle ownership;
- buffer handling;
- black-frame detection;
- memory management;
- configuration handling.

Keeping every intermediate version in the main repository would make it difficult for another STVID user to determine which implementation is the current experimental reference.

V31 is therefore presented as the current consolidated experimental version.

The comments referring to V22, V25, V29 and V30 remain in the source because they document important decisions and failures that led to V31.

---

# 22. Recommended testing commands

For normal night operation, V31 is started without `-t`:

```bash
nohup python3 acquire_SVBONY_two_buffer_handshake_timed_v31_config_controls.py > v31_nacht.log 2>&1 &
```

For a short immediate test:

```bash
nohup python3 acquire_SVBONY_two_buffer_handshake_timed_v31_config_controls.py -t 30 > v31_test.log 2>&1 &
```

To use a different configuration file:

```bash
nohup python3 acquire_SVBONY_two_buffer_handshake_timed_v31_config_controls.py -c another_configuration.ini -t 30 > v31_test.log 2>&1 &
```

The configuration file should contain the normal STVID `[Setup]` section and the camera-specific section selected by `camera_type`.

The acquisition log should be checked for:

```text
V31 CONFIG: ...
```

and:

```text
PySVB setup exposure = ...
gain = ...
```

before interpreting a test result.

---

# 23. A practical warning about memory

The V31 architecture deliberately creates and destroys a worker for each 100-frame buffer.

This should not be "optimized away" simply because repeatedly opening and closing the camera looks inefficient.

The process boundary is part of the workaround.

In testing, a single worker capturing 100 frames at 1920 × 1080 showed RSS growth from a few hundred MiB to more than 1.1 GiB.

The worker then exited cleanly.

The operating system is very good at reclaiming an entire process address space. It is much less predictable to rely on Python garbage collection and the C allocator to return every large temporary allocation immediately.

Therefore:

> **The worker lifetime is intentional.**

Any future attempt to make the camera worker persistent should be treated as a new experiment, not as a harmless optimization.

---

# 24. Development history in one paragraph

The path to V31 was essentially:

```text
SVBONY camera connection works
        ↓
frames arrive, but format is surprising
        ↓
RGBA8 discovered
        ↓
R=G=B discovered
        ↓
SV305M PRO treated as monochrome
        ↓
black-frame behaviour discovered
        ↓
multiple camera/SDK lifecycle paths proved unreliable
        ↓
PySVB made sole camera owner
        ↓
large per-frame memory retention discovered
        ↓
malloc_trim / garbage collection investigated
        ↓
persistent worker considered unsuitable
        ↓
one worker per 100-frame STVID buffer
        ↓
explicit buffer completion / compressor shutdown
        ↓
configuration-driven exposure and gain
        ↓
V31
```

This is the context that is otherwise difficult to understand from the final source code alone.

---

# 25. Why this may be useful to other STVID users

Most STVID installations appear to use ZWO/ASI cameras through the established ASI path.

The SV305M PRO is different enough that simply replacing the camera name is not sufficient.

The useful lessons are broader than this particular camera:

1. A camera SDK can report a colour capability while the useful astronomical data are effectively monochrome.
2. An SDK wrapper can package a single-channel sensor as RGBA, multiplying the incoming data volume by four.
3. A frame-size check alone does not prove that useful image data are present.
4. Python/C-library memory behaviour can dominate the design of a long-running acquisition process.
5. Process isolation can sometimes be a more reliable memory-reclamation mechanism than increasingly aggressive garbage collection.
6. Configuration-driven camera controls are essential for reproducible experiments.
7. FITS file size must be interpreted in terms of the statistical products stored in the file, not just the size of one incoming camera frame.
8. FITS viewer display scaling should not be confused with the underlying numerical image data.

---

## Status

**V31 is experimental.**

It has successfully demonstrated acquisition from an SVBONY SV305M PRO through PySVB, conversion of the returned RGBA representation to a monochrome STVID frame, isolated 100-frame worker operation, configuration-driven exposure/gain, and FITS generation.

Further long-duration night testing is required before considering the implementation production-ready.

The purpose of publishing V31 is therefore not to claim that the problem is completely solved, but to make the current working point, the failures encountered, and the reasoning behind the design available to other STVID users who may want to reproduce, test, improve, or challenge the approach.
