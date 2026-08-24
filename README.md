# svbony-cam-support
Trying to get the svbony camera SV305M working in acquire / stvid 

Software has been created with help of ChatGPT

Software emulates the SV305M camera as ASI camera for now but gives great results. Average clear nights > 150 observations.

Added detect_svbony.py to detect which SVB camera's are attached and show their name, camera properties and serial number. 
to do is to add the camera serial number to configuration.ini so if multiple camera's are connected the right one is selected for acquire

(the other one is ment to be used on indi allsky on the same computer)

## Requirements

This tool requires the SVBONY Camera SDK.

Tested with **SVBONY Camera SDK v1.10.2**.

Download the Linux SDK from the official SVBONY website:

https://www.svbony.com/downloads/software-driver

The SDK is listed under **SDK → Linux**. :contentReference[oaicite:0]{index=0}

Newer SDK versions may also work, but have not been tested with this tool.

output shows on https://github.com/zanco/svbony-cam-support/blob/main/resultaten/detect_svbony_results.png

Last nights run (2026-08-23 ) with the SVB Backend version showed no memory problems and resulted in 23 classfd and 255 regular satellites in observation. 
