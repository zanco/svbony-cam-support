# svbony-cam-support
Trying to get the svbony camera SV305M working in acquire / stvid 

Software has been created with help of ChatGPT

Software emulates the SV305M camera as ASI camera for now but gives great results. Average clear nights > 150 observations.

Added detect_svbony.py to detect which SVB camera's are attached and show their name, camera properties and serial number. 
to do is to add the camera serial number to configuration.ini so if multiple camera's are connected the right one is selected for acquire

(the other one is ment to be used on indi allsky on the same computer)

SVBONY SDK required
This tool requires the SVBONY Camera SDK. Download the Linux SDK from the official SVBONY Software & Driver Downloads page before running detect_svbony.py. 

