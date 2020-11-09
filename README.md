# ITMO-mobileCV
This repository contains all laboratory works for the [ITMO University mobile systems for computer vision course](https://github.com/zeanfa/mobileCV).

## Lab 1 (passed)
Captures a camera image and displays it with the area of interest in the center (60x60). 
When the image in that area is completely red, green or blue, its border is changed to the corresponding color.
This program uses HSV color model.

Usage: `python var1.py [-h <int>|--hue=<int>] [-s <int,int>|--saturation=<int,int>] [-v <int,int>|--value=<int,int>] [--fps=<int>]`

**Params:**<br>
***-h*** or ***--hue*** - Sensitivity of Hue param in HSV model **(optional)**<br>
***-s*** or ***--saturation*** - Min and max values of Saturation **(optional)**<br>
***-v*** or ***--value*** - Min and max values of third param in HSV model **(optional)**<br>
***--fps*** - Frames per second to capture **(optional)**

**Defaults:**<br>
Hue sensitivity: 20<br>
Saturation: 70..255<br>
Value: 50..255

## Lab 2 (TODO)
## Lab 3 (TODO)
