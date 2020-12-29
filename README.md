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

## Lab 2 (passed)
2 implementations of mean filter with hardcoded 3x3 area for each pixel. `void mean_filter(Mat, Mat, int, int)` function uses vectors to proceed image filtering. `void mean_filter_neon(uint8_t*, uint8_t*, int, int)` takes an advantage of parallel computing using Arm NEON command system.

Compile: `make`<br>
Usage: `./var1 <imagepath>`<br>
Output: <i>mean.png</i> and <i>neon.png</i> as the result of each function

**Testing:**<br>
This code was tested on Jetson Nano platform. The results are listed below in the table.
| Code<br>optimization | Image<br>resolution |	Mean, us	| Mean NEON, us |
|:----:|:----:|:----:|:----:|
| 0 |	SD | 186677 |	31728 |
| 1 |	SD | 20457 | 5935 |
| 2 |	SD | 19770 | 5429 |
| 3	| SD | 10169 | 4776 |
| 0 | HD | 211443 | 35402 |
| 1 | HD | 21619 | 6256 |
| 2	| HD | 22037 | 6254 |
| 3 | HD | 11333 | 5433 |
| 0 | Full HD | 1258991 | 209369 |
| 1 | Full HD | 130749 | 38703 |
| 2 | Full HD | 126249 | 37714 |
| 3 | Full HD | 65100 | 33504 |
| 0 | Ultra HD | 5007538 | 854234 |
| 1 | Ultra HD | 528413 | 172613 |
| 2 | Ultra HD | 508814 | 168508 |
| 3 | Ultra HD | 258277 | 152980 |

## Lab 3 (TODO)
