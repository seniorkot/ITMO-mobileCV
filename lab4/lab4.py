# ITMO University
# Mobile Computer Vision course
# 2021
# by seniorkot & atepaevm

import getopt
import sys
import time

import cv2
import face_recognition
import numpy as np


def gstreamer_pipeline(capture_width=1280,
                       capture_height=720,
                       display_width=1280,
                       display_height=720,
                       framerate=30,
                       flip_method=0):
    """ Returns a correct GStreamer pipeline for NVIDIA CSI camera """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def print_usage(exit_code: int):
    print('Usage: python lab4.py [-d <dir>|--data=<dir>]'
          '[-v <path>|--video=<path>]')
    sys.exit(exit_code)


def main(argv: list):
    try:
        opts, args = getopt.getopt(argv, "d:v:", ["data=", "video=", "help"])
    except getopt.GetoptError:
        print_usage(1)

    for opt, arg in opts:
        if opt in ['-d', '--data']:
            pass  # TODO: Set data path
        elif opt in ['-v', '--video']:
            pass  # TODO: Set video path
        elif opt in ['--help']:
            print_usage(0)
        else:
            print_usage(1)
    # TODO: Get face encodings first
    # TODO: Show camera or video, locate faces, etc.


if __name__ == "__main__":
    main(sys.argv[1:])
