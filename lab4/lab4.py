# ITMO University
# Mobile Computer Vision course
# 2021
# by seniorkot & atepaevm

import getopt
import sys
import os
import time

import cv2
import pickle
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


def get_encodings(data_path: str) -> dict:
    encoding_path = os.path.join(data_path, 'encodings.data')
    encodings = {}

    if os.path.isfile(encoding_path):
        with open(encoding_path, 'rb') as f:
            encodings = pickle.load(f)
    else:
        for img in os.listdir(data_path):
            image = cv2.imread(f'{data_path}/{img}', )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(image)[0]
            encodings[os.path.splitext(img)[0]] = encode
        with open(encoding_path, 'wb') as f:
            pickle.dump(encodings, f)
    return encodings


def print_usage(exit_code: int):
    print('Usage: python lab4.py [-d <dir>|--data=<dir>] '
          '[-v <path>|--video=<path>]')
    sys.exit(exit_code)


def main(argv: list,
         data_path: str = './data',
         video=None):
    try:
        opts, args = getopt.getopt(argv, "d:v:", ["data=", "video=", "help"])
    except getopt.GetoptError:
        print_usage(1)

    for opt, arg in opts:
        if opt in ['-d', '--data']:
            if not os.path.isdir(arg):
                print('Data directory doesn\'t exist')
                sys.exit(2)
            data_path = arg
        elif opt in ['-v', '--video']:
            if not os.path.isfile(arg):
                print('Video must be an existing file')
                sys.exit(2)
            video = arg
        elif opt in ['--help']:
            print_usage(0)
        else:
            print_usage(1)

    # Encode faces
    timest = time.time()
    encodings = get_encodings(data_path)
    print(f'Completed encoding of {len(encodings)} faces in '
          f'{time.time() - timest} sec')
    # TODO: Show camera or video, locate faces, etc.


if __name__ == "__main__":
    main(sys.argv[1:])
