# ITMO University
# Mobile Computer Vision course
# 2020
# by seniorkot & atepaevm

import cv2
import numpy as np


def gstreamer_pipeline(capture_width=1280,
                       capture_height=720,
                       display_width=1280,
                       display_height=720,
                       framerate=30,
                       flip_method=0,):
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


def show_camera():
    # print(gstreamer_pipeline(flip_method=4))
    # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4),
    #                        cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        # Create a window
        _ = cv2.namedWindow("MCV Lab_1 Var_1", cv2.WINDOW_AUTOSIZE)

        # Check fullscreen window property to determine if it's still available
        while cv2.getWindowProperty("MCV Lab_1 Var_1", 0) >= 0:
            ret, frame = cap.read()

            # step 1 - Get coords of the area of interest
            height, width, _ = frame.shape
            # 30x30 square in the middle of frame
            upper_left = (width // 2 - 30, height // 2 + 30)
            bottom_right = (width // 2 + 30, height // 2 - 30)

            # step 2 - TODO: Write this step

            # Draw green rectangle (square)
            cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0),
                          thickness=1)
            cv2.imshow('frame', frame)

            # Stop the program on the ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
