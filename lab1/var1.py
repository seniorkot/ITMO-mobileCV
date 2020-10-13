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


def show_camera(h_sensitivity: int = 20,
                s_lower: int = 70,
                s_higher: int = 255,
                v_lower: int = 50,
                v_higher: int = 255):
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

            # Step 1 - Get coords of the area of interest
            height, width, _ = frame.shape
            # 60x60 square in the middle of frame
            upper_left = (width // 2 - 30, height // 2 - 30)
            bottom_right = (width // 2 + 29, height // 2 + 29)

            # Step 2 - Detecting colors in the area of interest
            # Converting from BGR to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create mask for red color
            # Range for lower red
            lower_red = np.array([0, s_lower, v_lower])
            upper_red = np.array([h_sensitivity, s_higher, v_higher])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            # Range for upper red
            lower_red = np.array([180 - h_sensitivity, s_lower, v_lower])
            upper_red = np.array([180, s_higher, v_higher])
            mask_red = mask1 + cv2.inRange(hsv, lower_red, upper_red)
            # Crop mask with the area of interest
            mask_red = mask_red[upper_left[1]:bottom_right[1] + 1,
                                upper_left[0]:bottom_right[0] + 1]
            if 0 not in mask_red:
                pass
                # TODO: do an action

            # Create mask for green color
            lower_green = np.array([60 - h_sensitivity, s_lower, v_lower])
            upper_green = np.array([60 + h_sensitivity, s_higher, v_higher])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            # Crop mask with the area of interest
            mask_green = mask_green[upper_left[1]:bottom_right[1] + 1,
                                    upper_left[0]:bottom_right[0] + 1]
            if 0 not in mask_green:
                pass
                # TODO: do an action

            # Create mask for blue color
            lower_blue = np.array([120 - h_sensitivity, s_lower, v_lower])
            upper_blue = np.array([120 + h_sensitivity, s_higher, v_higher])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            # Crop mask with the area of interest
            mask_blue = mask_blue[upper_left[1]:bottom_right[1] + 1,
                                  upper_left[0]:bottom_right[0] + 1]
            if 0 not in mask_blue:
                pass
                # TODO: do an action

            # Draw green rectangle (square) and show the frame
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
