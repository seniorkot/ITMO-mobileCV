# ITMO University
# Mobile Computer Vision course
# 2020
# by seniorkot & atepaevm

import getopt
import sys
import time

import cv2
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


def show_frame(frame, start_time,
               upper_left: tuple,
               bottom_right: tuple,
               color: tuple = (0, 255, 255)) -> None:
    # Draw a square of the area of interest
    cv2.rectangle(frame, upper_left, bottom_right, color, thickness=2)
    # Print processing time
    cv2.putText(frame, "Time: {:.4f}sec".format(time.time() - start_time),
                org=(10, 50), color=(255, 0, 0), thickness=2,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
    # Show the frame
    cv2.imshow('frame', frame)


def show_camera(h_sensitivity: int,
                s_lower: int,
                s_higher: int,
                v_lower: int,
                v_higher: int,
                fps: int = 30):
    # print(gstreamer_pipeline(flip_method=4, framerate=fps))
    # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4, framerate=fps),
    #                        cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        # Create a window
        _ = cv2.namedWindow("MCV Lab_1 Var_1", cv2.WINDOW_AUTOSIZE)

        # Check fullscreen window property to determine if it's still available
        while cv2.getWindowProperty("MCV Lab_1 Var_1", 0) >= 0:
            ret, frame = cap.read()

            # Check return value (in case no frames available)
            if not ret:
                break

            start_time = time.time()

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

            # Create mask for green color
            lower_green = np.array([60 - h_sensitivity, s_lower, v_lower])
            upper_green = np.array([60 + h_sensitivity, s_higher, v_higher])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            # Crop mask with the area of interest
            mask_green = mask_green[upper_left[1]:bottom_right[1] + 1,
                                    upper_left[0]:bottom_right[0] + 1]

            # Create mask for blue color
            lower_blue = np.array([120 - h_sensitivity, s_lower, v_lower])
            upper_blue = np.array([120 + h_sensitivity, s_higher, v_higher])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

            # Crop mask with the area of interest
            mask_blue = mask_blue[upper_left[1]:bottom_right[1] + 1,
                                  upper_left[0]:bottom_right[0] + 1]

            # Stop the program on the ESC key or check masks
            if cv2.waitKey(1) & 0xFF == 27:
                break
            elif 0 not in mask_red:
                show_frame(frame, start_time, upper_left, bottom_right,
                           color=(0, 0, 255))
            elif 0 not in mask_green:
                show_frame(frame, start_time, upper_left, bottom_right,
                           color=(0, 255, 0))
            elif 0 not in mask_blue:
                show_frame(frame, start_time, upper_left, bottom_right,
                           color=(255, 0, 0))
            else:
                show_frame(frame, start_time, upper_left, bottom_right)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


def print_usage():
    print('Usage: python var1.py [-h <int>|--hue=<int>] '
          '[-s <int,int>|--saturation=<int,int>] '
          '[-v <int,int>|--value=<int,int>] [--fps=<int>]')


def main(argv: list,
         h_sensitivity: int = 20,
         s_lower: int = 70,
         s_higher: int = 255,
         v_lower: int = 50,
         v_higher: int = 255,
         fps: int = 30):
    try:
        opts, args = getopt.getopt(argv, "h:s:v:", ["hue=", "saturation=",
                                                    "value=", "help", "fps="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ['-h', '--hue']:
            try:
                h_sensitivity = int(arg)
            except ValueError:
                print_usage()
                sys.exit(2)
        elif opt in ['-s', '--saturation']:
            try:
                arg = tuple(map(int, arg.split(',')))
                s_lower = tuple(arg)[0]
                s_higher = tuple(arg)[1]
            except ValueError:
                print_usage()
                sys.exit(2)
        elif opt in ['-v', '--value']:
            try:
                arg = tuple(map(int, arg.split(',')))
                v_lower = tuple(arg)[0]
                v_higher = tuple(arg)[1]
            except ValueError:
                print_usage()
                sys.exit(2)
        elif opt in ['--fps']:
            try:
                fps = arg
            except ValueError:
                print_usage()
                sys.exit(2)
        else:
            print_usage()
            sys.exit(2)
    show_camera(h_sensitivity, s_lower, s_higher, v_lower, v_higher, fps)


if __name__ == "__main__":
    main(sys.argv[1:])
