# ITMO University
# Mobile Computer Vision course
# 2021
# by seniorkot & atepaevm

import getopt
import sys
import os
import time
import pickle
import csv
from pathlib import Path
from datetime import datetime

import cv2
import math
import face_recognition as fcr
from sklearn import neighbors


Path("./output").mkdir(exist_ok=True)


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


def show_frame(knn_clf: neighbors.KNeighborsClassifier,
               video=None):
    attendace = {}
    if video is not None:
        print('Not none')
        cap = cv2.VideoCapture(video)
    else:
        # print(gstreamer_pipeline(flip_method=4))
        # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4),
        #                        cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if cap.isOpened():
        # Create a window
        _ = cv2.namedWindow("MCV Lab 4", cv2.WINDOW_AUTOSIZE)

        # Check fullscreen window property to determine if it's still available
        while cv2.getWindowProperty("MCV Lab 4", 0) >= 0:
            ret, frame = cap.read()

            # Check return value (in case no frames available)
            if not ret:
                break

            # Resize to increase face detection computations
            imgS = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            face_locations = fcr.face_locations(imgS)
            if len(face_locations) > 0:
                predictions = predict(knn_clf, imgS, face_locations)

                for prediction in predictions:
                    name = prediction[0]
                    if name != 'Unknown' and name not in attendace:
                        att_time = \
                            datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                        attendace[name] = att_time
                        y1, x2, y2, x1 = prediction[1]
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        img_copy = frame.copy()
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2),
                                      (0, 255, 0), 2)
                        cv2.rectangle(img_copy, (x1, y2 - 30), (x2, y2),
                                      (0, 255, 0), cv2.FILLED)
                        text = str(name + ' ' + att_time)
                        cv2.putText(img_copy, text, (x1 + 6, y2 - 6),
                                    cv2.FONT_HERSHEY_PLAIN, 1.0,
                                    (255, 255, 255), 2)
                        filename = str('./output/' + name + '.jpg')
                        cv2.imwrite(filename, img_copy)

            # Stop the program on the ESC key or check masks
            if cv2.waitKey(1) & 0xFF == 27:
                break
            else:
                cv2.imshow('frame', frame)

        cap.release()
        cv2.destroyAllWindows()

        with open('./output/attendance.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['name', 'time'])
            w.writerows(attendace.items())
    else:
        print("Unable to open camera")


def get_encodings(data_path: str) -> dict:
    encodings = {}

    timest = time.time()
    for img in os.listdir(data_path):
        try:
            image = cv2.imread(f'{data_path}/{img}', )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            locations = fcr.face_locations(image)
            if len(locations) != 1:
                print("Image {} not suitable for training: {}"
                      .format(img, "Didn't find a face" if len(locations) < 1
                              else "Found more than one face"))
            else:
                encode = fcr.face_encodings(image)[0]
                encodings[os.path.splitext(img)[0]] = encode
        except cv2.error:
            pass
    print(f'Completed encoding of {len(encodings)} faces in '
          f'{time.time() - timest} sec')
    return encodings


def train(data_path: str,
          model_path=None,
          n_neighbors=None) -> neighbors.KNeighborsClassifier:
    if model_path is not None and os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    else:
        encodings = get_encodings(data_path)
        X = list(encodings.values())
        y = list(encodings.keys())

        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            print("Chose n_neighbors automatically:", n_neighbors)

        timest = time.time()
        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                 algorithm='ball_tree',
                                                 weights='distance')
        knn_clf.fit(X, y)
        print(f'Completed classifier training in {time.time() - timest} sec')

        # Save the trained KNN classifier
        if model_path is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(knn_clf, f)

    return knn_clf


def predict(knn_clf: neighbors.KNeighborsClassifier,
            img,
            face_locations,
            distance_threshold=0.6):
    # Find encodings for faces in the test iamge
    face_encodings = fcr.face_encodings(img, face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold
                   for i in range(len(face_locations))]

    # Predict classes and remove classifications that aren't
    # within the threshold
    return [(pred, loc) if rec else ("Unknown", loc)
            for pred, loc, rec in zip(knn_clf.predict(face_encodings),
                                      face_locations, are_matches)]


def print_usage(exit_code: int):
    print('Usage: python lab4.py [-d <dir>|--data=<dir>] '
          '[-v <path>|--video=<path>] [-c <path>|--clf==<path>]')
    sys.exit(exit_code)


def main(argv: list,
         data_path: str = './data',
         video=None,
         model_path=None):
    try:
        opts, args = getopt.getopt(argv, "d:v:c:", ["data=", "video=",
                                                    "clf=", "help"])
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
        elif opt in ['-c', '--clf']:
            model_path = arg
        elif opt in ['--help']:
            print_usage(0)
        else:
            print_usage(1)

    # Train classifier
    knn_clf = train(data_path, model_path=model_path)
    if knn_clf is None:
        print('Failed to train classifier')
        sys.exit(3)

    show_frame(knn_clf, video=video)


if __name__ == "__main__":
    main(sys.argv[1:])
