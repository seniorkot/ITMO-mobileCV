#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    uint8_t* bgr_arr;

    if (argc != 2) {
        cout << "Usage: var1 image_name" << endl;
        return -1;
    }

    // Read image
    Mat bgr_image;
    bgr_image = imread(argv[1], IMREAD_COLOR); // IMREAD_COLOR - converted to 3 channel BGR
    if (!bgr_image.data) {
        cout << "Could not open the image" << endl;
        return -1;
    }
    if (bgr_image.isContinuous()) {
        bgr_arr = bgr_image.data;
    }
    else {
        cout << "data is not continuous" << endl;
        return -2;
    }

    return 0;
}
