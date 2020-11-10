#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

/**
 * Implementation of mean filter.
 *
 * @param src Source data array
 * @param dst Destination
 * @param num_pixels Number of pixels
 */
void mean_filter(const uint8_t* src, uint8_t* dst, int num_pixels) {
    // TODO: Implement
}

/**
 * Implementation of mean filter using NEON command system.
 *
 * @param src Source data array
 * @param dst Destination
 * @param num_pixels Number of pixels
 */
void mean_filter_neon(const uint8_t* src, uint8_t* dst, int num_pixels) {
    // TODO: Implement
}

/**
 * Main function.
 *
 * @param argc Arguments count
 * @param argv Arguments array
 * @return 0 - success, -1 - bad params, -2 - image not found, -3 - bad input data
 */
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
        return -2;
    }
    if (bgr_image.isContinuous()) {
        bgr_arr = bgr_image.data;
    }
    else {
        cout << "data is not continuous" << endl;
        return -3;
    }

    // Get image params
    int width = bgr_image.cols;
    int height = bgr_image.rows;
    int num_pixels = width * height;

    // Call mean filter function
    Mat mean_dst(height, width, bgr_image.type());
    auto t1_mean = chrono::high_resolution_clock::now();
    mean_filter(bgr_arr, mean_dst, num_pixels);
    auto t2_mean = chrono::high_resolution_clock::now();
    auto duration_mean = chrono::duration_cast<chrono::microseconds>(t2_mean-t1_mean).count();
    cout << "Mean filter duration: ";
    cout << duration_mean << " us" << endl;
    imwrite("mean.png", mean_dst);

    // Call neon mean filter function
    Mat neon_dst(height, width, bgr_image.type());
    auto t1_neon = chrono::high_resolution_clock::now();
    mean_filter_neon(bgr_arr, neon_dst, num_pixels);
    auto t2_neon = chrono::high_resolution_clock::now();
    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
    cout << "NEON mean filter duration: ";
    cout << duration_neon << " us" << endl;
    imwrite("neon.png", neon_dst);

    return 0;
}
