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
 * @param src Source matrix
 * @param dst Destination matrix
 * @param width Image width
 * @param height Image height
 */
void mean_filter(const Mat src, Mat dst, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Destination pixel
            Vec3b dst_px = dst.at<Vec3b>(Point(x, y));

            // X and Y coords
            int x_lt, x_rt, y_top, y_bot;
            x_lt = x == 0 ? 0 : x - 1;
            x_rt = x + 1 == width ? x : x + 1;
            y_top = y == 0 ? 0 : y - 1;
            y_bot = y + 1 == height ? y : y + 1;

            // Source pixels
            Vec3b src_px1 = src.at<Vec3b>(Point(x_lt, y_top));
            Vec3b src_px2 = src.at<Vec3b>(Point(x, y_top));
            Vec3b src_px3 = src.at<Vec3b>(Point(x_rt, y_top));
            Vec3b src_px4 = src.at<Vec3b>(Point(x_lt, y));
            Vec3b src_px5 = src.at<Vec3b>(Point(x, y));
            Vec3b src_px6 = src.at<Vec3b>(Point(x_rt, y));
            Vec3b src_px7 = src.at<Vec3b>(Point(x_lt, y_bot));
            Vec3b src_px8 = src.at<Vec3b>(Point(x, y_bot));
            Vec3b src_px9 = src.at<Vec3b>(Point(x_rt, y_bot));

            for (int i = 0; i < 3; i++) {
                dst_px.val[i] = (src_px1.val[i] +
                        src_px2.val[i] +
                        src_px3.val[i] +
                        src_px4.val[i] +
                        src_px5.val[i] +
                        src_px6.val[i] +
                        src_px7.val[i] +
                        src_px8.val[i] +
                        src_px9.val[i]) / 9;
            }

            dst.at<Vec3b>(Point(x, y)) = dst_px;
        }
    }
}

/**
 * Implementation of mean filter using NEON command system.
 *
 * @param src Source data array
 * @param dst Destination data array
 * @param width Image width
 * @param height Image height
 */
void mean_filter_neon(const uint8_t* src, uint8_t* dst, int width, int height) {

    // Mul width and height with 3
    width *= 3;
    height *= 3;

    for (int x = 0; x < width; x += 8 * 3) {
        for (int y = 0; y < height; y++) {

            // X and Y coords
            int x_lt, x_rt, y_top, y_bot;
            x_lt = x == 0 ? 0 : x - 3;
            x_rt = x + 3 == width ? x : x + 3;
            y_top = y == 0 ? 0 : y - 3;
            y_bot = y + 3 == height ? y : y + 3;

            // Load pixels into 27 registers split by channel
            uint8x8x3_t src_px1 = vld3_u8(&src[width * y_top + x_lt]);
            uint8x8x3_t src_px2 = vld3_u8(&src[width * y_top + x]);
            uint8x8x3_t src_px3 = vld3_u8(&src[width * y_top + x_rt]);
            uint8x8x3_t src_px4 = vld3_u8(&src[width * y + x_lt]);
            uint8x8x3_t src_px5 = vld3_u8(&src[width * y + x]);
            uint8x8x3_t src_px6 = vld3_u8(&src[width * y + x_rt]);
            uint8x8x3_t src_px7 = vld3_u8(&src[width * y_bot + x_lt]);
            uint8x8x3_t src_px8 = vld3_u8(&src[width * y_bot + x]);
            uint8x8x3_t src_px9 = vld3_u8(&src[width * y_bot + x_rt]);

            for (int i = 0; i < 3; i++) {
                uint16x8_t temp;
                uint8x8_t result;

                temp = vaddl_u8(src_px1.val[i], src_px2.val[i]);
                temp = vaddw_u8(temp, src_px3.val[i]);
                temp = vaddw_u8(temp, src_px4.val[i]);
                temp = vaddw_u8(temp, src_px5.val[i]);
                temp = vaddw_u8(temp, src_px6.val[i]);
                temp = vaddw_u8(temp, src_px7.val[i]);
                temp = vaddw_u8(temp, src_px8.val[i]);
                temp = vaddw_u8(temp, src_px9.val[i]);

                // TODO: Div by 9 and put the result into result variable

//                vst1_u8(dst, result);
            }
        }
    }

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

    // Call mean filter function
    Mat mean_dst(height, width, bgr_image.type());
    auto t1_mean = chrono::high_resolution_clock::now();
    mean_filter(bgr_image, mean_dst, width, height);
    auto t2_mean = chrono::high_resolution_clock::now();
    auto duration_mean = chrono::duration_cast<chrono::microseconds>(t2_mean-t1_mean).count();
    cout << "Mean filter duration: ";
    cout << duration_mean << " us" << endl;
    imwrite("mean.png", mean_dst);

    // Call neon mean filter function
    Mat neon_dst(height, width, bgr_image.type());
    auto t1_neon = chrono::high_resolution_clock::now();
    mean_filter_neon(bgr_arr, neon_dst.data, width, height);
    auto t2_neon = chrono::high_resolution_clock::now();
    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
    cout << "NEON mean filter duration: ";
    cout << duration_neon << " us" << endl;
    imwrite("neon.png", neon_dst);

    return 0;
}
