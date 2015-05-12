#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

using std::vector;
using cv::Mat;
using cv::Mat_;

/**
 * @brief Estimate class weights for unbalanced datasets
 * @param class_weight dict or None, if None is given, the class weights will be uniform.
 * @param y Array of original class labels per sample.
 * @return Mat, shape = [n_classes, 1]
 */
Mat_<double> compute_class_weight(Mat_<double> class_weight,
                                  Mat_<double> y);

/**
 * @brief Estimate sample weights by class for unbalanced datasets.
 * @param class_weight The weight of classes from 0 to n, shape = [n_classes, 1]
 * @param y The class information of every sample, shape = [n_samples, 1]
 * @return Weight of every sample, shape = [n_samples]
 */
Mat_<double> compute_sample_weight(Mat_<double> class_weight,
                                   Mat_<double> y);
/**
 * @brief Get the unique value of Mat
 * @param input Mat, every value is double
 * @param sort
 * @return The unique value
 */
vector<double> unique(const Mat& input, bool sort=false);

#endif // UTIL_H
