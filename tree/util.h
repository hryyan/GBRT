#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

using std::vector;
using cv::Mat;
using cv::Mat_;

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

template <typename T, typename Compare>
std::vector<int> sort_permutation(
    std::vector<T> const& vec,
    Compare compare)
{
    std::vector<int> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](int i, int j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(
    std::vector<T> const& vec,
    std::vector<int> const& p)
{
    std::vector<T> sorted_vec(p.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](int i){ return vec[i]; });
    return sorted_vec;
}
#endif // UTIL_H
