#include "util.h"

Mat_<double> compute_sample_weight(Mat_<double> class_weight,
                                   Mat_<double> y)
{
    // The smallest class info response to the class_weight[0]
    // The largest class info response to the class_weight[end]
    // Default
    Mat_<double> weight = Mat::ones(y.rows, 1, CV_32F);
    if (class_weight.total() != 0)
    {
        for (int i = 0; i < y.rows(); i++)
        {
            int index = static_cast<int>(y.at<double>(i, 0));
            weight.at<double>(i, 0) = class_weight.at<double>(index, 0);
        }
    }
    return weight;
}

vector<double> unique(const Mat &input, bool sort)
{
    vector<double> out;
    for (int y = 0; y < input.rows; ++y)
    {
        const double* row_ptr = input.ptr<double>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            int value = static_cast<int>(row_ptr[x]);
            if (std::find(out.begin(), out.end(), value) == out.end())
                out.push_back(value);
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());

    return out;
}
