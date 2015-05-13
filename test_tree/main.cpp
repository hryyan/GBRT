#define DEBUG
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <stdio.h>
#include <iostream>
#include "criterion.h"
#include "splitter.h"
#include "util.h"
#include "criterion_test.h"
#include "splitter_test.h"
#include "util_test.h"
#include "tools.h"
using namespace cv;
using namespace std;

int main()
{
    // Criterion_test
//    Gini_test();
//    Entropy_test();
//    MSE_test();
//    FriedmanMSE_test();

    // Splitter_test

    // Util_test
//    sort_apply_permutation_test();

    // Tools
    pair<Mat, Mat> pmat = read_data_from_txt("../test1.txt");
    Mat X = pmat.first;
    Mat y = pmat.second;

    Mat sample_weight = Mat::ones(200, 1, CV_64F);

    vector<int> vec;
    for (int i = 0; i < 10; i++)
        vec.push_back(i);

   SplitRecord split;
    int const_feature = 0;
    Gini g = Gini();
    BestSplitter bs(&g, 20, 2, 1., 0);
    bs.init(X, y, sample_weight);
    double weighted_n_samples = bs.node_reset(0, 200);
    double impurity = bs.node_impurity();
    bs.node_split(impurity, &split, &const_feature);
}
