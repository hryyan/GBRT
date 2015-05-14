#include "splitter_test.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "splitter.h"
#include "tools.h"
using namespace cv;
using namespace std;

int BestSplitter_classification_test(char* criterion_name, QString filename)
{
    QString fn = QString("../test_data/Classification/").append(filename);
    pair<Mat, Mat> pmat = read_data_from_txt_classification(fn);
    Mat X = pmat.first;
    Mat y = pmat.second;

    Mat sample_weight = Mat::ones(200, 1, CV_64F);

    vector<int> vec;
    for (int i = 0; i < 10; i++)
        vec.push_back(i);

    Criterion* g;
    if (strcmp(criterion_name, "Gini") == 0)
        g = new Gini();
    else if (strcmp(criterion_name, "Entropy") == 0)
        g = new Entropy();
    else if (strcmp(criterion_name, "MSE") == 0)
        g = new MSE();
    else if (strcmp(criterion_name, "FriedmanMSE") == 0)
        g = new FriedmanMSE();

    SplitRecord split;
    int const_feature = 0;
    BestSplitter bs(g, 20, 2, 1., 0);
    bs.init(X, y, sample_weight);
    double weighted_n_samples = bs.node_reset(0, 200);
    double impurity = bs.node_impurity();
    bs.node_split(impurity, &split, &const_feature);
}

int BestSplitter_regression_test(char* criterion_name, QString filename)
{
    QString fn = QString("../test_data/Regression/").append(filename);
    pair<Mat, Mat> pmat = read_data_from_txt_regression(fn);
    Mat X = pmat.first;
    Mat y = pmat.second;

    Mat sample_weight = Mat::ones(200, 1, CV_64F);

    vector<int> vec;
    for (int i = 0; i < 10; i++)
        vec.push_back(i);

    Criterion* g;
    if (strcmp(criterion_name, "Gini") == 0)
        g = new Gini();
    else if (strcmp(criterion_name, "Entropy") == 0)
        g = new Entropy();
    else if (strcmp(criterion_name, "MSE") == 0)
        g = new MSE();
    else if (strcmp(criterion_name, "FriedmanMSE") == 0)
        g = new FriedmanMSE();

    SplitRecord split;
    int const_feature = 0;
    BestSplitter bs(g, 20, 2, 1., 0);
    bs.init(X, y, sample_weight);
    double weighted_n_samples = bs.node_reset(0, 200);
    double impurity = bs.node_impurity();
    bs.node_split(impurity, &split, &const_feature);
}
