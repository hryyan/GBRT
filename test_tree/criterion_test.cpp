#include "criterion_test.h"
#include <stdio.h>
#include <vector>
#include <utility>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "criterion.h"
using namespace cv;
using namespace std;

int Gini_test()
{
    Mat y = Mat(10, 1, CV_64F);
    for (int i = 0; i < 5; i++)
        y.at<double>(i) = 0.;
    for (int i = 5; i < 10; i++)
        y.at<double>(i) = 1.;

    Mat sample_weight = Mat::ones(10, 1, CV_64F);

    double _weight_n_samples = 10;
    vector<int> vec;
    for (int i = 0; i < 10; i++)
        vec.push_back(i);

    int start = 0;
    int end = 10;

    Gini g = Gini();
    g.init(y, sample_weight, _weight_n_samples, vec, start, end);

    for (int i = 0; i < 11; i++)
    {
        g.update(i);
        double impurity = g.node_impurity();
        double d = g.impurity_improvement(impurity);
        pair<double, double> p = g.children_impurity();
        cout << "impurity_improvement: " << d << "\t";
        cout << "left_impurity: " << p.first << "\t" \
             << "right_impurity: " << p.second << endl;
    }
    return 0;
}

int Entropy_test()
{
    Mat y = Mat(10, 1, CV_64F);
    for (int i = 0; i < 5; i++)
        y.at<double>(i) = 0.;
    for (int i = 5; i < 10; i++)
        y.at<double>(i) = 1.;

    Mat sample_weight = Mat::ones(10, 1, CV_64F);

    double _weight_n_samples = 10;
    vector<int> vec;
    for (int i = 0; i < 10; i++)
        vec.push_back(i);

    int start = 0;
    int end = 10;

    Entropy g = Entropy();
    g.init(y, sample_weight, _weight_n_samples, vec, start, end);

    for (int i = 0; i < 11; i++)
    {
        g.update(i);
        double impurity = g.node_impurity();
        double d = g.impurity_improvement(impurity);
        pair<double, double> p = g.children_impurity();
        cout << "impurity_improvement: " << d << "\t";
        cout << "left_impurity: " << p.first << "\t" \
             << "right_impurity: " << p.second << endl;
    }
    return 0;
}

int MSE_test()
{
    Mat y = Mat(10, 1, CV_64F);
    for (int i = 0; i < 5; i++)
        y.at<double>(i) = 0.;
    for (int i = 5; i < 10; i++)
        y.at<double>(i) = 1.;

    Mat sample_weight = Mat::ones(10, 1, CV_64F);

    double _weight_n_samples = 10;
    vector<int> vec;
    for (int i = 0; i < 10; i++)
        vec.push_back(i);

    int start = 0;
    int end = 10;

    MSE g = MSE();
    g.init(y, sample_weight, _weight_n_samples, vec, start, end);

    for (int i = 0; i < 11; i++)
    {
        g.update(i);
        double impurity = g.node_impurity();
        double d = g.impurity_improvement(impurity);
        pair<double, double> p = g.children_impurity();
        cout << "impurity_improvement: " << d << "\t";
        cout << "left_impurity: " << p.first << "\t" \
             << "right_impurity: " << p.second << endl;
    }
    return 0;
}

int FriedmanMSE_test()
{
    Mat y = Mat(10, 1, CV_64F);
    for (int i = 0; i < 5; i++)
        y.at<double>(i) = 0.;
    for (int i = 5; i < 10; i++)
        y.at<double>(i) = 1.;

    Mat sample_weight = Mat::ones(10, 1, CV_64F);

    double _weight_n_samples = 10;
    vector<int> vec;
    for (int i = 0; i < 10; i++)
        vec.push_back(i);

    int start = 0;
    int end = 10;

    FriedmanMSE g = FriedmanMSE();
    g.init(y, sample_weight, _weight_n_samples, vec, start, end);

    for (int i = 0; i < 11; i++)
    {
        g.update(i);
        double impurity = g.node_impurity();
        double d = g.impurity_improvement(impurity);
        pair<double, double> p = g.children_impurity();
        cout << "impurity_improvement: " << d << "\t";
        cout << "left_impurity: " << p.first << "\t" \
             << "right_impurity: " << p.second << endl;
    }
    return 0;
}
