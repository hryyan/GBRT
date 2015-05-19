#include "decisiontree_test.h"
#include <QtCore>
#include <utility>
#include <opencv2/opencv.hpp>
#include "tree.h"
#include "tools.h"
using std::pair;
using cv::Mat;

int DecisionTreeClassification_test(QString filename)
{
    QString fn = QString("../test_data/Classification/").append(filename);
    pair<Mat, Mat> pMat = read_data_from_txt_classification(fn);
    Mat X = pMat.first;
    Mat y = pMat.second;

    Mat sample_weight = Mat::ones(200, 1, CV_64F);
    Mat class_weight = Mat::ones(0, 0, CV_64F);

    DecisionTreeClassifier c("Gini", "Best", 10, 1, 1, 1, 20, -1, 0, class_weight);
    c.fit(X, y, sample_weight);
    Mat result = c.predict(X);
    for (int i = 0; i < result.total(); i++)
    {
        if (result.at<double>(i) == y.at<double>(i))
            cout << "Correct" << endl;
        else
            cout << "Wrong" << " " << result.at<double>(i) << " " << y.at<double>(i) << endl;
    }
}

int DecisionTreeRegression_test(QString filename)
{
    QString fn = QString("../test_data/Regression/").append(filename);
    pair<Mat, Mat> pMat = read_data_from_txt_regression(fn);
    Mat X = pMat.first;
    Mat y = pMat.second;

    Mat sample_weight = Mat::ones(200, 1, CV_64F);
    Mat class_weight = Mat::ones(0, 0, CV_64F);

    DecisionTreeRegressor r("MSE", "Best", 10, 1, 1, 1, 20, -1, 0, class_weight);
    r.fit(X, y, sample_weight);
    Mat result = r.predict(X);
    for (int i = 0; i < result.total(); i++)
    {
        if (result.at<double>(i) == y.at<double>(i))
            cout << "Correct" << endl;
        else
            cout << "Wrong" << " " << result.at<double>(i) << " " << y.at<double>(i) << endl;
    }
}
