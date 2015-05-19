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

    DecisionTreeClassifier c("Gini", "Best", 10, 2, 2, 1, 20, -1, 0, sample_weight);
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

}
