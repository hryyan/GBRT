#ifndef TOOLS_H
#define TOOLS_H
#include <iostream>
#include <QtCore>
#include <utility>
#include <opencv2/opencv.hpp>
using cv::Mat;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

pair<Mat, Mat> read_data_from_txt_classification(QString str);
pair<Mat, Mat> read_data_from_txt_regression(QString str);

#endif // TOOLS_H
