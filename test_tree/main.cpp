#define DEBUG
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <stdio.h>
#include <iostream>
#include "criterion_test.h"
using namespace cv;
using namespace std;

int main()
{
//    Gini_test();
//    Entropy_test();
//    MSE_test();
//    FriedmanMSE_test();
    vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    cout << vec.front() << endl;
}
