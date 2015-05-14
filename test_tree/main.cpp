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
    BestSplitter_test("FriedmanMSE");
//    RandomSplitter_test();

    // Util_test
//    sort_apply_permutation_test();

    // Tools
}
