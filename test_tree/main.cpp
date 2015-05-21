#define DEBUG
#include <opencv2/opencv.hpp>
#include <QtCore>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <stdio.h>
#include <iostream>
#include "criterion.h"
#include "splitter.h"
#include "basetree.h"
#include "tree.h"
#include "treebuilder.h"
#include "util.h"
#include "criterion_test.h"
#include "splitter_test.h"
#include "decisiontree_test.h"
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
//    BestSplitter_classification_test("Gini", "test4.txt");
//    BestSplitter_classification_test("Entropy", "test4.txt");
//    BestSplitter_regression_test("MSE", "test4.txt");
//    BestSplitter_regression_test("FriedmanMSE", "test1.txt");
//    RandomSplitter_test();

    // Util_test
//    sort_apply_permutation_test();

    // DesicitionTree_test
//    DecisionTreeClassification_test("test3.txt");
<<<<<<< HEAD
    DecisionTreeRegression_test("test3.txt");
=======
    DecisionTreeRegression_test("test2.txt");
>>>>>>> 1e27a83d59ca7aa0217d9599d4463d8e96293e1b

    // Tools
}
