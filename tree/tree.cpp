#include "tree.h"
#include <stdlib.h>
#include <set>
#include "criterion.h"
#include "splitter.h"
#include "basetree.h"
#include "treebuilder.h"

BaseDecisionTree::BaseDecisionTree(char* criterion_name,
                                   char* splitter_name,
                                   int max_depth,
                                   int min_samples_split,
                                   int min_samples_leaf,
                                   double min_weight_fraction_leaf,
                                   int max_features,
                                   int max_leaf_nodes,
                                   int random_state,
                                   Mat class_weight,
                                   int is_classification)
// Need constructor paras for Tree
    : _criterion_name(criterion_name),
      _splitter_name(splitter_name),
      _max_depth(max_depth),
      _min_samples_split(min_samples_split),
      _min_samples_leaf(min_samples_leaf),
      _min_weight_fraction_leaf(min_weight_fraction_leaf),
      _max_features(max_features),
      _max_leaf_nodes(max_leaf_nodes),
      _random_state(random_state),
      _class_weight(class_weight),
      _n_samples(0),
      _n_features(0),
      _is_classification(is_classification)
{

}

BaseDecisionTree::~BaseDecisionTree()
{

}

int BaseDecisionTree::fit(Mat X,
                          Mat y,
                          Mat sample_weight)
{
    // Validation
    if (X.rows == 0 || X.cols == 0)
        return 1;

    // Determine output setting
    _n_samples = X.rows;
    _n_features = X.cols;

    // Reshape y to shape[n_samples, 1]
    y = y.reshape(1, y.total());

    // Validation
    if (y.rows != _n_samples)
        return 2;

    // Calculate class_weight
    Mat expended_class_weight(0, 0, CV_64F);
    // Get class_weight
    if (_class_weight.total() != 0)
        expended_class_weight = compute_sample_weight(_class_weight, y);

    // Validation
    if (_max_depth <= 0)
        _max_depth = static_cast<int>(pow(2, 31) - 1);
    if (_max_leaf_nodes <= 0)
        _max_leaf_nodes = -1;
    if (_max_features <= 0)
        _max_features = _n_features;
    if (_max_leaf_nodes > -1 && _max_leaf_nodes < 2)
        return 3;
    if (_min_samples_split <= 0)
        return 4;
    if (_min_samples_leaf <= 0)
        return 5;
    if (_min_weight_fraction_leaf >= 0 && _min_weight_fraction_leaf <= 0.5)
        return 6;

    // Get _n_classes
    std::set<double> s;
    for (int i = 0; i < y.total(); i++)
    {
        s.insert(y.at<double>(i));
    }
    int _n_classes = s.size();

    // Set samples' weight
    if (expended_class_weight.total())
    {
        for (int i = 0; i < sample_weight.total(); i++)
        {
            sample_weight.at<double>(i, 0) = sample_weight.at<double>(i, 0) * \
                                             expended_class_weight.at<double>(i, 0);
        }
    }
    else
    {
        sample_weight = expended_class_weight;
    }

    // Set min_weight_fraction_leaf
//    if (_min_weight_fraction_leaf != 0.)
//        _min_weight_fraction_leaf = _min_weight_fraction_leaf * cv::sum(sample_weight)[0];
//    else
//        _min_weight_fraction_leaf = 0.;

    // Set min_samples_split
    _min_samples_split = max(_min_samples_split, 2 * _min_samples_leaf);

    // Select a Criterion
    if (strcmp(_criterion_name, "Gini") == 0)
        _criterion = new Gini();
    else if (strcmp(_criterion_name, "Entropy") == 0)
        _criterion = new Entropy();
    else if (strcmp(_criterion_name, "MSE") == 0)
        _criterion = new MSE();
    else if (strcmp(_criterion_name, "FriedmanMSE") == 0)
        _criterion = new FriedmanMSE();
    else
        exit(1);

    // Select a Splitter
    if (strcmp(_splitter_name, "Best") == 0)
        _splitter = new BestSplitter(_criterion,
                                     _max_features,
                                     _min_samples_leaf,
                                     _min_weight_fraction_leaf,
                                     _random_state);
    else if (strcmp(_splitter_name, "Random") == 0)
        _splitter = new RandomSplitter(_criterion,
                                       _max_features,
                                       _min_samples_leaf,
                                       _min_weight_fraction_leaf,
                                       _random_state);
    else
        exit(1);

    // Select a Tree
    _tree = new Tree(_n_features, _n_classes);

    // Select a Tree Builder
    if (_max_leaf_nodes < 0)
        _tree_builder = new DepthFirstBuilder(_splitter,
                                              _min_samples_split,
                                              _min_samples_leaf,
                                              _min_weight_fraction_leaf,
                                              _max_depth,
                                              _max_leaf_nodes);
    else
        _tree_builder = new BestFirstTreeBuilder(_splitter,
                                                 _min_samples_split,
                                                 _min_samples_leaf,
                                                 _min_weight_fraction_leaf,
                                                 _max_depth,
                                                 _max_leaf_nodes);

    // Build a tree
    _tree_builder->build(_tree, X, y, sample_weight);
}

Mat BaseDecisionTree::predict(Mat X)
{
    int n_samples = X.rows;
    int n_features = X.cols;

    Mat proba = _tree->predict(X);

    if (_is_classification)
    {

    }
    else
    {
        return proba;
    }
}

Mat BaseDecisionTree::feature_importances()
{

}

DecisionTreeClassifier::DecisionTreeClassifier(char* criterion_name,
                                               char* splitter_name,
                                               int max_depth,
                                               int min_samples_split,
                                               int min_samples_leaf,
                                               double min_weight_fraction_leaf,
                                               int max_features,
                                               int max_leaf_nodes,
                                               int random_state,
                                               Mat class_weight)
    : BaseDecisionTree(criterion_name,
                       splitter_name,
                       max_depth,
                       min_samples_split,
                       min_samples_leaf,
                       min_weight_fraction_leaf,
                       max_features,
                       max_leaf_nodes,
                       random_state,
                       class_weight,
                       0)
{

}

DecisionTreeClassifier::~DecisionTreeClassifier()
{

}

DecisionTreeRegressor::DecisionTreeRegressor(char* criterion_name,
                                             char* splitter_name,
                                             int max_depth,
                                             int min_samples_split,
                                             int min_samples_leaf,
                                             double min_weight_fraction_leaf,
                                             int max_features,
                                             int max_leaf_nodes,
                                             int random_state,
                                             Mat class_weight)
    : BaseDecisionTree(criterion_name,
                       splitter_name,
                       max_depth,
                       min_samples_split,
                       min_samples_leaf,
                       min_weight_fraction_leaf,
                       max_features,
                       max_leaf_nodes,
                       random_state,
                       class_weight,
                       1)
{

}

DecisionTreeRegressor::~DecisionTreeRegressor()
{

}
