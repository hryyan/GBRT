#include "tree.h"
#include "criterion.h"
#include "splitter.h"

BaseDecisionTree::BaseDecisionTree(Criterion* criterion,
                                   Splitter* splitter,
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
    : _criterion(criterion),
      _splitter(splitter),
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
      _is_classification(is_classification),
      _max_features(0)
{

}

BaseDecisionTree::~BaseDecisionTree()
{

}

int BaseDecisionTree::fit(Mat _X,
                          Mat _y,
                          Mat sample_weight)
{
    // Validation
    if (_X.rows == 0 || _X.cols == 0)
        return 1;

    // Determine output setting
    _n_samples = _X.rows;
    _n_features = _X.cols;

    // Reshape y to shape[n_samples, 1]
    _y = _y.reshape(1, _y.total());

    // Validation
    if (_y.rows != _n_samples)
        return 2;

    // Calculate class_weight
    Mat expended_class_weight(0, 0, CV_32F);
    // Get class_weight
    if (_class_weight.total() != 0)
        expended_class_weight = compute_sample_weight(_class_weight, _y);

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
    if (_min_weight_fraction_leaf != 0.)
        _min_weight_fraction_leaf = _min_weight_fraction_leaf * cv::sum(sample_weight);
    else
        _min_weight_fraction_leaf = 0.;

    // Set min_samples_split
    _min_samples_split = max(_min_samples_split, 2 * _min_samples_leaf);




}

Mat BaseDecisionTree::predict(Mat _X)
{

}

Mat BaseDecisionTree::feature_importances()
{

}

DecisionTreeClassifier::DecisionTreeClassifier(Criterion* criterion,
                                               Splitter* splitter,
                                               int max_depth,
                                               int min_samples_split,
                                               int min_samples_leaf,
                                               double min_weight_fraction_leaf,
                                               int max_features,
                                               int max_leaf_nodes,
                                               int random_state,
                                               Mat class_weight)
    : BaseDecisionTree(criterion,
                       splitter,
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

DecisionTreeRegressor::DecisionTreeRegressor(Criterion* criterion,
                                             Splitter* splitter,
                                             int max_depth,
                                             int min_samples_split,
                                             int min_samples_leaf,
                                             double min_weight_fraction_leaf,
                                             int max_features,
                                             int max_leaf_nodes,
                                             int random_state,
                                             Mat class_weight)
    : BaseDecisionTree(criterion,
                       splitter,
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
