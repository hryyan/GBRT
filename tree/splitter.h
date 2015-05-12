#ifndef SPLITTER_H
#define SPLITTER_H

#include <vector>
#include <utility>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "criterion.h"

using std::vector;
using cv::Mat_;

const double FEATURE_THRESHOLD = 1e-1;

/**
 * @brief Data to track sample split
 */
struct SplitRecord
{
    int feature;            // Which feature to split on
    int pos;                // Split samples array at the given position.
                            // i.e. count of samples below threshold for feature
                            // pos is >= end if the node is a leaf
    double threshold;       // Threshold to split at.
    double improvement;     // Impurity improvement given parent node.
    double impurity_left;   // Impurity of the left split.
    double impurity_right;  // Impurity of the right split.

    void init_split(int start_pos);
};

/**
 * @brief The Splitter searches in the input space for a feature and a threshold
 * to split the samples sampls[start:end].
 *
 * The impurity computations are delegated to a criterion object.
 */
class Splitter
{
public:
    Splitter(Criterion& criterion,
             int max_feature,
             int min_samples_leaf,
             double min_weight_leaf,
             int random_state);
    virtual ~Splitter();

    /**
     * @brief Initialize the splitter.
     * @param X
     * @param y
     * @param sample_weight
     */
    virtual void init(Mat_<double> X,
                      Mat_<double> y,
                      Mat_<double> sample_weight);

    /**
     * @brief Reset splitter on node samples[start:end].
     * @param start
     * @param end
     * @return
     */
    double node_reset(int start,
                      int end);

    /**
     * @brief Find a split on onde samples[start:end].
     * @param impurity
     * @param split
     * @param n_constant_features
     */
    virtual void node_split(double impurity,
                            SplitRecord* split,
                            int* n_constant_features)=0;

    /**
     * @brief return the value of node samples[start:end]
     * @return node_value
     */
    vector<double> node_value(){
        return criterion.node_value();
    }

    /**
     * @brief Copy the impurity of node samples[start:end].
     * @return impurity
     */
    double node_impurity(){
        return criterion.node_impurity();
    }

public:
    Criterion& criterion;                // impurity Criterion
    int max_features;                   // Number of features to test
    int min_samples_leaf;               // Min samples in a leaf
    double min_weight_leaf;             // Minimum weight in a leaf

    int random_state;                   // Random state

    int n_samples;                      // X.shape[0]
    int n_features;                     // X.shape[1]
    vector<int> samples;                // Sample indices in X, y
    vector<int> features;               // Feature indices in x
    vector<int> constant_features;      // Constant features indices
    vector<double> feature_values;      // temp. array holding feature values
    double weighted_n_samples;          // Weighted number of samples

    int start;                          // Start position for the current nodes
    int end;                            // End position for the current nodes

    Mat_<double> X;
    Mat_<double> y;
    Mat_<double> sample_weight;

/**
 * The samples vector `samples` is maintained by the Splitter object such
 * that the samples contained in a node are contiguous. With this setting,
 * `node_split` reorganizes the node samples `samples[start:end]` in two
 * subsets `samples[start:pos]` and `samples[pos:end]`.
 *
 * The 1-d  `features` array of size n_features contains the features
 * indices and allows fast sampling without replacement of features.
 *
 * The 1-d `constant_features` array of size n_features holds in
 * `constant_features[:n_constant_features]` the feature ids with
 * constant values for all the samples that reached a specific node.
 * The value `n_constant_features` is given by the the parent node to its
 * child nodes.  The content of the range `[n_constant_features:]` is left
 * undefined, but preallocated for performance reasons
 * This allows optimization with depth-based tree building.
 */
};

class BaseDenseSplitter : public Splitter
{
public:
    BaseDenseSplitter(Criterion& criterion,
                      int max_features,
                      int min_samples_leaf,
                      double min_weight_leaf,
                      int random_state);
    virtual ~BaseDenseSplitter();

    /**
     * @brief Initialize the splitter.
     * @param X
     * @param y
     * @param sample_weight
     */
    virtual void init(Mat_<double> X,
                      Mat_<double> y,
                      Mat_<double> sample_weight);

    /**
     * @brief Find a split on onde samples[start:end].
     * @param impurity
     * @param split
     * @param n_constant_features
     */
//    virtual void node_split(double impurity,
//                       SplitRecord *split,
//                       int* n_constant_features);

public:
    Mat_<double> X;
};

/**
 * @brief Splitter for finding the best split
 */
class BestSplitter : public BaseDenseSplitter
{
public:
    BestSplitter(Criterion& criterion,
                 int max_features,
                 int min_samples_leaf,
                 double min_weight_leaf,
                 int random_state);
    virtual ~BestSplitter();

    /**
     * @brief Find a split on onde samples[start:end].
     * @param impurity
     * @param split
     * @param n_constant_features
     */
    virtual void node_split(double impurity,
                            SplitRecord *split,
                            int* n_constant_features);
};

class RandomSplitter : public BaseDenseSplitter
{
public:
    RandomSplitter(Criterion& criterion,
                   int max_features,
                   int min_samples_leaf,
                   double min_weight_leaf,
                   int random_state);
    virtual ~RandomSplitter();

    /**
     * @brief Find a split on onde samples[start:end].
     * @param impurity
     * @param split
     * @param n_constant_features
     */
    virtual void node_split(double impurity,
                            SplitRecord *split,
                            int* n_constant_features);
};

class PresortBestSplitter : public BaseDenseSplitter
{
public:
    PresortBestSplitter(Criterion& _criterion,
                        int _max_features,
                        int _min_samples_leaf,
                        double _min_weight_leaf,
                        int _random_state);
    virtual ~PresortBestSplitter();

    virtual void init(Mat_<double> X,
                      Mat_<double> y,
                      Mat_<double> sample_weight);

    virtual void node_split(double impurity,
                            SplitRecord *split,
                            int *n_constant_features);

public:
    vector<int> X_argsorted_ptr;

    Mat_<double> X_old;
    Mat_<int> X_argsorted;

    int n_total_samples;
    vector<uchar> sample_mask;
};

class BaseSparseSplitter : public Splitter
{
public:
    BaseSparseSplitter();
    virtual ~BaseSparseSplitter();
};

class BestSparseSplitter : public BaseSparseSplitter
{
public:
    BestSparseSplitter();
    virtual ~BestSparseSplitter();
};

class RandomSparseSplitter : public BaseSparseSplitter
{
public:
    RandomSparseSplitter();
    virtual ~RandomSparseSplitter();
};

inline int rand_int(int low, int high, int random_state)
{
    std::srand(random_state);
    int random_variable = std::rand();
    return low + random_variable % (high - low);
}

inline int rand_double(int low, int high, int random_state)
{
    std::srand(random_state);
    int random_variable = std::rand();
    return ((high - low) * (double)random_variable) / RAND_MAX + low;
}

#endif // SPLITTER_H
