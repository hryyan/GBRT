#ifndef BASETREE_H
#define BASETREE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
using std::vector;
using cv::Mat;
using cv::Mat_;

class Criterion;
class Splitter;

enum TreeType
{
    TREE_LEAF,
    TREE_UNDEFINED
};

/**
 * @brief Base storage structure for the nodes in a Tree object
 */
struct Node
{
    int left_child;                 // id of the left child of the node
    int right_child;                // id of the right child of the node
    int feature;                    // Feature used for splitting the node
    double threshold;               // Threshold value at the node
    double impurity;                // Impurity of the node (i.e., the value of the criterion)
    int n_node_samples;             // Number of samples at the node
    double weighted_n_node_samples; // Weighted number of samples at the node

    bool operator== (const Node& a){
        if (a.left_child == left_child &&
            a.right_child == right_child &&
            a.feature == feature &&
            a.threshold == threshold &&
            a.impurity == impurity &&
            a.n_node_samples == n_node_samples &&
            a.weighted_n_node_samples == weighted_n_node_samples)
            return true;
        return false;
    }
};

/**
 * @brief The Tree object is a binary tree structure constructed by the
 * TreeBuilder. The tree structure is used for predictions and
 * feature importances.
 */
class Tree
{
public:
    /**
     * Array-based representation of a binary decision tree.
     * The binary tree is represented as a number of parallel arrays. The i-th
     * element of each array holds information about the node `i`. Node 0 is the
     * tree's root. You can find a detailed description of all arrays in
     * `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
     * nodes, resp. Inpredict this case the values of nodes of the other type are
     * arbitrary!
     *
     * Attributes
     * ----------
     * node_count : int
     *  The number of nodes (internal nodes + leaves) in the tree.
     *
     * capacity : int
     *  The current capacity (i.e., size) of the arrays, which is at least as
     * great as `node_count`.
     *
     * max_depth : int
     *  The maximal depth of the tree.
     *
     * children_left : array of int, shape [node_count]
     *  children_left[i] holds the node id of the left child of node i.
     *  For leaves, children_left[i] == TREE_LEAF. Otherwise,
     *  children_left[i] > i. This child handles the case where
     *  X[:, feature[i]] <= threshold[i].
     *
     * children_right : array of int, shape [node_count]
     *  children_right[i] holds the node id of the right child of node i.
     *  For leaves, children_right[i] == TREE_LEAF. Otherwise,
     *
     * children_right[i] > i. This child handles the case where
     *  X[:, feature[i]] > threshold[i].
     *
     * feature : array of int, shape [node_count]
     *  feature[i] holds the feature to split on, for the internal node i.
     *
     * threshold : array of double, shape [node_count]
     *  threshold[i] holds the threshold for the internal node i.
     *
     * value : array of double, shape [node_count, n_outputs, max_n_classes]
     *  Contains the constant prediction value of each node.
     *
     * impurity : array of double, shape [node_count]
     *  impurity[i] holds the impurity (i.e., the value of the splitting
     *  criterion) at node i.
     *
     * n_node_samples : array of int, shape [node_count]
     *  n_node_samples[i] holds the number of training samples reaching node i.
     *
     * weighted_n_node_samples : array of int, shape [node_count]
     *  weighted_n_node_samples[i] holds the weighted number of training samples
     *  reaching node i.
     *
     * # Wrap for outside world.
     * # WARNING: these reference the current `nodes` and `value` buffers, which
     * # must not be be freed by a subsequent memory allocation.
     * # (i.e. through `_resize` or `__setstate__`)
     **/
    Tree(int _n_features,
         int _n_classes);
    ~Tree();

    /**
     * @brief Add a node to the tree
     * @param parent
     * @param is_left
     * @param is_leaf
     * @param feature
     * @param threshold
     * @param impurity
     * @param n_node_samples
     * @param weighted_n_node_samples
     * @return
     */
    int _add_node(int parent,
                  bool is_left,
                  bool is_leaf,
                  int feature,
                  double threshold,
                  double impurity,
                  int n_node_samples,
                  double weighted_n_node_samples);
    /**
     * @brief Predict target for X.
     * @param X
     * @return
     */
    Mat predict(Mat X);

    /**
     * @brief Finds the terminal region (=leaf node) for each sample in X.
     * @param X
     * @return
     */
    Mat apply(Mat X);

    /**
     * @brief Finds the terminal region (=leaf node) for each sample in X.
     * @param X
     * @return
     */
    Mat _apply_dense(Mat X);

    /**
     * @brief Computes the importance of each feature (aka variable).
     * @param normalize
     * @return
     */
    Mat compute_feature_importances(bool normalize);

public:
    // Input/Output layout
    int _n_features;             // Number of features in X
    int _n_classes;              // max(n_classes)

    // Inner structures: values are stored separately from node structure,
    // since size is determined at runtime.
    int _max_depth;              // Mat depth of the tree
    int _node_count;             // Counter for node IDs
    int _capacity;               // Capacity of tree, in terms of nodes
    vector<Node> _nodes;         // Array of nodes
    vector<double> _value;       // The value of every node
};

#endif // BASETREE_H
