#ifndef BASETREE_H
#define BASETREE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include "criterion.h"
#include "splitter.h"
using std::vector;
using cv::Mat_;
using cv::Mat;

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
    Tree(int _n_features,
         int _n_classes);
    ~Tree();

    int _add_node(int parent,
                  bool is_left,
                  bool is_leaf,
                  int feature,
                  double threshold,
                  double impurity,
                  int n_node_samples,
                  double weighted_n_node_samples);


    Mat_<double> predict(Mat_<double> X);
    Mat_<double> apply(Mat_<double> X);
    Mat_<double> _apply_dense(Mat_<double> X);

    Mat_<double> compute_feature_importances(bool normalize);

public:
    // Input/Output layout
    int n_features;             // Number of features in X
    int n_classes;              // max(n_classes)

    // Inner structures: values are stored separately from node structure,
    // since size is determined at runtime.
    int max_depth;              // Mat depth of the tree
    int node_count;             // Counter for node IDs
    int capacity;               // Capacity of tree, in terms of nodes
    vector<Node> nodes;         // Array of nodes
    vector<double> value;       // The value of every node
};

#endif // BASETREE_H
