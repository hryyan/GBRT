#ifndef TREEBUILDER_H
#define TREEBUILDER_H

#include <cmath>
#include <stack>
#include <queue>
#include <opencv2/opencv.hpp>
#include "tree.h"
using std::vector;
using std::stack;
using cv::Mat;
using cv::Mat_;

class Criterion;
class Splitter;

const double MIN_IMPURITY_SPLIT = 1e-7;

struct N
{
    N(int _start,
      int _end,
      int _depth,
      int _parent,
      bool _is_left,
      double _impurity,
      int _n_constant_feautes)
        : start(_start),
          end(_end),
          depth(_depth),
          parent(_parent),
          is_left(_is_left),
          impurity(_impurity),
          n_constant_features(_n_constant_feautes){
    };

    int start;
    int end;
    int depth;
    int parent;
    bool is_left;
    double impurity;
    int n_constant_features;
};

class TreeBuilder
{
public:
    TreeBuilder(Splitter& splitter,
                int min_samples_split,
                int min_samples_leaf,
                double min_weight_leaf,
                int max_depth,
                int max_leaf_nodes);
    virtual ~TreeBuilder();

    /**
     * @brief Build a decision tree from the training set (X, y)
     * @param tree
     * @param X
     * @param y
     * @param sample_weight
     */
    virtual void build(BaseDecisionTree& tree,
                       Mat_<double> X,
                       Mat_<double> y,
                       Mat_<double> sample_weight)=0;
public:
    Splitter& splitter;
    int min_samples_split;
    int min_samples_leaf;
    double min_weight_leaf;
    int max_depth;
    int max_leaf_nodes;

    Mat_<double> sample_weight;
};

class DepthFirstBuilder : public TreeBuilder
{
public:
    /**
     * @brief Build a decision tree in depth-first fashion
     * @param splitter
     * @param min_samples_split
     * @param min_samples_leaf
     * @param min_weight_leaf
     * @param max_depth
     * @param max_leaf_nodes
     */
    DepthFirstBuilder(Splitter& splitter,
                      int min_samples_split,
                      int min_samples_leaf,
                      double min_weight_leaf,
                      int max_depth,
                      int max_leaf_nodes);
    virtual ~DepthFirstBuilder();

    /**
     * @brief Build a decision tree from the training set (X, y)
     * @param tree
     * @param X
     * @param y
     * @param sample_weight
     */
    virtual void build(BaseDecisionTree& tree,
                       Mat_<double> X,
                       Mat_<double> y,
                       Mat_<double> sample_weight);

public:
    Mat_<double> sample_weight;
};

class BestFirstTreeBuilder : public TreeBuilder
{
public:
    /**
     * @brief Build a decision tree in best-first fashion.
     * The best node to expand is given by the node at the frontier that has the
     * highest impurity improvement.
     * Note: this TreeBuilder will ignore tree.max_depth
     * @param splitter
     * @param min_samples_split
     * @param min_samples_leaf
     * @param min_weight_leaf
     * @param max_depth
     * @param max_leaf_nodes
     */
    BestFirstTreeBuilder(Splitter& splitter,
                         int min_samples_split,
                         int min_samples_leaf,
                         double min_weight_leaf,
                         int max_depth,
                         int max_leaf_nodes);
    virtual ~BestFirstTreeBuilder();

    /**
     * @brief Build a decision tree from the training set (X, y)
     * @param tree
     * @param X
     * @param y
     * @param sample_weight
     */
    virtual void build(BaseDecisionTree& tree,
                       Mat_<double> X,
                       Mat_<double> y,
                       Mat_<double> sample_weight);

    /**
     * @brief Adds node w/ partition [start, end) to the frontier
     * @param splitter
     * @param tree
     * @param start
     * @param end
     * @param impurity
     * @param is_first
     * @param is_left
     * @param parent
     * @param depth
     * @param res
     * @return
     */
    int _add_split_node(Splitter& splitter,
                        BaseDecisionTree& tree,
                        int start,
                        int end,
                        double impurity,
                        bool is_first,
                        bool is_left,
                        Node* parent,
                        int depth,
                        N* res);

public:
    int max_leaf_nodes;
};

#endif // TREEBUILDER_H
