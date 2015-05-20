#ifndef TREEBUILDER_H
#define TREEBUILDER_H

#include <opencv2/opencv.hpp>
using cv::Mat;

class Criterion;
class Splitter;
class Node;
class Tree;

const double MIN_IMPURITY_SPLIT = 1e-7;

struct N
{
    int start;
    int end;
    int depth;
    int parent;
    bool is_left;
    double impurity;
    int n_constant_features;

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
    }
};

struct P
{
    int node_id;
    int start;
    int end;
    int pos;
    int depth;
    bool is_leaf;
    double impurity;
    double impurity_left;
    double impurity_right;
    double improvement;

    P(int _node_id,
      int _start,
      int _end,
      int _pos,
      int _depth,
      bool _is_leaf,
      double _impurity,
      double _impurity_left,
      double _impurity_right,
      double _improvement)
        : node_id(_node_id),
          start(_start),
          end(_end),
          pos(_pos),
          depth(_depth),
          is_leaf(_is_leaf),
          impurity(_impurity),
          impurity_left(_impurity_left),
          impurity_right(_impurity_right),
          improvement(_improvement){
    }
};

class TreeBuilder
{
public:
    TreeBuilder(Splitter* splitter,
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
    virtual void build(Tree* tree,
                       Mat X,
                       Mat y,
                       Mat sample_weight)=0;
public:
    Splitter* splitter;
    int min_samples_split;
    int min_samples_leaf;
    double min_weight_leaf;
    int max_depth;
    int max_leaf_nodes;

    Mat sample_weight;
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
    DepthFirstBuilder(Splitter* splitter,
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
    virtual void build(Tree* tree,
                       Mat X,
                       Mat y,
                       Mat sample_weight);

public:
    Mat sample_weight;
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
    BestFirstTreeBuilder(Splitter* splitter,
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
    virtual void build(Tree* tree,
                       Mat X,
                       Mat y,
                       Mat sample_weight);

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
    int _add_split_node(Splitter* splitter,
                        Tree* tree,
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
