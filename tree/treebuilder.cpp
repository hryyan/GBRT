#include "treebuilder.h"
#include "criterion.h"
#include "splitter.h"
#include "basetree.h"
#include "tree.h"
#include <stack>
#include <queue>
using std::stack;
using std::priority_queue;

TreeBuilder::TreeBuilder(Splitter* _splitter,
                         int _min_samples_split,
                         int _min_samples_leaf,
                         double _min_weight_leaf,
                         int _max_depth,
                         int _max_leaf_nodes)
    : splitter(_splitter),
      min_samples_split(_min_samples_split),
      min_samples_leaf(_min_samples_leaf),
      min_weight_leaf(_min_weight_leaf),
      max_depth(_max_depth),
      max_leaf_nodes(_max_leaf_nodes)
{

}

TreeBuilder::~TreeBuilder()
{

}

DepthFirstBuilder::DepthFirstBuilder(Splitter* _splitter,
                                     int _min_samples_split,
                                     int _min_samples_leaf,
                                     double _min_weight_leaf,
                                     int _max_depth,
                                     int _max_leaf_nodes)
    : TreeBuilder(_splitter,
                  _min_samples_split,
                  _min_samples_leaf,
                  _min_weight_leaf,
                  _max_depth,
                  _max_leaf_nodes)
{

}


DepthFirstBuilder::~DepthFirstBuilder()
{

}

void DepthFirstBuilder::build(Tree* _tree,
                              Mat _X,
                              Mat _y,
                              Mat _sample_weight)
{
    if (_sample_weight.total() != 0)
        sample_weight = _sample_weight;

    splitter->init(_X, _y, _sample_weight);

    int n_node_samples = splitter->n_samples;
    double weighted_n_node_samples = splitter->weighted_n_samples;
    bool is_leaf;
    SplitRecord split;
    int node_id;
    int max_depth_seen = -1;

    int start;
    int end;
    int depth;
    int parent;
    bool is_left;
    double impurity;
    int n_constant_features;

    bool first = true;

    stack<N> stk;
    // Push root node onto stack
    stk.push(N(0, n_node_samples, 0, TREE_UNDEFINED, 0, INFINITY, 0));

    while (!stk.empty())
    {
        N n = stk.top();
        stk.pop();
        start = n.start;
        end = n.end;
        depth = n.depth;
        parent = n.parent;
        is_left = n.is_left;
        impurity = n.impurity;
        n_constant_features = n.n_constant_features;

        n_node_samples = end - start;
        weighted_n_node_samples = splitter->node_reset(start, end);

        is_leaf = ((n.depth >= max_depth) ||
                   (n_node_samples < min_samples_split) ||
                   (n_node_samples < 2 * min_samples_leaf) ||
                   (weighted_n_node_samples < min_weight_leaf));

        if (first)
        {
            impurity = splitter->node_impurity();
            first = false;
        }

        is_leaf = is_leaf || (impurity <= MIN_IMPURITY_SPLIT);

        if (!is_leaf)
        {
            splitter->node_split(impurity, &split, &n_constant_features);
            is_leaf = is_leaf || (split.pos >= end);
        }

        node_id = _tree->_add_node(parent, is_left, is_leaf, split.feature,
                                  split.threshold, impurity, n_node_samples,
                                  weighted_n_node_samples);

        if (is_leaf)
        {
            // Don't store value for internal nodes
            if (_tree->_value.size() < node_id+1)
                _tree->_value.resize(node_id+1);
            _tree->_value.at(node_id) = splitter->node_value();
        }
        else
        {
            // Push right child on stack
            stk.push(N(split.pos+start, end, depth+1, node_id, 0,
                       split.impurity_right, n_constant_features));
            stk.push(N(start, split.pos+start, depth+1, node_id, 1,
                       split.impurity_left, n_constant_features));
        }
        if (depth > max_depth)
            max_depth_seen = depth;
    }
}

BestFirstTreeBuilder::BestFirstTreeBuilder(Splitter* _splitter,
                                           int _min_samples_split,
                                           int _min_samples_leaf,
                                           double _min_weight_leaf,
                                           int _max_depth,
                                           int _max_leaf_nodes)
    : TreeBuilder(_splitter,
                  _min_samples_split,
                  _min_samples_leaf,
                  _min_weight_leaf,
                  _max_depth,
                  _max_leaf_nodes)
{

}

BestFirstTreeBuilder::~BestFirstTreeBuilder()
{

}

void BestFirstTreeBuilder::build(Tree* _tree,
                                 Mat _X,
                                 Mat _y,
                                 Mat _sample_weight)
{
    if (_sample_weight.total() != 0)
        sample_weight = _sample_weight;

    splitter->init(_X, _y, _sample_weight);

    int n_node_samples = splitter->n_samples;
    double weighted_n_node_samples = splitter->weighted_n_samples;
    bool is_leaf;
    SplitRecord split;
    int node_id;
    int max_depth_seen = -1;

    int start;
    int end;
    int depth;
    int parent;
    bool is_left;
    double impurity;
    int n_constant_features;

    bool first = true;

    priority_queue<P> pq;
    P record, split_node_left, split_node_right;
    // Push root to frontier
    _add_split_node(splitter,
                    _tree,
                    0,
                    n_node_samples,
                    INFINITY,
                    1,
                    1,
                    -1,
                    0,
                    &split_node_left);

    _add_to_frontier(&split_node_left, pq);

    while (!pq.empty())
    {
        record = pq.pop();

        node =
    }
}

int BestFirstTreeBuilder::_add_split_node(Splitter* _splitter,
                                          Tree* _tree,
                                          int _start,
                                          int _end,
                                          double _impurity,
                                          bool _is_first,
                                          bool _is_left,
                                          int _parent,
                                          int _depth,
                                          P *res)
{
    SplitRecord split;
    int n_constant_features = 0;
    int node_id = 0;
    bool is_leaf = false;
    double weighted_n_node_samples = _splitter->node_reset(_start, _end);

    if (_is_first)
        _impurity = _splitter->node_impurity();

    is_leaf = ((_depth > max_depth) ||
               (n_node_samples < min_samples_split) ||
               (n_node_samples < 2 * min_samples_leaf) ||
               (impurity <= MIN_IMPURITY_SPLIT));

    if (!is_leaf)
    {
        _splitter->node_split(_impurity, split, &n_constant_features);
        is_leaf = is_leaf || (split.pos >= end);
    }

    if (_parent == -1)
        _parent = TREE_UNDEFINED;
    node_id = _tree->_add_node(_parent,
                               _is_left,
                               is_leaf,
                               split.feature,
                               split.threshold,
                               _impurity,
                               n_node_samples,
                               weighted_n_node_samples);

    _tree->_value.at(node_id) = splitter->node_value();

    res->_node_id = node_id;
    res->_start = _start;
    res->_end;
    res->_depth;
    res->_impurity;

    if (!is_leaf)
    {
        res->_pos = split.pos + _start;
        res->_is_leaf = 0;
        res->_improvement = split.improvement;
        res->_impurity_left = split.impurity_left;
        res->_impurity_right = split.impurity_right;
    }
    else
    {
        res->_pos = _end;
        res->_is_leaf = 1;
        res->_improvement = 0.0;
        res->_impurity_left = _impurity;
        res->_impurity_right = _impurity;
    }
    return 0;
}
