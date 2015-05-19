#include "treebuilder.h"
#include "criterion.h"
#include "splitter.h"
#include "basetree.h"
#include "tree.h"

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
                              Mat_<double> _X,
                              Mat_<double> _y,
                              Mat_<double> _sample_weight)
{
    if (_sample_weight.total() != 0)
        sample_weight = _sample_weight;

    splitter->init(_X, _y, _sample_weight);

    int n_node_samples = splitter->n_samples;
    bool is_leaf;
    double weighted_n_node_samples = splitter->weighted_n_samples;
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

    std::stack<N> stk;
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
            is_leaf = is_leaf or (split.pos >= end);
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
//            splitter->node_value().at(_tree->_value.at(node_id));
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
                                 Mat_<double> _X,
                                 Mat_<double> _y,
                                 Mat_<double> _sample_weight)
{
    if (_sample_weight.total() != 0)
        sample_weight = _sample_weight;

    int n_node_samples;
    bool is_leaf;
    double weighted_n_node_samples;
//    SplitterRecord split;
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

//    X = _X;
//    y = _y;
//    sample_weight = _sample_weight;
//    splitter.init(_X, y, sample_weight);

//    std::priority_queue q;

}

int BestFirstTreeBuilder::_add_split_node(Splitter* splitter,
                                          Tree* tree,
                                          int start,
                                          int end,
                                          double impurity,
                                          bool is_first,
                                          bool is_left,
                                          Node *parent,
                                          int depth,
                                          N *res)
{
//    SplitRecord split;
//    int node_id;
//    int n_node_samples;
//    int n_constant_feature = 0;
//    double weighted_n_samples = splitter.weighted_n_samples;
//    double weighted_n_node_samples;
//    bool is_leaf;
//    int n_left, n_right;
//    double imp_diff;

//    weighted_n_node_samples = splitter.node_reset(start, end);

//    if (is_first)
//        impurity = splitter.node_impurity();

//    n_node_samples = end - start;
//    is_leaf = ((depth > max_depth) ||
//               (n_node_samples < min_samples_split) ||
//               (n_node_samples < 2 * min_samples_leaf) ||
//               (weighted_n_node_samples < min_weight_leaf) ||
//               (impurity <= MIN_IMPURITY_SPLIT));

//    if (!is_leaf)
//    {
//        splitter.node_split(impurity, &split, &n_constant_feature);
//        is_leaf = is_leaf || (split.pos >= end);
//    }

//    bool on;
//    if (parent != NULL)
//        on = true;
//    else
//        on = false;
//    node_id = tree._add_node(parent - tree.nodes,
//                             on,
//                             is_left,
//                             is_leaf,
//                             split.feature,
//                             split.threshold,
//                             impurity,
//                             n_node_samples,
//                             weighted_n_node_samples);

//    if (node_id == -1)
//        return -1;

//    // Compute values also for split nodes (might become leafs later).
//    splitter.node_value(tree.value.at(node_id));

//    res.node_id = node_id;
//    res.start = start;
//    res.end = end;
//    res.depth = depth;
//    res.impurity = impurity;

//    if (!is_leaf)
//    {
//        // is split node
//        res.pos = split.pos;
//        res.is_leaf = 0;
//        res.improvement = split.improvement;
//        res.impurity_left = split.impurity_left;
//        res.imputity_right = split.impurity_right;
//    }
//    else
//    {
//        // is leaf => 0 improment
//        res.pos = end;
//        res.is_leaf = 1;
//        res.improvement = 0.0;
//        res.impurity_left = impurity;
//        res.impurity_right = impurity;
//    }
    return 0;
}























