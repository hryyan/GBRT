#include "basetree.h"
#include <algorithm>
#include "criterion.h"
#include "splitter.h"

Tree::Tree(int n_features,
           int n_classes)
    : _n_features(n_features),  // Input/Output layout
      _n_classes(n_classes),    // Inner structures
      _max_depth(0),
      _node_count(0),
      _capacity(0)
{

}

Tree::~Tree()
{

}

int Tree::_add_node(int parent,
                    bool is_left,
                    bool is_leaf,
                    int feature,
                    double threshold,
                    double impurity,
                    int n_node_samples,
                    double weighted_n_node_samples)
{
    int node_id = _node_count;

    // Insure _nodes has enough elements
    Node n = Node();
    _nodes.push_back(n);

    Node* node = &(_nodes.at(node_id));
    node->impurity = impurity;
    node->n_node_samples = n_node_samples;
    node->weighted_n_node_samples = weighted_n_node_samples;

    if (parent != TREE_UNDEFINED)
    {
        if (is_left)
        {
            _nodes.at(parent).left_child = node_id;
        }
        else
        {
            _nodes.at(parent).right_child = node_id;
        }
    }

    if (is_leaf)
    {
        node->left_child = TREE_LEAF;
        node->right_child = TREE_LEAF;
        node->feature = TREE_UNDEFINED;
        node->threshold = TREE_UNDEFINED;
    }
    else
    {
        // left_child and right_child will be set later
        node->feature = feature;
        node->threshold = threshold;
    }

    _node_count += 1;
    return node_id;
}

Mat Tree::predict(Mat _X)
{
    Mat out = _apply_dense(_X);
//    for (int i = 0; i < out.total(); i++)
//        result.at<double>(i, 0) = _value.at(out.at<int>(i, 0));
    return out;
}

Mat Tree::_apply_dense(Mat _X)
{
    Node* node;
    int drop = 0;
    int n_samples = _X.rows;
    Mat_<double> result(n_samples, 1);

    for (int i = 0; i < n_samples; i++)
    {
        drop = 0;
        node = &(_nodes.at(drop));

        // While node is not a leaf
        while (node->left_child != TREE_LEAF)
        {
            // and node.right_child != TreeType::TREE_LEAF
            if (_X.at<double>(i, node->feature) <= node->threshold)
            {
                drop = node->left_child;
                node = &(_nodes.at(node->left_child));
            }
            else
            {
                drop = node->right_child;
                node = &(_nodes.at(node->right_child));
            }
        }
        int offset = drop;
//        vector<Node>::iterator it = std::find(_nodes.begin(), _nodes.end(), *node);
//        int offset = (int)(it - _nodes.begin());
        vector<double>::iterator c = max_element(_value.at(offset).begin(), _value.at(offset).end());
        int n = distance(_value.at(offset).begin(), c);
        result.at<double>(i, 0) = static_cast<double>(distance(_value.at(offset).begin(), c));
    }
    return result;
}

Mat Tree::compute_feature_importances(bool normalize)
{
    Mat result = Mat::zeros(_n_features, 1, CV_64F);
    Node left, right;

    for (vector<Node>::iterator it = _nodes.begin(); it != _nodes.end(); it++)
    {
        if (it->left_child != TREE_LEAF)
        {
            left = _nodes.at(it->left_child);
            right = _nodes.at(it->right_child);

            result.at<double>(it->feature, 0) += (
                        it->weighted_n_node_samples * it->impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity);
        }
    }

    result /= _nodes.at(0).weighted_n_node_samples;

    if (normalize)
    {
        double normalizer = cv::sum(result)[0];

        if (normalizer > 0.0)
            for (int i = 0; i != result.total(); i++)
                result.at<double>(i, 0) = result.at<double>(i, 0) / normalizer;
    }
    return result;
}































