#include "basetree.h"
#include <algorithm>

/**
 * @brief Tree::
 */
Tree::Tree(int _n_features,
           int _n_classes)
{
    // Input/Output layout
    n_features = _n_features;
    n_classes = _n_classes;

    // Inner structures
    max_depth = 0;
    node_count = 0;
    capacity = 0;
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
    int node_id = node_count;

    Node* node = &nodes.at(node_id);
    node->impurity = impurity;
    node->n_node_samples = n_node_samples;
    node->weighted_n_node_samples = weighted_n_node_samples;

    if (parent != TREE_UNDEFINED)
    {
        if (is_left)
        {
            nodes.at(parent).left_child = node_id;
        }
        else
        {
            nodes.at(parent).right_child = node_id;
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

    node_count += 1;
    return node_id;
}

Mat_<double> Tree::predict(Mat_<double> _X)
{
    Mat out = _apply_dense(_X);
    Mat_<double> result(_X.rows, 1);
    for (int i = 0; i < out.total(); i++)
        result.at<double>(i, 0) = value.at(out.at<int>(i, 0));
    return result;
}

Mat_<double> Tree::_apply_dense(Mat_<double> _X)
{
    Node* node;
    int n_samples = _X.rows;
    Mat result = Mat(n_samples, 1, CV_32F);

    for (int i = 0; i < n_samples; i++)
    {
        node = &nodes.at(0);

        // While node is not a leaf
        while (node->left_child != TREE_LEAF)
        {
            // and node.right_child != TreeType::TREE_LEAF
            if (_X.at<double>(i, node->feature) <= node->threshold)
            {
                node = &(nodes.at(node->left_child));
            }
            else
            {
                node = &(nodes.at(node->right_child));
            }
        }
        std::vector<Node>::iterator it = std::find(nodes.begin(), nodes.end(), *node);
        result.at<double>(i, 0) = (int)(it - nodes.begin());
    }
    return result;
}

Mat_<double> Tree::compute_feature_importances(bool normalize)
{
    Mat result = Mat::zeros(n_features, 1, CV_32F);
    Node left, right;

    for (vector<Node>::iterator it = nodes.begin(); it != nodes.end(); it++)
    {
        if (it->left_child != TREE_LEAF)
        {
            left = nodes.at(it->left_child);
            right = nodes.at(it->right_child);

            result.at<double>(it->feature, 0) += (
                        it->weighted_n_node_samples * it->impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity);
        }
    }

    result /= nodes.at(0).weighted_n_node_samples;

    if (normalize)
    {
        double normalizer = cv::sum(result)[0];

        if (normalizer > 0.0)
            for (int i = 0; i != result.total(); i++)
                result.at<double>(i, 0) = result.at<double>(i, 0) / normalizer;
    }
    return result;
}































