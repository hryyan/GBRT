#include "criterion.h"

Criterion::Criterion()
    : start(0),
      pos(0),
      end(0),
      n_node_samples(0),
      weighted_n_samples(0.0),
      weighted_n_node_samples(0.0),
      weighted_n_left(0.0),
      weighted_n_right(0.0)
{

}

Criterion::~Criterion()
{

}

double Criterion::impurity_improvement(double impurity)
{
    double impurity_left, impurity_right;
    pair<double, double> p = children_impurity();

    impurity_left = p.first;
    impurity_right = p.second;

    return (weighted_n_node_samples / weighted_n_samples) *
           (impurity - weighted_n_right / weighted_n_node_samples * impurity_right
                     - weighted_n_left / weighted_n_node_samples * impurity_left);
}

ClassificationCriterion::ClassificationCriterion()
    : Criterion(),
      n_classes(0)
{

}

ClassificationCriterion::~ClassificationCriterion()
{

}

void ClassificationCriterion::init(Mat _y,
                                   Mat _sample_weight,
                                   double _weight_n_samples,
                                   vector<int>& _samples,
                                   int _start,
                                   int _end)
{
    y = _y;
    sample_weight = _sample_weight;
    weighted_n_samples = _weight_n_samples;
    samples = _samples;
    start = _start;
    end = _end;

    // Find how many classes in y
    std::set<double> unique;
    for (int i = 0; i < y.rows; i++)
    {
        if (unique.find(y.at<double>(i)) == unique.end())
            unique.insert(y.at<double>(i));
    }
    n_classes = unique.size();

    // Initialize
    label_count_total.resize(n_classes);
    label_count_left.resize(n_classes);
    label_count_right.resize(n_classes);

    double _weighted_n_node_samples = 0.0;
    double w = 1.0;
    int index;
    for (int i = start; i < end; i++)
    {
        index = samples.at(i);

        if (sample_weight.total() == 0)
        {
            w = sample_weight(index);
        }

        // Get count of every class
        int c = (int)y.at<double>(index, 0);
        label_count_total.at(c) += w;

        _weighted_n_node_samples += w;
    }
    weighted_n_node_samples = _weighted_n_node_samples;
    reset();
}

void ClassificationCriterion::reset()
{
    pos = start;

    weighted_n_left = 0.0;
    weighted_n_right = weighted_n_node_samples;

    for (int i = 0; i < n_classes; i++)
    {
        label_count_left.at(i) = 0.0;
        label_count_right.at(i) = label_count_total.at(i);
    }
}

void ClassificationCriterion::update(int new_pos)
{
    int index;
    double w = 1.0;
    double diff_w = 0.0;
    for (int i = pos; i < new_pos; i++)
    {
        index = samples.at(i);

        if (sample_weight.total() != 0)
        {
            w = sample_weight.at<double>(index);
        }

        int label_index = (int)y.at<double>(index);
        label_count_left.at(label_index) += w;
        label_count_right.at(label_index) -= w;

        diff_w += w;
    }
    weighted_n_left += diff_w;
    weighted_n_right -= diff_w;

    pos = new_pos;
}

vector<double> ClassificationCriterion::node_value()
{
    return label_count_total;
}

Entropy::Entropy()
    :ClassificationCriterion()
{

}

Entropy::~Entropy()
{

}

double Entropy::node_impurity()
{
    double total = 0.0;
    double tmp = 0.0;
    double entropy = 0.0;
    for (int i = 0; i < n_classes; i++)
    {
        tmp = label_count_total.at(i);
        if (tmp > 0.0)
        {
            tmp /= weighted_n_node_samples;
            entropy -= tmp * log(tmp);
        }
        total += entropy;
    }
    return total;
}

std::pair<double, double> Entropy::children_impurity()
{
    double entropy_left = 0.0;
    double entropy_right = 0.0;
    double total_left = 0.0;
    double total_right = 0.0;
    double tmp = 0.0;

    for (int i = 0; i < n_classes; i++)
    {
        tmp = label_count_left.at(i);
        if (tmp > 0.0)
        {
            tmp /= weighted_n_left;
            entropy_left -= tmp * log(tmp);
        }

        tmp = label_count_right.at(i);
        if (tmp > 0.0)
        {
            tmp /= weighted_n_right;
            entropy_right -= tmp * log(tmp);
        }
    }
    total_left += entropy_left;
    total_right += entropy_right;

    return std::make_pair(total_left, total_right);
}

Gini::Gini()
    :ClassificationCriterion()
{

}

Gini::~Gini()
{

}

double Gini::node_impurity()
{
    double gini = 0.0;
    double tmp = 0.0;

    for (int i = 0; i < n_classes; i++)
    {
        tmp = label_count_total[i];
        gini += tmp * tmp;
    }
    gini = 1.0 - gini / (weighted_n_node_samples *
                         weighted_n_node_samples);
    return gini;
}

std::pair<double, double> Gini::children_impurity()
{
    double gini_left = 0.0;
    double gini_right = 0.0;
    double tmp = 0.0;

    for (int i = 0; i < n_classes; i++)
    {
        tmp = label_count_left[i];
        gini_left += tmp * tmp;
        tmp = label_count_right[i];
        gini_right += tmp * tmp;
    }
    gini_left = 1.0 - gini_left / (weighted_n_left *
                                   weighted_n_left);
    gini_right = 1.0 - gini_right / (weighted_n_right *
                                     weighted_n_right);

    return std::make_pair(gini_left, gini_right);
}

RegressionCriterion::RegressionCriterion()
    : Criterion(),
      mean_left(0.0),
      mean_right(0.0),
      mean_total(0.0),
      sq_sum_left(0.0),
      sq_sum_right(0.0),
      sq_sum_total(0.0),
      var_left(0.0),
      var_right(0.0),
      sum_left(0.0),
      sum_right(0.0),
      sum_total(0.0)
{

}


RegressionCriterion::~RegressionCriterion()
{

}

void RegressionCriterion::init(Mat _y,
                               Mat _sample_weight,
                               double _weight_n_samples,
                               vector<int>& _samples,
                               int _start,
                               int _end)
{
    y = _y;
    sample_weight = _sample_weight;
    weighted_n_samples = _weight_n_samples;
    samples = _samples;
    start = _start;
    end = _end;

    int index;
    double w = 1.0;
    double y_i = 0.0;
    double w_y_i = 0.0;

    for (int i = start; i < end; i++)
    {
        index = samples.at(i);

        if (sample_weight.total() != 0)
            w = sample_weight.at<double>(index);

        y_i = y.at<double>(index);
        w_y_i = w * y_i;
        sum_total += w_y_i;
        sq_sum_total += w_y_i * y_i;

        weighted_n_node_samples += w;
    }
    mean_total = sum_total / weighted_n_node_samples;

    reset();
}

void RegressionCriterion::reset()
{
    mean_right = mean_total;
    mean_left = 0.0;
    sq_sum_right = sq_sum_total;
    sq_sum_left = 0.0;
    var_right = (sq_sum_right / weighted_n_node_samples -
                 mean_right * mean_right);
    var_left = 0.0;
    sum_right = sum_total;
    sum_left = 0.0;

    weighted_n_right = weighted_n_node_samples;
    weighted_n_left = 0;
}

void RegressionCriterion::update(int new_pos)
{
    double w = 1.0;
    double y_i = 0.0;
    double w_y_i = 0.0;
    double diff_w = 0.0;

    int index = 0;

    for (int i = pos; i < new_pos; i++)
    {
        index = samples.at(i);

        if (sample_weight.total() != 0)
            w  = sample_weight.at<double>(index);

        y_i = y.at<double>(index);
        w_y_i = w * y_i;

        sum_left += w_y_i;
        sum_right -= w_y_i;

        sq_sum_left += w_y_i * y_i;
        sq_sum_right -= w_y_i * y_i;

        diff_w += w;
    }
    weighted_n_left += diff_w;
    weighted_n_right -= diff_w;

    mean_left = sum_left / weighted_n_left;
    mean_right = sum_right / weighted_n_right;
    var_left = (sq_sum_left / weighted_n_left -
                mean_left * mean_left);
    var_right = (sq_sum_right / weighted_n_right -
                 mean_right * mean_right);

    pos = new_pos;
}

vector<double> RegressionCriterion::node_value()
{
    vector<double> vec;
    vec.push_back(mean_total);
    return vec;
}

MSE::MSE()
    : RegressionCriterion()
{

}

MSE::~MSE()
{

}

double MSE::node_impurity()
{
    return (sq_sum_total / weighted_n_node_samples -
            mean_total * mean_total);
}

pair<double, double> MSE::children_impurity()
{
    return std::make_pair(var_left, var_right);
}

FriedmanMSE::FriedmanMSE()
    : MSE()
{

}

FriedmanMSE::~FriedmanMSE()
{

}

double FriedmanMSE::impurity_improvement(double impurity)
{
    double total_sum_left = 0.0;
    double total_sum_right = 0.0;
    double diff = 0.0;

    total_sum_left += sum_left;
    total_sum_right += sum_right;
    diff = (total_sum_left / weighted_n_left) -
           (total_sum_right / weighted_n_right);

    return weighted_n_left * weighted_n_right * diff * diff /
            (weighted_n_left + weighted_n_right);
}
























