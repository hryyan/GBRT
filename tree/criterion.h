#ifndef CRITERION_H
#define CRITERION_H

//========================================
//Criterion
//author: vincent yan, 2015-04-27
//========================================
#define DEBUG
#ifdef DEBUG
#include <iostream>
using namespace std;
#endif

#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
using std::pair;
using std::vector;
using cv::Mat;
using cv::Mat_;

class Criterion
{
public:
    Criterion();
    virtual ~Criterion();

    /**
     * @brief Initialize the criterion at node samples[start:end] and children samples[start:start] and samples[start:end]
     * @param y: y's value or label
     * @param sample_weight: weight of sample
     * @param weight_n_samples: sum(wi) for i in sample.size
     * @param samples: sample index
     * @param start:
     * @param end:
     */
    virtual void init(Mat y,
                      Mat sample_weight,
                      double weight_n_samples,
                      vector<int>& samples,
                      int start,
                      int end)=0;

    /**
     * @brief Reset the criterion at pos=start
     */
    virtual void reset()=0;

    /**
     * @brief Update the collected statistics by moving samples[pos:new_pos] from the right child to the left child
     * @param new_pos
     */
    virtual void update(int new_pos)=0;

    /**
     * @brief Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
     */
    virtual double node_impurity()=0;

    /**
     * @brief Evaluate the impurity in children nodes, i.e. the impurity of samples[start:pos] + the impurity of samples[pos:end].
     * @return pair<impurity_left, impurity_right>
     */
    virtual std::pair<double, double> children_impurity()=0;

    /**
     * @brief Compute the node value of samples[start:end]
     * @return
     */
    virtual vector<double> node_value()=0;

    /**
     * @brief Weighted impurity improvement, i.e.
     *
     *     N_t / N * (impurity - N_t_L / N_t * left impurity
     *                         - N_t_R / N_t * right impurity),
     *
     *     where N is the total number of samples, N_t is the number of samples
     *     in the current node, N_t_L is the number of samples in the left
     *     child and N_t_R is the number of samples in the right child
     * @return impurity_improvement
     */
    double impurity_improvement(double impurity);

public:
    Mat_<double> y;                 // Values of y
    Mat_<double> sample_weight;     // Sample weights

    vector<int> samples;            // Sample indice in X, y
    int start;                      // samples[start:pos] are the samples in the left node
    int pos;                        // samples[pos:end] are the samples in the right node
    int end;

    int n_node_samples;             // Number of samples in the node (end-start)
    double weighted_n_samples;      // Weighted number of samples (in total)
    double weighted_n_node_samples; // Weighted number of samples in the node
    double weighted_n_left;         // Weighted number of samples in the left node
    double weighted_n_right;        // Weighted number of samples in the right node

    vector<double> label_count_left;
    vector<double> label_count_right;
    vector<double> label_count_total;
};

class ClassificationCriterion : public Criterion
{
public:
    /**
     * @brief Abstract criterion for classification
     */
    ClassificationCriterion();
    virtual ~ClassificationCriterion();

    /**
     * @brief Initialize the criterion at node samples[start:end] and children samples[start:start] and samples[start:end]
     * @param y: y's value or label
     * @param sample_weight: weight of sample
     * @param weight_n_samples: sum(wi) for i in sample.size
     * @param samples: sample index
     * @param start:
     * @param end:
     */
    virtual void init(Mat y,
                      Mat sample_weight,
                      double weight_n_samples,
                      vector<int>& samples,
                      int start,
                      int end);

    /**
     * @brief Reset the criterion at pos=start
     */
    virtual void reset();

    /**
     * @brief Update the collected statistics by moving samples[pos:new_pos] from the right child to the left child
     * @param new_pos
     */
    virtual void update(int new_pos);

    /**
     * @brief Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
     */
    virtual double node_impurity()=0;

    /**
     * @brief Evaluate the impurity in children nodes, i.e. the impurity of samples[start:pos] + the impurity of samples[pos:end].
     * @return pair<impurity_left, impurity_right>
     */
    virtual std::pair<double, double> children_impurity()=0;

    /**
     * @brief Compute the node value of samples[start:end]
     * @return
     */
    virtual vector<double> node_value();

public:
    int n_classes;
};

class Entropy : public ClassificationCriterion
{
public:
    /**
     * @brief Entropy::Cross Entropy impurity criteria.
     *
     *  Let the target be a classification outcome taking values in 0, 1, ..., K-1.
     *  If node m represents a region Rm with Nm observations, then let
     *
     *       pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)
     *
     *     be the proportion of class k observations in node m.
     *
     *     The cross-entropy is then defined as
     *
     *       cross-entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
     */
    Entropy();
    virtual ~Entropy();

    /**
     * @brief Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
     */
    virtual double node_impurity();

    /**
     * @brief Evaluate the impurity in children nodes, i.e. the impurity of samples[start:pos] + the impurity of samples[pos:end].
     * @return pair<impurity_left, impurity_right>
     */
    virtual std::pair<double, double> children_impurity();
};

class Gini : public ClassificationCriterion
{
public:
    /**
     * @brief Gini::Gini Index impurity criteria.
     *
     *     Let the target be a classification outcome taking values in 0, 1, ..., K-1.
     *     If node m represents a region Rm with Nm observations, then let
     *
     *         pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)
     *
     *     be the proportion of class k observations in node m.
     *
     *     The Gini Index is then defined as:
     *
     *         index = \sum_{k=0}^{K-1} pmk (1 - pmk)
     *               = 1 - \sum_{k=0}^{K-1} pmk ** 2
     */
    Gini();
    virtual ~Gini();

    /**
     * @brief Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
     */
    virtual double node_impurity();

    /**
     * @brief Evaluate the impurity in children nodes, i.e. the impurity of samples[start:pos] + the impurity of samples[pos:end].
     * @return pair<impurity_left, impurity_right>
     */
    virtual std::pair<double, double> children_impurity();
};

class RegressionCriterion : public Criterion
{
public:
    /** Abstract criterion for regression.
     *
     * Computes variance of the target values left and right of the split point.
     * Computation is linear in `n_samples` by using ::
     *
     *     var = \sum_i^n (y_i - y_bar) ** 2
     *         = (\sum_i^n y_i ** 2) - n_samples y_bar ** 2
     */
    RegressionCriterion();
    virtual ~RegressionCriterion();

    /**
     * @brief Initialize the criterion at node samples[start:end] and children samples[start:start] and samples[start:end]
     * @param y: y's value or label
     * @param sample_weight: weight of sample
     * @param weight_n_samples: sum(wi) for i in sample.size
     * @param samples: sample index
     * @param start:
     * @param end:
     */
    virtual void init(Mat y,
                      Mat sample_weight,
                      double weight_n_samples,
                      vector<int>& samples,
                      int start,
                      int end);

    /**
     * @brief Reset the criterion at pos=start
     */
    virtual void reset();

    /**
     * @brief Update the collected statistics by moving samples[pos:new_pos] from the right child to the left child
     * @param new_pos
     */
    virtual void update(int new_pos);

    /**
     * @brief Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
     */
    virtual double node_impurity()=0;

    /**
     * @brief Evaluate the impurity in children nodes, i.e. the impurity of samples[start:pos] + the impurity of samples[pos:end].
     * @return pair<impurity_left, impurity_right>
     */
    virtual std::pair<double, double> children_impurity()=0;

    /**
     * @brief Compute the node value of samples[start:end]
     * @return
     */
    virtual vector<double> node_value();

public:
    double mean_left;
    double mean_right;
    double mean_total;
    double sq_sum_left;
    double sq_sum_right;
    double sq_sum_total;
    double var_left;
    double var_right;
    double sum_left;
    double sum_right;
    double sum_total;
};

class MSE : public RegressionCriterion
{
public:
    /** Mean squared error impurity criterion.
     * MSE = var_left + vaar_right
     */
    MSE();
    virtual ~MSE();

    /**
     * @brief Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
     */
    virtual double node_impurity();

    /**
     * @brief Evaluate the impurity in children nodes, i.e. the impurity of samples[start:pos] + the impurity of samples[pos:end].
     * @return pair<impurity_left, impurity_right>
     */
    virtual std::pair<double, double> children_impurity();
};

class FriedmanMSE : public MSE
{
public:
    /**
     * Mean squared error impurity criterion with improvement score by Friedman
     * Uses the formula (35) in Friedmans original Gradient Boosting paper:
     *     diff = mean_left - mean_right
     *     improvement = n_left * n_right * diff^2 / (n_left + n_right)
     */
    FriedmanMSE();
    virtual ~FriedmanMSE();

    virtual double impurity_improvement(double impurity);
};

#endif // CRITERION_H
