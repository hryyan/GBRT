#include "splitter.h"
#include <algorithm>

void SplitRecord::init_split(int start_pos)
{
    impurity_left = INFINITY;
    impurity_right = INFINITY;
    pos = start_pos;
    feature = 0;
    threshold = 0;
    improvement = -INFINITY;
}

Splitter::Splitter(Criterion* _criterion,
                   int _max_feature,
                   int _min_samples_leaf,
                   double _min_weight_leaf,
                   int _random_state)
    : criterion(_criterion),
      max_features(_max_feature),
      min_samples_leaf(_min_samples_leaf),
      min_weight_leaf(_min_weight_leaf),
      random_state(_random_state),
      n_samples(0),
      n_features(0),
      weighted_n_samples(0.0),
      start(0),
      end(0)
{

}

Splitter::~Splitter()
{

}

int Splitter::init(Mat_<double> _X,
                    Mat_<double> _y,
                    Mat_<double> _sample_weight)
{
    // Init some value
    n_samples = _X.rows;
    n_features = _X.cols;

    weighted_n_samples = 0.0;

    // Validation
    // _X.rows == _y.rows == _y.total
    // _y.rows == _samples_weight.rows == _samples_weight.total
    if (_X.rows != _y.rows)
        return 1;
    if (_y.rows != _y.total())
        return 2;
    if (_y.rows != _sample_weight.rows)
        return 3;
    if (_sample_weight.rows != _sample_weight.total())
        return 4;

    // Calculate the weight sum
    int j = 0;
    for (int i = 0; i < n_samples; i++)
    {
        if (_sample_weight.total() != 0 || _sample_weight.at<double>(i) != 0.0)
        {
            samples.push_back(i);
            j += 1;
        }

        if (_sample_weight.total() != 0)
            weighted_n_samples += _sample_weight.at<double>(i);
        else
            weighted_n_samples += 1.0;
    }
    n_samples = j;

    // Store all feature index
    for (int i = 0; i < n_features; i++)
        features.push_back(i);

    // Store the constant feature index
    constant_features.resize(n_features);

    // Init the size of feature_values
    feature_values.resize(n_samples);

    // Store the data
    X = _X;
    y = _y;
    sample_weight = _sample_weight;
}

double Splitter::node_reset(int _start, int _end)
{
    start = _start;
    end = _end;

    criterion->init(y,
                   sample_weight,
                   weighted_n_samples,
                   samples,
                   start,
                   end);

    weighted_n_samples = criterion->weighted_n_node_samples;
    return weighted_n_samples;
}

BaseDenseSplitter::BaseDenseSplitter(Criterion* criterion,
                                     int max_feature,
                                     int min_samples_leaf,
                                     double min_weight_leaf,
                                     int random_state)
    : Splitter(criterion,
               max_feature,
               min_samples_leaf,
               min_weight_leaf,
               random_state)
{

}

BaseDenseSplitter::~BaseDenseSplitter()
{

}

int BaseDenseSplitter::init(Mat_<double> _X,
                             Mat_<double> _y,
                             Mat_<double> _sample_weight)
{
    Splitter::init(_X, _y, _sample_weight);
}

BestSplitter::BestSplitter(Criterion* criterion,
                           int max_features,
                           int min_samples_leaf,
                           double min_weight_leaf,
                           int random_state)
    : BaseDenseSplitter(criterion,
                        max_features,
                        min_samples_leaf,
                        min_weight_leaf,
                        random_state)
{

}

BestSplitter::~BestSplitter()
{

}

void BestSplitter::node_split(double impurity,
                         SplitRecord *split,
                         int* n_constant_features)
{
    split->init_split(end);

    std::pair<double, double> pdd;

    SplitRecord best, current;

    int p;
    int tmp;
    int partition_end;
    int n_visited_features = 0;
    // Num of features discovered to be constant during the split search
    int n_found_constants = 0;
    // Num of features known to be constant and drawn without replacement
    int n_drawn_constants = 0;
    int n_known_constants = *n_constant_features;
    // n_total_constants = n_known_constants + n_found_constants
    int n_total_constants = n_known_constants;

    /**
      * Sample up to max_features without replacement using a
      * Fisher-Yates-based algorithm (using the local variables 'f_i' and
      * 'f_j' to compute a permutation of the 'features' array.
      *
      * Skip the CPU intensive evaluation of the impurity criterion for
      * features that were already detected as constant (hence not suitable
      * for good splitting) by ancestor nodes and save the information on
      * newly discovered constant features to spare computation on descendant
      * node.
      */
    int f_i = n_features;
    int f_j = 0;
    while (f_i > n_total_constants && // Stop early if remaining features
                                      // are constant
           (n_visited_features < max_features ||
            // At least one drawn features must be non constant
            n_visited_features <= n_found_constants + n_drawn_constants))
    {
        n_visited_features += 1;

        /**
          * Loop invariant: elements of features in
          * - [0:n_drawn_constants] holds drawn and known constant features;
          * - [n_drawn_constants:n_known_constants] holds known constant
          *   features that haven't been drawn yet;
          * - [n_known_constants:n_total_constants] holds newly found constant
          *   features;
          * - [n_total_constants:f_i] holds features that haven't been drawn
          *   yet and aren't constant apriori;
          * - [f_i:n_features] holds features that have been drawn and aren't
          *   constant.
          */

        // Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state);

        if (f_j < n_known_constants) // in the interval [n_drawn_constasn, n_known_constants]
        {
            // swap features[f_j] and features[n_drawn_constants]
            // move the constant feature to the end
            tmp = features[f_j];
            features[f_j] = features[n_drawn_constants];
            features[n_drawn_constants] = tmp;

            n_drawn_constants += 1;
        }
        else
        {
            // f_j in the interval [n_known_constants, f_i-n_found_constatns]
            f_j += n_found_constants;
            // f_j in the interval [n_total_constants, f_i]

            current.feature = features[f_j];

            /**
              * Sort sampels along that feature; first copy the feature
              * values for the active samples into feature_values, s.t.
              * feature_values[i] == X[sampels[i], j], so the sort uses the cache more
              * effectively.
              */
            for (int i = start; i < end; i++)
            {
                feature_values.at(i) = X.at<double>(samples[i], current.feature);
            }

            // sort feature_values and apply the squence to samples
            // std::sort(feature_values.begin(), feature_values.end());
            auto sequence = sort_permutation(feature_values,
                                [](double const& a, double const &b){return a<b;});
            feature_values = apply_permutation(feature_values, sequence);
            samples = apply_permutation(samples, sequence);

            if (feature_values.back() <= feature_values.front() + FEATURE_THRESHOLD)
            {
                // The feature is constant
                // Move it to the features[n_total_constants]
                features[f_j] = features[n_total_constants];
                features[n_total_constants] = current.feature;

                n_found_constants += 1;
                n_total_constants += 1;
            }
            else
            {
                // The feature is good
                f_i -= 1;
                tmp = features[f_i];
                features[f_i] = features[f_j];
                features[f_j] = tmp;

                // Evaluate all splits
                criterion->reset();
                p = start;

                while (p < end)
                {
                    while (p + 1 < end && \
                           feature_values.at(p+1) <= feature_values.at(p) + FEATURE_THRESHOLD)
                        p += 1;

                    p += 1;

                    if (p < end)
                    {
                        current.pos = p;

                        // Reject if min_samples_leaf is not guaranteed
                        if (((current.pos - start) < min_samples_leaf) ||
                             ((end - current.pos) < min_samples_leaf))
                            continue;

                        criterion->update(current.pos);

                        // Reject if min_weight_leaf is not satisfied
                        if ((criterion->weighted_n_left < min_weight_leaf) ||
                                criterion->weighted_n_right < min_weight_leaf)
                            continue;

                        double a = feature_values.at(p);
                        current.improvement = criterion->impurity_improvement(impurity);

                        if (current.improvement > best.improvement)
                        {
                            pdd = criterion->children_impurity();
                            current.impurity_left = pdd.first;
                            current.impurity_right = pdd.second;
                            current.threshold = (feature_values.at(p-1) + feature_values.at(p)) / 2.0;

                            if (current.threshold == feature_values.at(p))
                                current.threshold = feature_values.at(p-1);

                            best = current;
                        }
                    }
                }
            }
        }
    }

    // Recoganize into samples[start:best.pos] + samples[best.pos:end]
    if (best.pos < end)
    {
        partition_end = end;
        p = start;

        while (p < partition_end)
        {
            if (X.at<double>(samples[p], best.feature) <= best.threshold)
                p += 1;
            else
            {
                partition_end -= 1;

                tmp = samples[partition_end];
                samples[partition_end] = samples[p];
                samples[p] = tmp;
            }
        }
    }

    // Respect invariant for constant features: the original order of
    // element in features[:n_known_constants] must be preserved for sibling
    // and child nodes
    for (int i = 0; i < n_known_constants; i++)
        features.at(i) = constant_features.at(i);

    // Copy newly found constant features
    for (int i = n_known_constants; i < n_known_constants+n_found_constants; i++)
        constant_features.at(i) = features.at(i);

    // Return values
    split[0] = best;
    n_constant_features[0] = n_total_constants;
}

RandomSplitter::RandomSplitter(Criterion* _criterion,
                               int _max_features,
                               int _min_samples_leaf,
                               double _min_weight_leaf,
                               int _random_state)
    : BaseDenseSplitter(_criterion,
                        _max_features,
                        _min_samples_leaf,
                        _min_weight_leaf,
                        _random_state)
{

}

RandomSplitter::~RandomSplitter()
{

}

void RandomSplitter::node_split(double impurity,
                                SplitRecord *split,
                                int *n_constant_features)
{
    split->init_split(end);

    feature_values.clear();
    std::pair<double, double> pdd;

    SplitRecord best, current;

    double min_feature_value;
    double max_feature_value;
    double current_feature_value;
    int p;
    int tmp;
    int partition_end;
    int n_visited_features = 0;
    // Num of features discovered to be constant during the split search
    int n_found_constants = 0;
    // Num of features known to be constant and drawn without replacement
    int n_drawn_constants = 0;
    int n_known_constants = *n_constant_features;
    // n_total_constants = n_known_constants + n_found_constants
    int n_total_constants = n_known_constants;

    /**
      * Sample up to max_features without replacement using a
      * Fisher-Yates-based algorithm (using the local variables 'f_i' and
      * 'f_j' to compute a permutation of the 'features' array.
      *
      * Skip the CPU intensive evaluation of the impurity criterion for
      * features that were already detected as constant (hence not suitable
      * for good splitting) by ancestor nodes and save the information on
      * newly discovered constant features to spare computation on descendant
      * node.
      */
    int f_i = n_features;
    int f_j = 0;
    while (f_i > n_total_constants && // Stop early if remaining features
                                      // are constant
           (n_visited_features < max_features ||
            // At least one drawn features must be non constant
            n_visited_features <= n_found_constants + n_drawn_constants))
    {
        n_visited_features += 1;

        /**
          * Loop invariant: elements of features in
          * - [0:n_drawn_constants] holds drawn and known constant features;
          * - [n_drawn_constants:n_known_constants] holds known constant
          *   features that haven't been drawn yet;
          * - [n_known_constants:n_total_constants] holds newly found constant
          *   features;
          * - [n_total_constants:f_i] holds features that haven't been drawn
          *   yet and aren't constant apriori;
          * - [f_i:n_features] holds features that have been drawn and aren't
          *   constant.
          */

        // Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state);

        if (f_j < n_known_constants) // in the interval [n_drawn_constasn, n_known_constants]
        {
            // swap features[f_j] and features[n_drawn_constants]
            // move the constant feature to the end
            tmp = features[f_j];
            features[f_j] = features[n_drawn_constants];
            features[n_drawn_constants] = tmp;

            n_drawn_constants += 1;
        }
        else
        {
            // f_j in the interval [n_known_constants, f_i - n_found_constants]
            f_j += n_found_constants;
            // f_j in the interval [n_total_constatns, f_i]

            current.feature = features[f_j];

            // Find min, max
            // This is faster than sort
            min_feature_value = X.at<double>(samples[start], current.feature);
            max_feature_value = min_feature_value;
            feature_values.at(start) = min_feature_value;

            for (int i = start+1; i < end; i++)
            {
                current_feature_value = X.at<double>(samples[i], current.feature);
                feature_values.at(i) = current_feature_value;

                if (current_feature_value < min_feature_value)
                    min_feature_value = current_feature_value;
                else if (current_feature_value > max_feature_value)
                    max_feature_value = current_feature_value;
            }

            if (max_feature_value <= min_feature_value + FEATURE_THRESHOLD)
            {
                features.at(f_j) = features[n_total_constants];
                features.at(n_total_constants) = current.feature;

                n_found_constants += 1;
                n_total_constants += 1;
            }
            else
            {
                f_i -= 1;
                tmp = features.at(f_j);
                features.at(f_j) = features.at(f_i);
                features.at(f_i) = tmp;

                // Draw a random threshold
                current.threshold = rand_double(min_feature_value,
                                                max_feature_value,
                                                random_state);

                if (current.threshold == max_feature_value)
                    current.threshold = min_feature_value;

                // Partition
                partition_end = end;
                p = start;
                while (p < partition_end)
                {
                    current_feature_value = feature_values.at(p);
                    if (current_feature_value <= current.threshold)
                        p += 1;
                    else
                    {
                        partition_end -= 1;

                        feature_values.at(p) = feature_values.at(partition_end);
                        feature_values.at(partition_end) = current_feature_value;

                        tmp = samples.at(partition_end);
                        samples.at(partition_end) = samples.at(p);
                        samples.at(p) = tmp;
                    }
                }
                current.pos = partition_end;

                // Reject if min_samples_leaf is not guaranteed
                if (((current.pos - start) < min_samples_leaf) ||
                        ((end - current.pos) < min_samples_leaf))
                    continue;

                // Evaluate split
                criterion->reset();
                criterion->update(current.pos);

                // Reject if min_weight_leaf is not satisfied
                if ((criterion->weighted_n_left < min_weight_leaf) ||
                        criterion->weighted_n_right < min_weight_leaf)
                    continue;

                current.improvement = criterion->impurity_improvement(impurity);

                if (current.improvement > best.improvement)
                {
                    pdd = criterion->children_impurity();
                    current.impurity_left = pdd.first;
                    current.impurity_right = pdd.second;
                    best = current;
                }
            }
        }
    }

    // Recoganize into samples[start:best.pos] + samples[best.pos:end]
    if (best.pos < end)
    {
        partition_end = end;
        p = start;

        while (p < partition_end)
        {
            if (X.at<double>(samples[p], best.feature) <= best.threshold)
                p += 1;
            else
            {
                partition_end -= 1;

                tmp = samples[partition_end];
                samples[partition_end] = samples[p];
                samples[p] = tmp;
            }
        }
    }

    // Respect invariant for constant features: the original order of
    // element in features[:n_known_constants] must be preserved for sibling
    // and child nodes
    for (int i = 0; i < n_known_constants; i++)
        features.at(i) = constant_features.at(i);

    // Copy newly found constant features
    for (int i = n_known_constants; i < n_known_constants+n_found_constants; i++)
        constant_features.at(i) = features.at(i);

    // Return values
    split[0] = best;
    n_constant_features[0] = n_total_constants;
}

PresortBestSplitter::PresortBestSplitter(Criterion* _criterion,
                                         int _max_features,
                                         int _min_samples_leaf,
                                         double _min_weight_leaf,
                                         int _random_state)
    : BaseDenseSplitter(_criterion,
                        _max_features,
                        _min_samples_leaf,
                        _min_weight_leaf,
                        _random_state)
{

}

PresortBestSplitter::~PresortBestSplitter()
{

}

int PresortBestSplitter::init(Mat_<double> _X,
                               Mat_<double> _y,
                               Mat_<double> _sample_weight)
{
    // Call parent initializer
    BaseDenseSplitter::init(_X, _y, _sample_weight);

    X = _X;

    // Pre-sort X
//    if (X_old == X)
    if (true)
    {
        X_old = X.clone();
        n_total_samples = X.rows;
        cv::sortIdx(_X, X_argsorted, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
        sample_mask.resize(n_total_samples);
    }
}

void PresortBestSplitter::node_split(double impurity,
                                     SplitRecord *split,
                                     int *n_constant_features)
{
    split->init_split(end);

    feature_values.clear();
    std::pair<double, double> pdd;

    SplitRecord best, current;

    int partition_end = 0;
    int p = 0;
    int tmp;
    int n_visited_features = 0;
    // Num of features discovered to be constant during the split search
    int n_found_constants = 0;
    // Num of features known to be constant and drawn without replacement
    int n_drawn_constants = 0;
    int n_known_constants = *n_constant_features;
    // n_total_constants = n_known_constants + n_found_constants
    int n_total_constants = n_known_constants;

    for (p = start; p < end; p++)
    {
        sample_mask.at(samples[p]) = 1;
    }

    /**
      * Sample up to max_features without replacement using a
      * Fisher-Yates-based algorithm (using the local variables 'f_i' and
      * 'f_j' to compute a permutation of the 'features' array.
      *
      * Skip the CPU intensive evaluation of the impurity criterion for
      * features that were already detected as constant (hence not suitable
      * for good splitting) by ancestor nodes and save the information on
      * newly discovered constant features to spare computation on descendant
      * node.
      */
    int f_i = n_features;
    int f_j = 0;
    while (f_i > n_total_samples && // Stop early if remaining features
                                     // are constant
           (n_visited_features < max_features ||
            // At least one drawn features must be non constant)
            n_visited_features <= n_found_constants + n_drawn_constants))
    {
        n_visited_features += 1;

        /**
          * Loop invariant: elements of features in
          * - [0:n_drawn_constants] holds drawn and known constant features;
          * - [n_drawn_constants:n_known_constants] holds known constant
          *   features that haven't been drawn yet;
          * - [n_known_constants:n_total_constants] holds newly found constant
          *   features;
          * - [n_total_constants:f_i] holds features that haven't been drawn
          *   yet and aren't constant apriori;
          * - [f_i:n_features] holds features that have been drawn and aren't
          *   constant.
          */

        // Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state);

        if (f_j < n_known_constants) // in the interval [n_drawn_constasn, n_known_constants]
        {
            // swap features[f_j] and features[n_drawn_constants]
            // move the constant feature to the end
            tmp = features[f_j];
            features[f_j] = features[n_drawn_constants];
            features[n_drawn_constants] = tmp;

            n_drawn_constants += 1;
        }
        else
        {
            // f_j in the interval [n_known_constants, f_i-n_found_constatns]
            f_j += n_found_constants;
            // f_j in the interval [n_total_constants, f_i]

            current.feature = features[f_j];

            // Extract ordering from X_argsorted
            p = start;

            for (int i = start, j = 0; i < n_total_samples; i++)
            {
                j = X_argsorted.at<int>(current.feature, i);
                if (sample_mask[j] == 1)
                {
                    samples[p] = j;
                    feature_values.at(p) = X.at<double>(j, current.feature);
                    p += 1;
                }
            }

            // Evaluate all splits
            if (feature_values.at(end-1) <= feature_values.at(start) + FEATURE_THRESHOLD)
            {
                features[f_j] = features[n_total_constants];
                features[n_total_constants] = current.feature;

                n_found_constants += 1;
                n_total_constants += 1;
            }
            else
            {
                f_i -= 1;
                tmp = features[f_i];
                features[f_i] = features[f_j];
                features[f_j] = tmp;

                criterion->reset();

                while (p < end)
                {
                    while (p + 1 < end &&
                           feature_values.at(p+1) <= feature_values.at(p) + FEATURE_THRESHOLD)
                        p += 1;

                    p += 1;

                    if (p < end)
                    {
                        current.pos = p;

                        // Reject if min_samples_leaf is not guaranteed
                        if (((current.pos - start) < min_samples_leaf) ||
                             ((end - current.pos) < min_samples_leaf))
                            continue;

                        criterion->update(current.pos);

                        // Reject if min_weight_leaf is not satisfied
                        if ((criterion->weighted_n_left < min_weight_leaf) ||
                                (criterion->weighted_n_right < min_weight_leaf))
                            continue;

                        current.improvement = criterion->impurity_improvement(impurity);

                        if (current.improvement > best.improvement)
                        {
                            pdd = criterion->children_impurity();
                            current.impurity_left = pdd.first;
                            current.impurity_right = pdd.second;
                            current.threshold = (feature_values.at(p-1) + feature_values.at(p)) / 2.0;

                            if (current.threshold == feature_values.at(p))
                                current.threshold = feature_values.at(p-1);

                            best = current;
                        }
                    }
                }
            }
        }
    }

    // Recoganize into samples[start:best.pos] + samples[best.pos:end]
    if (best.pos < end)
    {
        partition_end = end;
        p = start;

        while (p < partition_end)
        {
            if (X.at<double>(samples[p], best.feature) <= best.threshold)
                p += 1;
            else
            {
                partition_end -= 1;

                tmp = samples[partition_end];
                samples[partition_end] = samples[p];
                samples[p] = tmp;
            }
        }
    }

    // Respect invariant for constant features: the original order of
    // element in features[:n_known_constants] must be preserved for sibling
    // and child nodes
    for (int i = 0; i < n_known_constants; i++)
        features.at(i) = constant_features.at(i);

    // Copy newly found constant features
    for (int i = n_known_constants; i < n_known_constants+n_found_constants; i++)
        constant_features.at(i) = features.at(i);

    // Return values
    split[0] = best;
    n_constant_features[0] = n_total_constants;
}





































