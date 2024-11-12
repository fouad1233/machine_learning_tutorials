#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>

using namespace std;

// TreeNode class
class TreeNode {
public:
    TreeNode *left;
    TreeNode *right;
    int value;
    double threshold;
    int feature_index;
    bool is_leaf;

    // Constructor
    TreeNode(TreeNode* left = nullptr, TreeNode* right = nullptr, int value = -1,
             double threshold = 0.0, int feature_index = -1, bool is_leaf = false)
        : left(left), right(right), value(value), threshold(threshold),
          feature_index(feature_index), is_leaf(is_leaf) {}
};

// DecisionTree class
class DecisionTree {
public:
    TreeNode* tree;
    int max_depth;

    DecisionTree(int max_depth = 10) : tree(nullptr), max_depth(max_depth) {}

    // Entropy calculation
    double entropy(const vector<int>& y) {
        map<int, int> class_counts;
        for (int cls : y) class_counts[cls]++;

        double entropy = 0.0;
        int n = y.size();
        for (auto& pair : class_counts) {
            double p = (double)pair.second / n;
            if (p != 0) entropy -= p * log2(p);
        }
        return entropy;
    }

    // Gini impurity calculation
    double gini(const vector<int>& y) {
        map<int, int> class_counts;
        for (int cls : y) class_counts[cls]++;

        double gini = 1.0;
        int n = y.size();
        for (auto& pair : class_counts) {
            double p = (double)pair.second / n;
            gini -= p * p;
        }
        return gini;
    }

    // Split function
    void split(const vector<vector<double>>& X, const vector<int>& y, int feature_index, double threshold,
               vector<vector<double>>& X_left, vector<vector<double>>& X_right, vector<int>& y_left, vector<int>& y_right) {
        for (size_t i = 0; i < X.size(); i++) {
            if (X[i][feature_index] <= threshold) {
                X_left.push_back(X[i]);
                y_left.push_back(y[i]);
            } else {
                X_right.push_back(X[i]);
                y_right.push_back(y[i]);
            }
        }
    }

    // Find best split
    pair<int, double> best_split(const vector<vector<double>>& X, const vector<int>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();
        double best_gini = numeric_limits<double>::infinity();
        int best_feature = -1;
        double best_threshold = 0.0;

        for (int feature_index = 0; feature_index < n_features; feature_index++) {
            vector<double> thresholds;
            for (const auto& row : X) thresholds.push_back(row[feature_index]);
            sort(thresholds.begin(), thresholds.end());
            thresholds.erase(unique(thresholds.begin(), thresholds.end()), thresholds.end());

            for (size_t i = 0; i < thresholds.size(); i++) {
                double threshold = thresholds[i];

                vector<vector<double>> X_left, X_right;
                vector<int> y_left, y_right;
                split(X, y, feature_index, threshold, X_left, X_right, y_left, y_right);

                if (y_left.empty() || y_right.empty()) continue;

                double gini_left = gini(y_left);
                double gini_right = gini(y_right);
                double weighted_gini = (y_left.size() * gini_left + y_right.size() * gini_right) / n_samples;

                if (weighted_gini < best_gini) {
                    best_gini = weighted_gini;
                    best_feature = feature_index;
                    best_threshold = threshold;
                }
            }
        }

        return {best_feature, best_threshold};
    }

    // Build tree recursively
    TreeNode* build_tree(const vector<vector<double>>& X, const vector<int>& y, int depth) {
        if (y.empty()) return nullptr;

        map<int, int> class_counts;
        for (int cls : y) class_counts[cls]++;
        int predicted_class = max_element(class_counts.begin(), class_counts.end(),
            [](const pair<int, int>& a, const pair<int, int>& b) { return a.second < b.second; })->first;

        TreeNode* node = new TreeNode(nullptr, nullptr, predicted_class);

        if (depth == 0 || gini(y) == 0.0) {
            node->is_leaf = true;
            return node;
        }

        auto [feature_index, threshold] = best_split(X, y);

        if (feature_index == -1) {
            node->is_leaf = true;
            return node;
        }

        vector<vector<double>> X_left, X_right;
        vector<int> y_left, y_right;
        split(X, y, feature_index, threshold, X_left, X_right, y_left, y_right);

        node->feature_index = feature_index;
        node->threshold = threshold;
        node->left = build_tree(X_left, y_left, depth - 1);
        node->right = build_tree(X_right, y_right, depth - 1);

        return node;
    }

    // Fit the model
    void fit(const vector<vector<double>>& X, const vector<int>& y, int max_depth = 10) {
        this->max_depth = max_depth;
        tree = build_tree(X, y, max_depth);
    }

    // Predict a single instance
    int _predict(TreeNode* node, const vector<double>& x) {
        if (node->is_leaf) return node->value;
        if (x[node->feature_index] <= node->threshold)
            return _predict(node->left, x);
        else
            return _predict(node->right, x);
    }

    // Predict multiple instances
    vector<int> predict(const vector<vector<double>>& X) {
        vector<int> predictions;
        for (const auto& x : X) predictions.push_back(_predict(tree, x));
        return predictions;
    }
};



int main() {
    DecisionTree dt(3);
    vector<vector<double>> X = {{2.5}, {1.0}, {3.0}, {1.5}};
    vector<int> y = {0, 1, 0, 1};
    dt.fit(X, y);

    vector<vector<double>> test = {{2.0}, {1.0}};
    vector<int> predictions = dt.predict(test);

    for (auto p : predictions) cout << "Prediction: " << p << endl;

    return 0;
}
