import math
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession


class DecisionTreeRegressor:
    def __init__(self, data, max_depth = 5, min_instances_per_node = 1):
        self.max_depth = max_depth
        self.min_instances_per_node = min_instances_per_node
        self.tree = self.__build_tree(data, 0)

    def __build_tree(self, data, depth):
        # Data is empty or reach max depth -> leaf node
        n_samples = data.count()
        if n_samples == 0 or depth >= self.max_depth:
            return self.__compute_leaf_value(data)

        # Check if variance is zero -> all labels identical -> leaf node
        labels = data.map(lambda x: x[1])
        y_mean = labels.mean()
        y_var = labels.map(lambda y: (y - y_mean) ** 2).mean()
        if y_var == 0:
            return y_mean

        # Find the best split
        best_split = None
        best_variance = float('inf')
        n_features = len(data.first()[0])

        # Loop thorugh all features to find the best split
        for feature_index in range(n_features):
            feature_label_pairs = [(row[0][feature_index], row[1]) for row in data.collect()]
            feature_label_pairs.sort(key=lambda x: x[0])
            feature_values = [x[0] for x in feature_label_pairs]
            labels_sorted = [x[1] for x in feature_label_pairs]

            # Compute cumulative sums for efficient variance calculation
            cum_sum, cum_sum_sq = 0.0, 0.0
            left_sums, left_sum_sqs = [], []
            for label in labels_sorted:
                cum_sum += label
                cum_sum_sq += label ** 2
                left_sums.append(cum_sum)
                left_sum_sqs.append(cum_sum_sq)
            total_sum = cum_sum
            total_sum_sq = cum_sum_sq

            # Evaluate all possible split points
            for i in range(1, n_samples):
                if feature_values[i] == feature_values[i - 1]:
                    continue  # Skip duplicate feature values
                left_count = i
                right_count = n_samples - i
                if left_count < self.min_instances_per_node or right_count < self.min_instances_per_node:
                    continue

                left_sum = left_sums[i - 1]
                left_sum_sq = left_sum_sqs[i - 1]
                right_sum = total_sum - left_sum
                right_sum_sq = total_sum_sq - left_sum_sq

                # Compute variances for left and right splits
                left_var = (left_sum_sq - (left_sum ** 2) / left_count) / left_count if left_count > 0 else 0
                right_var = (right_sum_sq - (right_sum ** 2) / right_count) / right_count if right_count > 0 else 0
                weighted_variance = (left_var * left_count + right_var * right_count) / n_samples

                if weighted_variance < best_variance:
                    best_variance = weighted_variance
                    threshold = (feature_values[i - 1] + feature_values[i]) / 2  # Midpoint as threshold
                    best_split = (feature_index, threshold)

        # No split is found -> leaf value
        if best_split is None:
            return self.__compute_leaf_value(data)

        print(f"Best split found at feature {best_split[0]} with threshold {best_split[1]}")

        # Split the full dataset using the best split from the sample
        feature_index, threshold = best_split
        left_data = data.filter(lambda x: x[0][feature_index] <= threshold).cache()
        right_data = data.filter(lambda x: x[0][feature_index] > threshold).cache()

        # Recursively build subtrees
        left_tree = self.__build_tree(left_data, depth + 1)
        right_tree = self.__build_tree(right_data, depth + 1)

        # Clean up cached RDDs
        left_data.unpersist()
        right_data.unpersist()

        # Return internal node structure
        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def __compute_leaf_value(self, data_rdd):
        labels = data_rdd.map(lambda x: x[1])
        return labels.mean() if data_rdd.count() > 0 else 0.0

    def __traverse_tree(self, tree, features):
        if isinstance(tree, dict):
            feature_index = tree["feature_index"]
            threshold = tree["threshold"]
            if features[feature_index] <= threshold:
                return self.__traverse_tree(tree["left"], features)
            else:
                return self.__traverse_tree(tree["right"], features)
        return tree

    def predict(self, features):
        return self.__traverse_tree(self.tree, features)


# Parse datetime into features
def parse_datetime(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return [dt.hour, dt.weekday(), dt.month, dt.timetuple().tm_yday]


# Compute Haversine distance
def compute_distance(long_1, lat_1, long_2, lat_2):
    long_diff = math.radians(long_1 - long_2) / 2
    lat_diff = math.radians(lat_1 - lat_2) / 2
    a = math.sin(lat_diff) ** 2 + \
        math.cos(math.radians(lat_1)) * \
        math.cos(math.radians(lat_2)) * \
        math.sin(long_diff) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


# Parse raw row into features and target
def parse_train(row):
    cols = row.split(",")
    passenger_count = float(cols[4])
    long_lats = [float(col) for col in cols[5:9]]
    trip_duration = float(cols[10])
    pickup_datetime = parse_datetime(cols[2])
    trip_distance = compute_distance(*long_lats)
    return [*pickup_datetime, passenger_count, *long_lats, trip_distance, trip_duration]


# Optimized outlier removal
def remove_outliers(data, column, lower_quantile=0.25, upper_quantile=0.75, k=1.5):
    values = data.map(lambda row: row[column]).collect()
    lower_quantile_value = np.percentile(values, lower_quantile * 100)
    upper_quantile_value = np.percentile(values, upper_quantile * 100)
    iqr = upper_quantile_value - lower_quantile_value
    lower_bound = lower_quantile_value - k * iqr
    upper_bound = upper_quantile_value + k * iqr
    return data.filter(lambda row: lower_bound <= row[column] <= upper_bound)


def main():
    spark = SparkSession.builder \
        .appName("DecisionTreeRegressor") \
        .master("local[*]") \
        .config("spark.logConfig", "false") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    raw_data = sc.textFile("train.csv")
    header = raw_data.first()
    train_data = raw_data.filter(lambda row: row != header).map(parse_train)
    train_data = remove_outliers(train_data, 9)  # distance
    train_data = remove_outliers(train_data, 10)  # duration

    train_rdd, test_rdd = train_data.randomSplit([0.8, 0.2], seed=42)
    train_rdd = train_rdd.map(lambda row: (row[:-1], row[-1])).cache()
    test_rdd = test_rdd.map(lambda row: (row[:-1], row[-1])).cache()

    model = DecisionTreeRegressor(train_rdd, max_depth=5, min_instances_per_node=5)
    print("Decision Tree Regressor model built successfully.")

    sample_test_cases = test_rdd.take(10)
    print("Sample test cases:")
    for row in sample_test_cases:
        prediction = model.predict(row[0])
        print(f"Features: {', '.join(map(str, row[0]))}, Actual: {row[1]}, Prediction: {prediction}")

    labels_and_predictions = test_rdd.map(lambda row: (row[1], model.predict(row[0])))

    test_rmse = math.sqrt(labels_and_predictions.map(lambda lp: (lp[0] - lp[1]) ** 2).mean())
    print(f"Test Root Mean Squared Error (RMSE) = {test_rmse}")

    test_mae = labels_and_predictions.map(lambda lp: abs(lp[0] - lp[1])).mean()
    print(f"Test Mean Absolute Error (MAE) = {test_mae}")

    y_mean = labels_and_predictions.map(lambda lp: lp[0]).mean()
    ss_res = labels_and_predictions.map(lambda lp: (lp[0] - lp[1]) ** 2).sum()
    ss_tot = labels_and_predictions.map(lambda lp: (lp[0] - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"Test R^2 = {r2}")

    train_rdd.unpersist()
    test_rdd.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
