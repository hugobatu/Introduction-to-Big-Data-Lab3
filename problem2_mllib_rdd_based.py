import math
import numpy as np
from datetime import datetime
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import Vectors

# Spark context
sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")

# Load the dataset
try:
    data = sc.textFile("train.csv")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sc.stop()
    exit(1)

# Parse the dataset
def parse_datetime(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return [dt.hour, dt.weekday(), dt.month, dt.timetuple().tm_yday] 

def compute_distance(long_1, lat_1, long_2, lat_2):
    long_diff = math.radians(long_1 - long_2) / 2
    lat_diff = math.radians(lat_1 - lat_2) / 2
    a = math.sin(lat_diff) ** 2 + \
        math.cos(math.radians(lat_1)) * \
        math.cos(math.radians(lat_2)) * \
        math.sin(long_diff) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))

def parse(row):
    cols = row.split(",")
    passenger_count = float(cols[4])
    long_lats = [float(col) for col in cols[5:9]]
    trip_duration = float(cols[10])
    pickup_datetime = parse_datetime(cols[2])
    trip_distance = compute_distance(*long_lats)
    # Features: hour, weekday, month, day_of_year, passenger_count, pickup_long, pickup_lat, dropoff_long, dropoff_lat, distance
    # Target: trip_duration
    return [*pickup_datetime, passenger_count, *long_lats, trip_distance, trip_duration]

header = data.first()
data = data.filter(lambda row: row != header)
data = data.map(parse)

# Initial filtering
data = data.filter(lambda row: row[9] > 0)  # distance > 0
data = data.filter(lambda row: row[10] > 0)  # duration > 0
data = data.filter(lambda row: row[4] > 0)  # passenger_count > 0

# Preprocess the data
def remove_outliers(data, column, lower_quantile=0.25, upper_quantile=0.75, k=1.5):
    values = data.map(lambda row: row[column]).collect()
    lower_quantile_value = np.percentile(values, lower_quantile * 100)
    upper_quantile_value = np.percentile(values, upper_quantile * 100)
    iqr = upper_quantile_value - lower_quantile_value
    lower_bound = lower_quantile_value - k * iqr
    upper_bound = upper_quantile_value + k * iqr
    return data.filter(lambda row: lower_bound <= row[column] <= upper_bound)

data = remove_outliers(data, 9)  # distance
data = remove_outliers(data, 10)  # duration

# Add log-transformed target
data = data.map(lambda row: (*row[:-1], np.log(row[-1]), row[-1]))  # (features, log_duration, original_duration)

# Convert to (LabeledPoint, original_duration) tuple and split train/val
data = data.map(lambda row: (LabeledPoint(row[-2], row[:-2]), row[-1]))
train_data, val_data = data.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = DecisionTree.trainRegressor(
    train_data.map(lambda x: x[0]),  # Extract LabeledPoint for training
    categoricalFeaturesInfo={},
    maxDepth=10,
)

# Evaluate the model
def evaluate_model(model, data, dataset_name):
    # Extract features, log labels, and original labels
    features = data.map(lambda x: x[0].features)
    log_labels = data.map(lambda x: x[0].label)  # log(trip_duration)
    original_labels = data.map(lambda x: x[1])   # original trip_duration

    # Get predictions in log space
    log_predictions = model.predict(features)

    # Compute metrics in log scale
    log_pred_and_labels = log_predictions.zip(log_labels).map(lambda x: (float(x[0]), float(x[1])))
    log_metrics = RegressionMetrics(log_pred_and_labels)

    # Exponentiate predictions to original scale
    predictions = log_predictions.map(lambda x: math.exp(x))
    pred_and_labels = predictions.zip(original_labels).map(lambda x: (float(x[0]), float(x[1])))
    metrics = RegressionMetrics(pred_and_labels)

    # Print metrics
    print(f"\n=== {dataset_name} Metrics ===")
    print("Log Scale:")
    print(f"RMSE: {log_metrics.rootMeanSquaredError}")
    print("Original Scale:")
    print(f"RMSE: {metrics.rootMeanSquaredError}")
    print(f"MSE: {metrics.meanSquaredError}")
    print(f"MAE: {metrics.meanAbsoluteError}")
    print(f"R2: {metrics.r2}")

evaluate_model(model, train_data, "Training Data")
evaluate_model(model, val_data, "Validation Data")

sc.stop()
