"""
Compare FastRandomSurvivalForest against scikit-survival's RandomSurvivalForest
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sksurv.datasets import load_whas500
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest as SkRandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from survival_forests import FastRandomSurvivalForest

# Load and prepare data
print("Loading data...")
X, y = load_whas500()
X = OneHotEncoder().fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define common parameters for both implementations
params = {
    "n_estimators": 100,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
}

# Train and evaluate scikit-survival's RSF
print("\nTraining scikit-survival RandomSurvivalForest...")
start_time = time.time()
sk_rsf = SkRandomSurvivalForest(**params, n_jobs=-1)
sk_rsf.fit(X_train, y_train)
sk_train_time = time.time() - start_time
print(f"Training time: {sk_train_time:.2f} seconds")

# Calculate predictions and metrics for scikit-survival RSF
start_time = time.time()
sk_pred = sk_rsf.predict(X_test)
sk_pred_time = time.time() - start_time
print(f"Prediction time: {sk_pred_time:.2f} seconds")

sk_cindex = concordance_index_censored(y_test["fstat"], y_test["lenfol"], sk_pred)[0]
print(f"Concordance index: {sk_cindex:.4f}")

# Train and evaluate FastRandomSurvivalForest
print("\nTraining FastRandomSurvivalForest...")
start_time = time.time()
fast_rsf = FastRandomSurvivalForest(**params, n_jobs=-1)
fast_rsf.fit(X_train, y_train)
fast_train_time = time.time() - start_time
print(f"Training time: {fast_train_time:.2f} seconds")

# Calculate predictions and metrics for FastRandomSurvivalForest
start_time = time.time()
fast_pred = fast_rsf.predict(X_test)
fast_pred_time = time.time() - start_time
print(f"Prediction time: {fast_pred_time:.2f} seconds")

fast_cindex = concordance_index_censored(y_test["fstat"], y_test["lenfol"], fast_pred)[
    0
]
print(f"Concordance index: {fast_cindex:.4f}")

# Compare performance
print("\nPerformance comparison:")
print(f"Training speedup: {sk_train_time / fast_train_time:.2f}x")
print(f"Prediction speedup: {sk_pred_time / fast_pred_time:.2f}x")
print(f"Concordance index difference: {fast_cindex - sk_cindex:.6f}")

# Calculate correlation between predictions
corr = np.corrcoef(sk_pred, fast_pred)[0, 1]
print(f"Correlation between predictions: {corr:.6f}")

# Calculate root mean squared error between predictions
# This is just to get an idea of how numerically close the predictions are
rmse = np.sqrt(mean_squared_error(sk_pred, fast_pred))
print(f"RMSE between predictions: {rmse:.6f}")

# Plot predictions comparison
plt.figure(figsize=(10, 6))
plt.scatter(sk_pred, fast_pred, alpha=0.5)
plt.plot([min(sk_pred), max(sk_pred)], [min(sk_pred), max(sk_pred)], "r--")
plt.xlabel("scikit-survival RSF predictions")
plt.ylabel("FastRandomSurvivalForest predictions")
plt.title("Prediction Comparison")
plt.grid(alpha=0.3)
plt.savefig("prediction_comparison.png")
print("Saved prediction comparison plot to 'prediction_comparison.png'")

# Compare survival function predictions for a few samples
print("\nComparing survival function predictions...")
n_samples = 5
sk_surv_funcs = sk_rsf.predict_survival_function(X_test.iloc[:n_samples])
fast_surv_funcs = fast_rsf.predict_survival_function(X_test.iloc[:n_samples])

# Plot the survival functions
plt.figure(figsize=(14, 8))

for i in range(n_samples):
    plt.subplot(n_samples, 2, i * 2 + 1)
    plt.step(
        sk_surv_funcs[i].x, sk_surv_funcs[i].y, where="post", label=f"Sample {i+1}"
    )
    plt.ylabel("Survival Probability")
    plt.title(f"scikit-survival - Sample {i+1}")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    plt.subplot(n_samples, 2, i * 2 + 2)
    plt.step(
        fast_surv_funcs[i].x, fast_surv_funcs[i].y, where="post", label=f"Sample {i+1}"
    )
    plt.ylabel("Survival Probability")
    plt.title(f"FastRandomSurvivalForest - Sample {i+1}")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("survival_function_comparison.png")
print("Saved survival function comparison plot to 'survival_function_comparison.png'")

# Plot the survival functions for each sample in the same plot for direct comparison
plt.figure(figsize=(14, 8))

for i in range(n_samples):
    plt.subplot(2, 3, i + 1)
    plt.step(
        sk_surv_funcs[i].x, sk_surv_funcs[i].y, where="post", label="scikit-survival"
    )
    plt.step(
        fast_surv_funcs[i].x,
        fast_surv_funcs[i].y,
        where="post",
        label="FastRSF",
        alpha=0.7,
    )
    plt.ylabel("Survival Probability")
    plt.xlabel("Time")
    plt.title(f"Sample {i+1}")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("survival_function_side_by_side.png")
print(
    "Saved side-by-side survival function comparison to 'survival_function_side_by_side.png'"
)

# Memory usage comparison (approximate)
try:
    import psutil
    import os
    import sys
    import gc

    def get_size(obj, seen=None):
        """Recursively estimate size of object in bytes"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if isinstance(obj, dict):
            size += sum(get_size(v, seen) for v in obj.values())
            size += sum(get_size(k, seen) for k in obj.keys())
        elif hasattr(obj, "__dict__"):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum(get_size(i, seen) for i in obj)
            except:
                pass

        return size

    # Clear memory before measurements
    gc.collect()

    # Train smaller models for memory measurement
    print("\nTraining smaller models for memory comparison...")
    X_small, _, y_small, _ = train_test_split(X, y, test_size=0.8, random_state=42)

    process = psutil.Process(os.getpid())
    base_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Measure scikit-survival RSF memory
    gc.collect()
    sk_rsf_small = SkRandomSurvivalForest(n_estimators=20, random_state=42, n_jobs=1)
    sk_rsf_small.fit(X_small, y_small)
    sk_memory = process.memory_info().rss / 1024 / 1024 - base_memory

    # Clear and measure FastRandomSurvivalForest memory
    del sk_rsf_small
    gc.collect()
    process = psutil.Process(os.getpid())
    base_memory = process.memory_info().rss / 1024 / 1024

    fast_rsf_small = FastRandomSurvivalForest(
        n_estimators=20, random_state=42, n_jobs=1
    )
    fast_rsf_small.fit(X_small, y_small)
    fast_memory = process.memory_info().rss / 1024 / 1024 - base_memory

    print(f"\nApproximate memory usage:")
    print(f"scikit-survival RSF: {sk_memory:.2f} MB")
    print(f"FastRandomSurvivalForest: {fast_memory:.2f} MB")
    print(f"Memory reduction: {(1 - fast_memory / sk_memory) * 100:.2f}%")

    has_memory_metrics = True
except ImportError:
    print("\nCouldn't import psutil, skipping memory usage comparison.")
    has_memory_metrics = False

# Print summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(
    f"Training speed: {'FASTER' if fast_train_time < sk_train_time else 'SLOWER'} by {abs(sk_train_time / fast_train_time - 1) * 100:.2f}%"
)
print(
    f"Prediction speed: {'FASTER' if fast_pred_time < sk_pred_time else 'SLOWER'} by {abs(sk_pred_time / fast_pred_time - 1) * 100:.2f}%"
)
print(
    f"Concordance index: {'BETTER' if fast_cindex > sk_cindex else 'WORSE'} by {abs(fast_cindex - sk_cindex):.6f}"
)
print(f"Prediction correlation: {corr:.6f}")
if has_memory_metrics:
    print(
        f"Memory usage: {'LESS' if fast_memory < sk_memory else 'MORE'} by {abs(1 - fast_memory / sk_memory) * 100:.2f}%"
    )
print("=" * 50)
