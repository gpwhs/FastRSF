from sksurv.datasets import load_whas500
from sksurv.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from survival_forests import FastRandomSurvivalForest

# Load and prepare data
X, y = load_whas500()
X = OneHotEncoder().fit_transform(X)

# Create and fit the fast random survival forest
rsf = FastRandomSurvivalForest(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,  # Use all available cores
    random_state=42,
)
rsf.fit(X, y)

# Predict survival functions for first 5 samples
surv_funcs = rsf.predict_survival_function(X.iloc[:5])

# Plot the predicted survival functions
plt.figure(figsize=(10, 6))
for i, fn in enumerate(surv_funcs):
    plt.step(fn.x, fn(fn.x), where="post", label=f"Sample {i+1}")

plt.ylabel("Survival Probability")
plt.xlabel("Time")
plt.ylim(0, 1)
plt.legend()
plt.title("Predicted Survival Functions")
plt.grid(alpha=0.3)
plt.show()

# Predict risk scores
risk_scores = rsf.predict(X.iloc[:10])
print("Risk scores (higher = higher risk):", risk_scores)

# You can also get the cumulative hazard function
chf_funcs = rsf.predict_cumulative_hazard_function(X.iloc[:5])
