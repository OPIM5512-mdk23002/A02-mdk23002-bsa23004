from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Setting up train/test split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)

#Checking train/test split
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Separate features and target using the existing train_df/test_df
X_train = train_df.drop(columns=["MedHouseVal"])
y_train = train_df["MedHouseVal"]

X_test = test_df.drop(columns=["MedHouseVal"])
y_test = test_df["MedHouseVal"]

# Scale features (recommended for MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the model (required: early_stopping=True + at least one custom hyperparameter)
mlp = MLPRegressor(
    random_state=42,
    early_stopping=True,        # required
    hidden_layer_sizes=(10, 5), # custom hyperparameter (counts)
    max_iter=500,
    batch_size=1000,
    activation="relu",
    validation_fraction=0.2
)

mlp.fit(X_train_scaled, y_train)

#Adding training predictions
train_preds = mlp.predict(X_train_scaled)

#training performance metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
def metrics_row(name, y_train, train_preds):
    return {
        "split": name,
        "R2": r2_score(y_train, train_preds),
        "MAE": mean_absolute_error(y_train, train_preds),
        "MAPE": mean_absolute_percentage_error(y_train, train_preds),
    }

metrics_df = pd.DataFrame([metrics_row("train", y_train, train_preds)])

print("Training Performance Metrics:")
print(metrics_df.to_string(index=False))

#Plotting model performance on training data
plt.figure(figsize=(8, 8))
plt.scatter(y_train, train_preds, alpha=0.3)
low = min(np.min(y_train), np.min(train_preds))
high = max(np.max(y_train), np.max(train_preds))
plt.plot([low, high], [low, high], color="red", linestyle="--") #ref line
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("MLP Regressor: Actual vs. Predicted MedHouseVal on Training Data")

#saving plot
plt.savefig("figures/train_actual_vs_pred.png")
plt.close()
print("Training performance plot saved as 'figures/train_actual_vs_pred.png'")

# Test predictions
test_preds = mlp.predict(X_test_scaled)

# Plotting model performance on test data
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_preds, alpha=0.3)

low = min(np.min(y_test), np.min(test_preds))
high = max(np.max(y_test), np.max(test_preds))
plt.plot([low, high], [low, high], color="red", linestyle="--")  # ref line

plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("MLP Regressor: Actual vs. Predicted MedHouseVal on Test Data")

# Saving plot
plt.savefig("figures/test_actual_vs_pred.png")
plt.close()

print("Test performance plot saved as 'figures/test_actual_vs_pred.png'")

