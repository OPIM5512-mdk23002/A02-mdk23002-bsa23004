from sklearn.datasets import fetch_california_housing
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
