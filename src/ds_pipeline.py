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
