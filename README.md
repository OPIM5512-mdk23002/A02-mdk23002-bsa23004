# A02 Mark Kulaga, Bayani Aquino Jr.
 This repo is for Assignment 2 in OPIM 5512. It was used to practice collaboration through branches and properly handling pull requests to manage changes. To practice this, we worked together to run a basic machine learning model predicting Median House Value from the California Housing Dataset and saving some plots from its outputs.

## Workflow
1. Loaded California Housing dataset uswing sklearn.
2. Split data into training and test partitions
3. Scaled features with StandardScaler
4. Developed an MLPRegressor model and fit it with the training split
5. Applied model to training data, then evaluated performance, and created output plot
6. Applied model to test data, then evaluated performance, and created output plot

## Data
California Housing Dataset

## How to Setup and Run
*Ensure that the figures folder exists in the repository
*Run the following code snippets in your terminal (From the repo root)
# Ensure  your working directory is the A02-mdk23002-bsa23004 repo
cd path\to\A02-mdk23002-bsa23004
# install packages and dependencies
pip install -r requirements.txt
# run the ds_pipeline.py script
python .\src\ds_pipeline.py
