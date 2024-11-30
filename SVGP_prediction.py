import json
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('data/df_cleaned_10292024.csv')

# Separate df into training and testing data
df_train = df[df['ground_truth'].notnull()]
df_test = df[df['ground_truth'].isnull()]

# Shuffle the data
df_train = df_train.sample(frac=1, random_state=42)
df_train = df_train.reset_index(drop=True)