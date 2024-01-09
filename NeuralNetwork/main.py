import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

cwd = str(Path.cwd())

# Load Data
arber_data = pd.read_csv(cwd + "/data/Arber.csv")
schorndorf_data = pd.read_csv(cwd + "/data/Schorndorf.csv")
straubing_data = pd.read_csv(cwd + "/data/Straubing.csv")

print("Initial Arber Data:")
print(arber_data.head())
print(arber_data.isnull().sum())