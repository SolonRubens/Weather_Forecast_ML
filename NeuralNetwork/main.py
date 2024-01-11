import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

cwd = str(Path.cwd())

# Load Data
arber_data = pd.read_csv(cwd + "/../data/Arber.csv")
schorndorf_data = pd.read_csv(cwd + "/../data/Schorndorf.csv")
straubing_data = pd.read_csv(cwd + "/../data/Straubing.csv")

# Strippen
arber_data.columns = arber_data.columns.str.strip()
schorndorf_data.columns = schorndorf_data.columns.str.strip()
straubing_data.columns = straubing_data.columns.str.strip()


# print("Initial Arber Data:")
# print(arber_data.head())
# print(arber_data.isnull().sum())

# Split numeric and non numeric columns
arber_numeric = arber_data.select_dtypes(include=["float64", "int64"])
arber_non_numeric = arber_data.select_dtypes(exclude=["float64", "int64"])

schorndorf_numeric = schorndorf_data.select_dtypes(include=["float64", "int64"])
schorndorf_non_numeric = schorndorf_data.select_dtypes(exclude=["float64", "int64"])

straubing_numeric = straubing_data.select_dtypes(include=["float64", "int64"])
straubing_non_numeric = straubing_data.select_dtypes(exclude=["float64", "int64"])


# Filling missing values with the mean value only on numeric data
imputer = SimpleImputer(strategy="mean")
arber_numeric_filled = pd.DataFrame(imputer.fit_transform(arber_numeric), columns=arber_numeric.columns)
schorndorf_numeric_filled = pd.DataFrame(imputer.fit_transform(schorndorf_numeric), columns=schorndorf_numeric.columns)
straubing_numeric_filled = pd.DataFrame(imputer.fit_transform(straubing_numeric), columns=straubing_numeric.columns)

# Combining back together the numeric and non numeric data
arber_data_filled = pd.concat([arber_non_numeric.reset_index(drop=True), arber_numeric_filled.reset_index(drop=True)], axis=1)
schorndorf_data_filled = pd.concat([schorndorf_non_numeric.reset_index(drop=True), schorndorf_numeric_filled.reset_index(drop=True)], axis=1)
straubing_data_filled = pd.concat([straubing_non_numeric.reset_index(drop=True), straubing_numeric_filled.reset_index(drop=True)], axis=1)

arber_data_selected = arber_data_filled[["DATE", "LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE"]]
schorndorf_data_selected = schorndorf_data_filled[["DATE", "LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE"]]
straubing_data_selected = straubing_data_filled[["DATE", "LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE"]]

date_format = "%d.%m.%Y"

# 4. Incorporating Time Lag
# Shift Arber and Schorndorf data by 3 days
arber_data_shifted = arber_data_selected.copy()
arber_data_shifted['DATE'] = pd.to_datetime(arber_data_shifted['DATE'], format=date_format) + pd.DateOffset(days=3)
schorndorf_data_shifted = schorndorf_data_selected.copy()
schorndorf_data_shifted['DATE'] = pd.to_datetime(schorndorf_data_shifted['DATE'], format=date_format) + pd.DateOffset(days=3)


# 5. Merging the Data
# Convert 'Date' to datetime for merging
straubing_data_selected.loc[:, 'DATE'] = pd.to_datetime(straubing_data['DATE'], format=date_format)

merged_data = straubing_data_selected.merge(arber_data_shifted, on='DATE', how='left', suffixes=('_straubing', '_arber'))
merged_data = merged_data.merge(schorndorf_data_shifted, on="DATE", how="left", suffixes=("", "_schorndorf"));

# 6. Normalization (Example: Using Min-Max Scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data.select_dtypes(include=['float64', 'int64']))

# Löschen von nicht vorhandenen Daten (das ist die ursache, dass das modell nan werte ausgibt)
scaled_data = scaled_data[~np.any(np.isnan(scaled_data), axis=1)]

data_col_index = 0
niederschlagshoehe_col_index = 2

X = np.delete(scaled_data, [data_col_index, niederschlagshoehe_col_index], axis=1)
y = scaled_data[:, 2]

train_size = int(len(scaled_data)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize the Neural Network
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))  # Adjust number of neurons
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)  # Adjust epochs and batch_size as needed

# Predictions
predictions = model.predict(X_test)

print("Acutal:", y_test)
print("Predicted:", predictions.flatten())

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(y_test, y_test, label='Actual')
plt.plot(y_test, predictions.flatten(), label='Predicted', alpha=0.7)
plt.title('NIEDERSCHLAGSHOEHE Prediction')
plt.xlabel('Date')
plt.ylabel('NIEDERSCHLAGSHOEHE')
plt.legend()
plt.show()
