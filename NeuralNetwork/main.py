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
arber_data = pd.read_csv(cwd + "/data/Arber.csv")
schorndorf_data = pd.read_csv(cwd + "/data/Schorndorf.csv")
straubing_data = pd.read_csv(cwd + "/data/Straubing.csv")

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

arber_data_selected = arber_data_filled[["DATE", "LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE", "DAMPFDRUCK", "NIEDERSCHLAGSHOEHE_IND", "SCHNEEHOEHE"]]
schorndorf_data_selected = schorndorf_data_filled[["DATE", "LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE", "DAMPFDRUCK", "NIEDERSCHLAGSHOEHE_IND", "SCHNEEHOEHE"]]
straubing_data_selected = straubing_data_filled[["DATE", "LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE", "DAMPFDRUCK", "NIEDERSCHLAGSHOEHE_IND", "SCHNEEHOEHE"]]

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

# Add Prefix to data
arber_data_prefixed = arber_data_shifted.add_prefix("arber_")
schorndorf_data_prefixed = schorndorf_data_shifted.add_prefix("schorndorf_")
straubing_data_prefixed = straubing_data_selected.add_prefix("straubing_")

merged_data = straubing_data_prefixed.merge(arber_data_prefixed, left_on='straubing_DATE', right_on="arber_DATE", how='left')
merged_data = merged_data.merge(schorndorf_data_prefixed, how="left", left_on="straubing_DATE", right_on="schorndorf_DATE")

# Löschen von nicht vorhandenen Daten (das ist die ursache, dass das modell nan werte ausgibt)
merged_data = merged_data[~np.any(np.isnan(merged_data), axis=1)]

# Löschen von 999 Werten (verzerrt die Optimierung des NN)
merged_data = merged_data[~merged_data.isin([-999]).any(axis=1)]

used_data = merged_data.copy()
used_data.drop(columns=["schorndorf_DATE", "straubing_DATE", "arber_DATE"], axis=1)


# 6. Normalization (Example: Using Min-Max Scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(used_data.select_dtypes(include=['float64', 'int64']))
scaled_data = used_data.copy()

column_names = used_data.select_dtypes(include=["float64", "int64"]).columns

# Convert scaled_data to DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=column_names)

data_col_index = 0


X = scaled_data.drop(columns=["straubing_LUFTTEMPERATUR"], axis=1)
y = scaled_data["straubing_LUFTTEMPERATUR"]

train_size = int(len(scaled_data)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

dates = merged_data[["straubing_DATE"]]
dates_train, dates_test = dates[:train_size], dates[train_size:]

# Initialize the Neural Network
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))  # Adjust number of neurons
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32)  # Adjust epochs and batch_size as needed

##########################################

# Predictions
predictions = model.predict(X_test)

# Assuming 'X_test' are your input features and 'y_test' are the actual values
# 'predictions' are the outputs from your model

# Convert predictions to a DataFrame for easier manipulation
predictions_df = pd.DataFrame(predictions, columns=['Predicted'])

# Reset index on y_test if it's a pandas Series or DataFrame
y_test_reset = y_test.reset_index(drop=True)

# Combine actual values, predictions, and inputs (X_test) into one DataFrame
results_df = pd.concat([X_test.reset_index(drop=True), y_test_reset, predictions_df], axis=1)

# Calculate errors (absolute or squared errors can be used depending on your need)
results_df['Error'] = abs(results_df['Predicted'] - results_df['straubing_LUFTTEMPERATUR'])

# Define a threshold for outliers, this is subjective and depends on your specific case
error_threshold = results_df['Error'].quantile(0.95)  # Example: errors in the top 5%

# Filter to find instances where the error exceeds this threshold
outlier_predictions = results_df[results_df['Error'] > error_threshold]

# Now, outlier_predictions contains the inputs and predictions that are outliers
# print(outlier_predictions)

###############################################

# Calculate Mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions.flatten())
print("Mean Squared Error:", mse)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test, label='Actual')
plt.plot(dates_test, predictions.flatten(), label='Predicted', alpha=0.7)
plt.title('TEMPERATUR Prediction')
plt.xlabel('Date')
plt.ylabel('TEMPERATUR')
plt.legend()
plt.show()
