import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
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
arber_numeric_filled = pd.DataFrame(imputer.fit_transform(arber_numeric.replace), columns=arber_numeric.columns)
schorndorf_numeric_filled = pd.DataFrame(imputer.fit_transform(schorndorf_numeric), columns=schorndorf_numeric.columns)
straubing_numeric_filled = pd.DataFrame(imputer.fit_transform(straubing_numeric), columns=straubing_numeric.columns)

# Combining back together the numeric and non numeric data
arber_data_filled = pd.concat([arber_non_numeric.reset_index(drop=True), arber_numeric_filled.reset_index(drop=True)], axis=1)
schorndorf_data_filled = pd.concat([schorndorf_non_numeric.reset_index(drop=True), schorndorf_numeric_filled.reset_index(drop=True)], axis=1)
straubing_data_filled = pd.concat([straubing_non_numeric.reset_index(drop=True), straubing_numeric_filled.reset_index(drop=True)], axis=1)

arber_data_selected = arber_data_filled[["DATE", "LUFTTEMPERATUR",  "REL_FEUCHTE", "LUFTTEMPERATUR_MAXIMUM", "LUFTTEMPERATUR_MINIMUM", "LUFTTEMP_AM_ERDB_MINIMUM", "NIEDERSCHLAGSHOEHE", "DAMPFDRUCK", "NIEDERSCHLAGSHOEHE_IND", "SCHNEEHOEHE"]]
schorndorf_data_selected = schorndorf_data_filled[["DATE", "LUFTTEMPERATUR", "REL_FEUCHTE", "NIEDERSCHLAGSHOEHE", "LUFTTEMPERATUR_MAXIMUM", "LUFTTEMPERATUR_MINIMUM", "LUFTTEMP_AM_ERDB_MINIMUM", "DAMPFDRUCK", "NIEDERSCHLAGSHOEHE_IND", "SCHNEEHOEHE"]]
straubing_data_selected = straubing_data_filled[["DATE", "LUFTTEMPERATUR", "REL_FEUCHTE", "NIEDERSCHLAGSHOEHE", "LUFTTEMPERATUR_MAXIMUM", "LUFTTEMPERATUR_MINIMUM", "LUFTTEMP_AM_ERDB_MINIMUM", "DAMPFDRUCK", "NIEDERSCHLAGSHOEHE_IND", "SCHNEEHOEHE"]]

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

column_means = merged_data.replace(-999, np.nan).mean()
# Replace NaN values with the corresponding column mean values
# merged_data.fillna(column_means, inplace=True) 
merged_data.replace(-999, column_means, inplace=True)


# Löschen von 999 Werten (verzerrt die Optimierung des NN)
#merged_data = merged_data[~merged_data.isin([-999]).any(axis=1)]

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

scaled_data["Target_Day1"] = scaled_data["straubing_LUFTTEMPERATUR"].shift(-1)
scaled_data["Target_Day2"] = scaled_data["straubing_LUFTTEMPERATUR"].shift(-2)
scaled_data["Target_Day3"] = scaled_data["straubing_LUFTTEMPERATUR"].shift(-3)

# scaled_data["Target_Day1"] = scaled_data["straubing_NIEDERSCHLAGSHOEHE"].shift(-1)
# scaled_data["Target_Day2"] = scaled_data["straubing_NIEDERSCHLAGSHOEHE"].shift(-2)
# scaled_data["Target_Day3"] = scaled_data["straubing_NIEDERSCHLAGSHOEHE"].shift(-3)

scaled_data = scaled_data[:-3]


data_col_index = 0


X = scaled_data.drop(columns=["straubing_LUFTTEMPERATUR", "Target_Day1", "Target_Day2", "Target_Day3"], axis=1)
# X = scaled_data.drop(columns=["straubing_NIEDERSCHLAGSHOEHE", "Target_Day1", "Target_Day2", "Target_Day3"], axis=1)
y = scaled_data[["Target_Day1", "Target_Day2", "Target_Day3"]]

train_size = int(len(scaled_data)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshaping X_train and _test for the RNN
# Example: Assuming X_train and X_test are your input data
# X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))


merged_data = merged_data[:-3]
dates = merged_data[["straubing_DATE"]]
dates_train, dates_test = dates[:train_size], dates[train_size:]


# Initialize the Neural Network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(32, activation='elu'))  # Adjust number of neurons
model.add(Dense(3))

# def modified_mse(y_true, y_pred):
#     penalty_weight = 10.0  # Adjust the penalty weight
#     squared_error = tf.square(y_true - y_pred)
#     penalty = tf.where(y_pred < 0, penalty_weight * squared_error, squared_error)
#     return tf.reduce_mean(penalty)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# model.compile(optimizer='adam', loss=modified_mse)

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=32)  # Adjust epochs and batch_size as needed

# ##########################################

# # Predictions
predictions = model.predict(X_test)

# # Assuming 'X_test' are your input features and 'y_test' are the actual values
# # 'predictions' are the outputs from your model

# # Convert predictions to a DataFrame for easier manipulation
# predictions_df = pd.DataFrame(predictions, columns=['Predicted'])

# # Reset index on y_test if it's a pandas Series or DataFrame
# y_test_reset = y_test.reset_index(drop=True)

# # Combine actual values, predictions, and inputs (X_test) into one DataFrame
# results_df = pd.concat([X_test.reset_index(drop=True), y_test_reset, predictions_df], axis=1)

# # Calculate errors (absolute or squared errors can be used depending on your need)
# results_df['Error'] = abs(results_df['Predicted'] - results_df['straubing_LUFTTEMPERATUR'])

# # Define a threshold for outliers, this is subjective and depends on your specific case
# error_threshold = results_df['Error'].quantile(0.95)  # Example: errors in the top 5%

# # Filter to find instances where the error exceeds this threshold
# outlier_predictions = results_df[results_df['Error'] > error_threshold]

# # Now, outlier_predictions contains the inputs and predictions that are outliers
# # print(outlier_predictions)

# ###############################################

# Prepare the actual values for comparison
N = len(predictions)
actual_day1 = y_test.iloc[:N, 0]
actual_day2 = y_test.iloc[:N, 1]
actual_day3 = y_test.iloc[:N, 2]


# Calculate Mean squared error and R2 Score
from sklearn.metrics import mean_squared_error, r2_score
mse1 = mean_squared_error(actual_day1, predictions[:, 0])
mse2 = mean_squared_error(actual_day2, predictions[:, 1])
mse3 = mean_squared_error(actual_day3, predictions[:, 2])
r2_1 = r2_score(actual_day1, predictions[:, 0])
r2_2 = r2_score(actual_day2, predictions[:, 1])
r2_3 = r2_score(actual_day3, predictions[:, 2])
print("Mean Squared Error1:", mse1)
print("Mean Squared Error2:", mse2)
print("Mean Squared Error3:", mse3)
print("R2 Score 1:", r2_1)
print("R2 Score 2:", r2_2)
print("R2 Score 3:", r2_3)

# Visualization
# plt.figure(figsize=(10, 6))
# plt.plot(dates_test, y_test, label='Actual')
# plt.plot(dates_test, predictions.flatten(), label='Predicted', alpha=0.7)
# plt.title('TEMPERATUR Prediction')
# plt.xlabel('Date')
# plt.ylabel('TEMPERATUR')
# plt.legend()
# plt.show()


dates_pred_day1 = dates_test.shift(-1)  # Exclude the last 3 dates for the 1st day prediction
dates_pred_day2 = dates_test.shift(-2) # Shift for the 2nd day prediction
dates_pred_day3 = dates_test.shift(-3) # Shift for the 3rd day prediction

plt.figure(figsize=(15, 5))


# Plot for Day 1 Prediction
plt.subplot(1, 3, 1)
plt.plot(dates_test, actual_day1, label='Actual', color='blue')
plt.plot(dates_pred_day1, predictions[:, 0], label='Predicted Day 1', color='red', alpha=0.7)
# plt.title('Day 1 Temperature Prediction')
plt.title('Day 1 Rainfall Prediction')
plt.xlabel('Date')
# plt.ylabel('Temperature')
plt.ylabel('Rainfall')
plt.legend()

# Plot for Day 2 Prediction
plt.subplot(1, 3, 2)
plt.plot(dates_test, actual_day2, label='Actual', color='blue')
plt.plot(dates_pred_day2, predictions[:, 1], label='Predicted Day 2', color='orange', alpha=0.7)
# plt.title('Day 2 Temperature Prediction')
plt.title('Day 2 Rainfall Prediction')
plt.xlabel('Date')
# plt.ylabel('Temperature')
plt.ylabel('Rainfall')
plt.legend()

# Plot for Day 3 Prediction
plt.subplot(1, 3, 3)
plt.plot(dates_test, actual_day3, label='Actual', color='blue')
plt.plot(dates_pred_day3, predictions[:, 2], label='Predicted Day 3', color='green', alpha=0.7)
# plt.title('Day 3 Temperature Prediction')
plt.title('Day 3 Rainfall Prediction')
plt.xlabel('Date')
# plt.ylabel('Temperature')
plt.ylabel('Rainfall')
plt.legend()

plt.tight_layout()
plt.show()