#
# Copyright (c) 2024 Jonathan Silber, Monja Biendl, Benedict Sohler
# 
# Im zuge einer Projektarbeit für die OTH Regensburg
#
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

##################################################################################################################
# Konfiguration
##################################################################################################################
featcher_columns = ["LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE", "DAMPFDRUCK", "NIEDERSCHLAGSHOEHE_IND", "SCHNEEHOEHE"]
join_columns = ["DATE"]
predicting_columns = ["LUFTTEMPERATUR", "NIEDERSCHLAGSHOEHE"]
linear_interpolation_columns = ["LUFTTEMPERATUR"]
predict_future_days = 3
use_past_days = 0
date_format = "%d.%m.%Y"

##################################################################################################################
# Laden der Daten
##################################################################################################################
cwd = str(Path.cwd())

# individuelle Daten der drei standorte laden
arber_data = pd.read_csv(cwd + "/data/Arber.csv")
schorndorf_data = pd.read_csv(cwd + "/data/Schorndorf.csv")
straubing_data = pd.read_csv(cwd + "/data/Straubing.csv")

# Strippen der Namen der Spalten
arber_data.columns = arber_data.columns.str.strip()
schorndorf_data.columns = schorndorf_data.columns.str.strip()
straubing_data.columns = straubing_data.columns.str.strip()

print("Daten Geladen...")

##################################################################################################################
# Ersetzen von nichtvorhandenen Daten
##################################################################################################################
for df in [arber_data, schorndorf_data, straubing_data]:
    df.replace(-999, np.nan, inplace=True)

print("NAN werte ersetzt...")

##################################################################################################################
# Verschieben der daten
##################################################################################################################

# Verwenden der Vergangenen Daten als Featchers
############################################################
for df in [arber_data, schorndorf_data, straubing_data]:
    for parst in range(1, use_past_days + 1, 1):
        for column in featcher_columns:
            df[f'{column}_past_{parst}Day'] = df[column].shift(parst, fill_value=np.nan)
            print("Spalte Hinzugefügt: ", f'{column}_past_{parst}Day')

print(f'Vergangene werte der letzten {use_past_days} Tage als Featchers integriert...')

# Verwenden der zukunftsdaten als Zieldaten
############################################################
for df in [straubing_data]: # Hier nur Straubing, da wir nur eine Prediction für Straubing machen
    for future in range(1, predict_future_days + 1, 1):
        for column in predicting_columns:
            df[f'{column}_future_{future}Day'] = df[column].shift(-future, fill_value=np.nan)
            print("Spalte Hinzugefügt: ", f'{column}_future_{future}Day')

print(f'Zukünftige werte der nächsten {predict_future_days} Tage als Ziele integriert...')

##################################################################################################################
# Lineare Interpolation
##################################################################################################################
for df in [arber_data, schorndorf_data, straubing_data]:
    for col in linear_interpolation_columns:
        df[col].interpolate(method="linear", inplace=True)

print("Spalten Interpoliert...")

##################################################################################################################
# Anpassung des Datums
##################################################################################################################
for df in [arber_data, schorndorf_data, straubing_data]:
    df["DATE"] = pd.to_datetime(df["DATE"], format=date_format).dt.strftime(date_format)

print("Datumsformat angepasst...")

##################################################################################################################
# Join der Daten
##################################################################################################################
print("Daten vor dem Join (Straubing): ", len(straubing_data))
print("Daten vor dem Join (Schorndorf): ", len(schorndorf_data))
print("Daten vor dem Join (Arber): ", len(arber_data))
arber_data = arber_data.add_prefix('arber_')
schorndorf_data = schorndorf_data.add_prefix('schorndorf_')
straubing_data = straubing_data.add_prefix('straubing_')

merged = straubing_data.merge(schorndorf_data, left_on="straubing_DATE", right_on="schorndorf_DATE", how="inner")
merged = merged.merge(arber_data, left_on="straubing_DATE", right_on="arber_DATE", how="inner")

print("Daten gejoint (inner join)...")
print("Daten nach dem Join: ", len(merged))

##################################################################################################################
# Filterung der Daten
##################################################################################################################

# verwendung nur der relevanten Spalten
############################################################


# Spalten Selectieren
#-------------------------
columns_featcher = []
columns_goal = []
for prefix in ['arber_', 'schorndorf_', 'straubing_']:
    for column in featcher_columns:
        columns_featcher.append(f'{prefix}{column}')
        for parst in range(1, use_past_days + 1, 1):
            columns_featcher.append(f'{prefix}{column}_past_{parst}Day')
for prefix in ['straubing_']:
    for column in predicting_columns:
        for future in range(1, predict_future_days + 1, 1):
            columns_goal.append(f'{prefix}{column}_future_{future}Day')

# Spalten entfernen
#-------------------------
merged = merged[columns_featcher + columns_goal + ["straubing_DATE"]]

print("Ungenutzte Spalten entfernt...")

# Entfernen von unvollständigen Daten
############################################################

merged = merged.dropna()

print("Unvollständige Daten entfernt...")
print("Verbleibende Daten: ", len(merged))

##################################################################################################################
# Daten aufteilen und Speichern
##################################################################################################################

# Speichern der Daten in sortierter reinfolge
############################################################
featchers = merged[columns_featcher]
goals = merged[columns_goal]
timing = merged[["straubing_DATE"]]

featchers.to_csv(cwd + "/data/Featchers_sorted.csv", index=False)
goals.to_csv(cwd + "/data/Goals_sorted.csv", index=False)
timing.to_csv(cwd + "/data/Timings_sorted.csv", index=False)

print("Gespeichert: " + cwd + "/data/Featchers_sorted.csv")
print("Gespeichert: " + cwd + "/data/Goals_sorted.csv")
print("Gespeichert: " + cwd + "/data/Timings_sorted.csv")

print("Daten in [Featcher, Ziel, Zeit] aufgeteilt und abgespeichert...")

# Speichern der Daten in Zufälliger reinfolge
############################################################
merged = merged.sample(frac=1).reset_index(drop=True)

featchers = merged[columns_featcher]
goals = merged[columns_goal]
timing = merged[["straubing_DATE"]]

featchers.to_csv(cwd + "/data/Featchers_randomized.csv", index=False)
goals.to_csv(cwd + "/data/Goals_randomized.csv", index=False)
timing.to_csv(cwd + "/data/Timings_randomized.csv", index=False)

print("Gespeichert: " + cwd + "/data/Featchers_randomized.csv")
print("Gespeichert: " + cwd + "/data/Goals_randomized.csv")
print("Gespeichert: " + cwd + "/data/Timings_randomized.csv")

print("Daten geshuffelt, in [Featcher, Ziel, Zeit] aufgeteilt und abgespeichert...")

print("Daten Aufteilung alles fertig!!!")
