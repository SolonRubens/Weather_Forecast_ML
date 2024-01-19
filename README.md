# Weather Forecast

## Autoren
- Jonathan Silber
- Monja Biendl
- Benedict Sohler

## Projektbeschreibung
Vorhersage des Niederschlags und der Lufttemperatur für die nächsten drei Tage. Hierfür werden Informationen aus den Wetterstationen von Arber, Schorndorf und Straubing verwendet. Die Prognose erfolgt mithilfe von Maschinenlern-Algorithmen. Die verwendeten Modelle sind:
- Linear Regression
- Verwandte Modelle des Regression Trees
- Neuronales Netzwerk

## Datenübersicht
Die Daten der Stationen stammen aus den jeweiligen CSV-Dateien (`data/Arber.csv`, `data/Schorndorf.csv`, `data/Straubing.csv`). Diese enthalten für jede Messung die folgenden Features. Eine Messung pro Tag wird in den Daten festgehalten:
- `DATE`
- `MESS_DATUM`
- `QUALITAETS_NIVEAU`
- `LUFTTEMPERATUR`
- `DAMPFDRUCK`
- `BEDECKUNGSGRAD`
- `LUFTDRUCK_STATIONSHOEHE`
- `REL_FEUCHTE`
- `WINDGESCHWINDIGKEIT`
- `LUFTTEMPERATUR_MAXIMUM`
- `LUFTTEMPERATUR_MINIMUM`
- `LUFTTEMP_AM_ERDB_MINIMUM`
- `WINDSPITZE_MAXIMUM`
- `NIEDERSCHLAGSHOEHE`
- `NIEDERSCHLAGSHOEHE_IND`
- `SONNENSCHEINDAUER`
- `SCHNEEHOEHE`

Nicht vorhandene Daten werden mit `-999` markiert. Herausforderungen bei den Daten sind:
1. Viele fehlende Daten
2. Die jeweiligen Zielfeatures werden nicht mit abgespeichert.
   Diese müssen selbstständig berechnet werden
3. Die Daten sind aufgeteilt in verschiedene Dateien.

## Datenaufbereitung
Die Datenaufbereitung (`Datenaufbereitung/DatenKombinieren.py`) hat die Funktionalität, die vorhandenen Daten so zu ändern, dass diese einfach in ein Modell geladen werden können. Dabei wird zusätzlich auf die Probleme der Daten eingegangen und behoben. Das Python-Script führt folgende Schritte aus:
- Laden der Daten
- Ersetzen von nicht vorhandenen Daten mit NaN-Werten. Diese sind in Python einfacher zu behandeln.
- Kopieren der Zielspalten und Verschieben um die Anzahl der Tage, in die die Prognose gemacht werden muss.
  Dabei werden die Zielfeatures in dieselbe Spalte wie die Daten für die Vorhersage platziert.
- Lineare Interpolation der Temperaturdaten.
- Zusammenführen der Daten aus den einzelnen Wetterstationen zu einem Datensatz, wobei das Datum als Schlüssel verwendet wird.
- Entfernen von Spalten, in denen viele Werte fehlen, oder von Spalten, von denen wenig aussagekräftige Informationen erwartet werden.
- Aufteilung der Daten in Quell-, Ziel- und Datumstabellen (`data/Featchers_sorted.csv`, `data/Goals_sorted.csv`, `data/Timings_sorted.csv`, `data/Featchers_randomized.csv`, `data/Goals_randomized.csv`, `data/Timings_randomized.csv`), um sicherzustellen, dass die Spalten weiterhin zueinander gehören.

Für das Neuronale Netz werden diese anpassungen seperat gemacht, da hier im Laufe des Projekts viele anpassungen gemacht worden sind.
Beschrieben sind diese im Jupiter notebook des Neuronalen Netzes.

## Lineare Regression
Die Lineare Regression wird im Jupiter notebook `Linear/Linear.ipynb` beschrieben.

## Regression Tree Modelle
Die Regression Bäume werden im Jupiter notebook `RandomForest/Project.ipynb` beschrieben.

## Neuronales Netz
Das Neuronale Netz wird im Jupiter notebook `NeuralNetwork/NeuralNetwork.ipynb` beschrieben.