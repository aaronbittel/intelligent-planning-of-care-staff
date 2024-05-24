# Schnellstart Anleitung

## Virtuelle Python Umgebung und Paketinstallation

Virtuelle Umgebung erstellen:
```console
python -m venv venv
```
oder
```console
python3 -m venv venv
```

* Virtuelles Environment aktivieren

MacOS
```console
.venv/bin/activate 
```
Windows
```console
venv\Scripts\activate 
```
Linux
```console
. venv/bin/activate
```

* Benötigte Pakete installieren
```console
pip install pyproject.toml
```

## Grafana Docker starten

```console
cd grafana
docker compose up -d
```

## GUI starten

```console
streamlit run gui/Home.py
```

## Einfache Vorhersage generieren

Falls Page nicht automatisch im Browser aufgerufen wird:  
http://localhost:8501

- Auf der Startpage "Start" klicken
- Im Setup auf "Predict" klicken
- Grafana Login Daten:
  - User: demo
  - Passwort: test


# Anwendungsstruktur

## GUI

### Seiten 
Der Folder gui enthält unsere streamlit Anwendung. Die Anwendung besteht aus 3 Pages:

**Home:**  
Startpage  

**Setup:**  
Möglichkeit zum Hochladen eines eigenen Datensatzes in CSV Format, andernfalls wird ein
Beispiel-Datensatz verwendet.  

Advanced (Optionale Konfiguration) 
  - Modell-Auswahl
  - eigene Parameter Auswahl für jedes Modell
  - Verschiedene Ausführungsmöglichkeiten
    - Forecast -> Tatsächliche Vorhersagenberechnung
    - Test -> Vergleich Datensatz + Vorhersage
    - Accurate -> Multiple Runs mit Teilen des Datensatzes um bestes der drei Modelle zu bestimmen
     
**Forecast:**  
Anzeige des Grafana Dashboards mit den Ergebnissen der Modelle, eingebunden als I-Frame.
Falls ein Vergleich mit dem Datensatz möglich ist (Test/Accurate), wird der Root Mean Squared Error
und der Mean Average Percentage Error mit ausgegeben.

## Modelle

Es wurden drei Modelle implementiert, SARIMA, Random Forest und Holt-Winters seasonal method. Jedes Modell hat seinen eigenen Ordner unter '*modelle/*'. 
Alle Modelle benötigen Input im Format eines Pandas Dataframe mit den Spalten 'dates' und 'occupancy'.

Die drei Modelle werden über das Script '*models/wrapper.py*' aufgerufen, dieses kann getestet werden via '*models/test_call_wrapper.py*'

## Daten Visualisierung

Zur Visualisierung der Daten wird ein Grafana Docker-Container verwendet. 
Die Konfiguration erfolgt in der Datei '*grafana/docker-compose.yml*'. 

Das in die GUI eingebundene Dashboard (mixed-data) zeigt vier Dateien an, die Daten auf denen die letzte Berechnung erstellt
wurde und die Ergebnisse der drei Vorhersagemodelle:  
- *output/latest_history.csv* 
- *output/lastest_holt_winter.csv* 
- *output/lastest_random_forest.csv*
- *output/lastest_sarima.csv*

Man kann das Dashboard direkt aufrufen, unter http://localhost:3000