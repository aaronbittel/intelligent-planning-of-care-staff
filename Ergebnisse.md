## Ergebnisse
Das Ergebnis unseres Projektes ist der Zusammenschluss mehrerer KI-Modelle innerhalb eines Wrappers und die Auswahl, Einstellung und Bewertung der Modelle in einer interaktiven graphischen Benutzeroberfläche. Die Nutzung der Software wird im Folgenden dargestellt.

1. Beim Öffnen der Website wird der Homescreen angezeigt. Links in der Abbildung werden eine Übersicht der Anwendung dargestellt und dienen der Nutzerfreundlichkeit und vereinfachten Navigation. Durch Klick auf „START“ gelangt man ins Setup-Menü, um die Predictions zu starten.

![1_home](/images/1_home.png)

2. Im Setup-Bereich wird die Eingabedatei (Datensatz) ausgewählt, für die die Modelle Vorhersagen treffen sollen.  

![2_setup](/images/2_setup.png)

3. Wurde eine kompatible Datei ausgewählt, kann die Basic-Vorhersage schnell und einfach durch Einstellen des Vorhersage-Zeitintervalls in Tagen und anschließender Bestätigung mit „PREDICT“ gestartet werden. Optional können über den Menüpunkt „Occupancy Analysis“ der wöchentliche sowie der monatliche Belegungsdurchschnitt übersichtlich angezeigt werden.  

![3_occupancy_analysis](/images/3_occupancy_analysis.png) 

4. Ist man mit den Modellen und Parametern vertraut, besteht weiterhin die Möglichkeit, spezifische Modelle, Parameter und Ablauftypen auszuwählen und anzupassen. Dies wird durch einen Klick auf „Advanced“ realisiert. Im Advanced-Bereich kann man einzelne zur Verfügung stehende KI-Modelle aus- oder abwählen.  

![4_advanced](/images/4_advanced.png)  

5. Wurde ein Modell ausgewählt, wird es zur Registerkarte „Modelle“ hinzugefügt und die Parameter des Modells (geprüfte Standardwerte) erscheinen. Die spezifischen Modellparameter können nun aus einer Auswahl von Werten oder innerhalb eines gültigen Wertebereichs neu eingestellt werden. Nur für das Modell gültige Parameterwerte werden akzeptiert. Dies ermöglicht eine zielspezifische Feinabstimmung der einzelnen Modelle. 

![5_advanced_parameters](/images/5_advanced_parameters.png)  

6. Nach der Feinabstimmung der Modellparameter kann der gewünschte Programmtyp ausgewählt werden. Es werden drei Typen unterschieden. Die Erläuterung der Typendifferenzierung ist ebenfalls im Hilfebereich „Type Explanation“ zu finden.  
- **Forecast**: Die Modelle treffen eine Vorhersage über die Belegung der Krankenhausbetten auf den ausgewählten Datensatz.  

- **Test**: Durch diese Auswahl können die Qualität und Genauigkeit der Modell-Vorhersagen für den entsprechenden Datensatz beurteilt werden. Die Modelle stellen Vorhersagen auf, welche mit einem Testbereich des Datensatzes verglichen werden. Die Abweichung der Vorhersage von den tatsächlichen Testwerten wird durch die Fehlermetriken „RMSE“ – Fehler im quadratischen Mittelwert, „MAPE“ – mittlerer absoluter prozentualer Fehler und „MAE“ – mittlerer absoluter Fehler berechnet.  

- **Accurate**: Stellt den gleichen Ansatz wie „Test“ dar. Der Unterschied zum Test besteht darin, dass die Berechnung der Vorhersagequalität und der entsprechenden Fehlermetriken in mehreren Iterationen und damit ausführlicher durchgeführt wird, was die Aussagekraft der Fehlermetriken erhöhen soll. Es wird der Zeitreihen-Kreuzvalidierer _TimeSeriesSplit_ aus der Python-Bibliothek _skicit-learn_ verwendet, der den Datensatz in mehrere logisch getrennte Test- und Trainingsdatensätze aufteilt. Anschließend wird der Mittelwert aller generierten Fehlermetriken berechnet und angezeigt.  

7. Durch Klick auf „PREDICT“ wird die Vorhersage gestartet.  

![6_type_explanation](/images/6_type_explanation.png)  

8. Mithilfe der plattformübergreifenden Open-Source-Anwendung _Grafana_ werden der Datensatz und die berechneten Vorhersagen der ausgewählten Modelle als interaktives I-Frame angezeigt.

![7_forecast](/images/7_forecast.png)  

9. Es können bevorzugte Bereiche des Graphen interaktiv ausgewählt und anschließend im Detail betrachtet werden.

![8_forecast_detailed](/images/8_forecast_detailed.png)  

10. Die Fehlermetriken zur Validierung der Aussagekraft werden unterhalb der Graphen dargestellt.  

11. Die getroffenen Vorhersagen können über „Download CSV“ heruntergeladen und zur weiteren Verarbeitung genutzt werden.  

![9_error_metrics_download](/images/9_error_metrics_download.png)  
