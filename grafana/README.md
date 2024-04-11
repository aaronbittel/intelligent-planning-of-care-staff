# Grafana Service

Mit dieser docker-compose Konfiguration richten wir Grafana, eine Open-Source-Plattform zur interaktiven Datenvisualisierung, ein. Grafana ermöglicht die graphische Darstellung der Daten in Form von Graphen oder Diagrammen, die in Dashboards zusammengefasst werden können und erleichtert die Interpretation und Analyse der Daten.

## Installation

[Docker](https://www.docker.com/get-started/) muss auf Ihrem lokalen Rechner installiert und geöffnet sein. 

1. Navigieren Sie im Terminal zum Verzeichnis 'grafana', in der sich die Datei 'docker-compose.yaml' befindet
2. Um Grafana zu starten führen Sie folgenen Befehl aus:  
    > docker compose up -d
3. Nachdem die Installation des Grafana Image abgeschlossen ist, können Sie in ihrem Webbrowser unter 'http://localhost:3000' Grafana aufrufen
4. Melden Sie sich mit folgenden Daten ein:  
	- Benutzername: demo  
	- Passwort: test  
5. Nun ist Grafana fertig eingerichtet und nutzbar  

## Einfügen und Nutzung einer neuen Datenquelle  

1. Fügen Sie eine neue csv-Datei in das Verzeichnis 'intelligent-planning-of-care-staff/output' hinzu  
3. Navagieren Sie zum Verzeichnis 'grafana', in der Sich die Datei 'build_datasources.sh' befindet und führen Sie diese aus   
2. In Grafana ist bereits ein Dashboard vorinstalliert, um die neue Datenquelle zu Nutzen, kopieren Sie das Dashboard   
3. Verändern Sie die Datenquelle und wählen Sie ihre gewünschte Datenquelle aus  
4. Klicken Sie auf 'query-option' und drücken Sie auf 'refresh'  
5. Übernehmen Sie die Änderungen 
6. Jetzt sind die neuen Daten sichtbar  

## Konfiguration

### Umgebungsvariablen 

	- "GF_INSTALL_PLUGINS=marcusolsson-csv-datasource" : Installation des Plugins 'marcusolsson-csv-datasource'
	- "GF_SECURITY_ADMIN_USER=demo" : Benutzername für den Grafana-Admin
	- "GF_SECURITY_ADMIN_PASSWORD=test" : Password für den Grafana-Admin
        
### Ports

	- '3000:3000' : Port 3000 auf dem lokalen Rechner wird mit dem Port 3000 im Container verbunden
	
### Volumes

	- ../output:/data : Bindet das Verzeichnis '../output' in das Verzeichnis '/data' vom Grafana-Container ein
	- ./grafana_etc:/etc/grafana/ : Bindet das Verzeichnis './grafana_etc' in das Verzeichnis '/etc/grafana' vom Grafana-Container ein





 
