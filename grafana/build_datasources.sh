#!/bin/bash

if [ "$1" = "-h" ];then
	echo """
This script takes every file from the folder 'intelligent-planning-of-care-staff/output' and creates a corresponding datasource
for grafana in the folder 'intelligent-planning-of-care-staff/grafana/grafana_etc/provisioning/datasources'.
To use these generated datasources:
1. start grafana and login
2. edit the panel displaying the data
3. select another datasource
4. click on query options > refresh
5. apply the changes and view your new data
	"""
fi

for FILE in `ls ../output/ | grep -v descriptive_analysis | grep -v README.md | sed 's/.csv//g'`; do
	cat << EOF > grafana_etc/provisioning/datasources/${FILE}.yaml
apiVersion: 1

datasources:
  - name: CSV-${FILE}
    type: marcusolsson-csv-datasource
    access: proxy
    url: "/data/${FILE}.csv"
    isDefault: false
    basicAuth: false
    editable: true
    jsonData:
      storage: "local"
EOF
done


