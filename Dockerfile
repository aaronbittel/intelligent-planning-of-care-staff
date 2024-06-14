# Verwende das offizielle Python-Image als Basis
FROM python:3.10

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere den Rest des Anwendungsquellcodes
COPY gui /app/gui
COPY models /app/models
COPY output/hero_dmc_heart_institute_india.csv /app/output/

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Kopiere die requirements-Datei und installiere die Abhängigkeiten
COPY ./pyproject.toml ./
RUN pip install .

# Exponiere den Port, den Streamlit verwendet
EXPOSE 8501

# Starte die Streamlit-Anwendung
CMD ["streamlit", "run", "gui/Home.py"]

