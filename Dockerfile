# syntax=docker/dockerfile:1.2
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

#Copiar e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copiar el paquete con la lógica y la API, y junto con los datos requeridos
COPY challenge /app/challenge
COPY data /app/data

EXPOSE 8080

#Iniciar la API con Uvicorn en el puerto que asigne Cloud Run ($PORT)
CMD ["sh", "-c", "uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}"]