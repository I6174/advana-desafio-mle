import fastapi
from fastapi import HTTPException, status
import pandas as pd
from typing import List, Dict, Any

from challenge.model import DelayModel

app = fastapi.FastAPI()

model = DelayModel()

#Listas para validación de entradas
VALID_OPERA = {
    "American Airlines", "Air Canada", "Air France", "Aeromexico",
    "Aerolineas Argentinas", "Austral", "Avianca", "Alitalia",
    "British Airways", "Copa Air", "Delta Air", "El Al",
    "Gol Trans", "Iberia", "KLM", "Lacsa",
    "Latin American Wings", "Latam", "Grupo LATAM", "Sky Airline",
    "United Airlines", "Vueling Airlines", "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas"
}

VALID_TIPOVUELO = {"I", "N"}


def validate_flight(flight: Dict[str, Any]) -> None:
    """Valida los campos obligatorios de cada vuelo."""
    if "OPERA" not in flight or "TIPOVUELO" not in flight or "MES" not in flight:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Faltan campos requeridos en uno o más vuelos."
        )

    opera = flight.get("OPERA")
    tipovuelo = flight.get("TIPOVUELO")
    mes = flight.get("MES")

    if not isinstance(mes, int) or mes < 1 or mes > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mes inválido: {mes}. Debe ser un entero entre 1 y 12."
        )

    if tipovuelo not in VALID_TIPOVUELO:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"TIPOVUELO inválido: {tipovuelo}. Debe ser 'I' o 'N'."
        )

    if opera not in VALID_OPERA:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OPERA inválido: {opera}."
        )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(payload: dict) -> dict:
    #Verificamos la llave
    if "flights" not in payload or not isinstance(payload["flights"], list) or len(payload["flights"]) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La petición debe incluir una lista no vacía en la clave 'flights'."
        )

    flights = payload["flights"]

    #Validamos el objeto
    for flight in flights:
        validate_flight(flight)

    data = pd.DataFrame(flights)
    features = model.preprocess(data=data)
    predictions = model.predict(features=features)

    return {
        "predict": predictions
    }