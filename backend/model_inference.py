import os
import numpy as np
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))

_model   = joblib.load(os.path.join(_HERE, "wildfire_model.pkl"))
_encoder = joblib.load(os.path.join(_HERE, "fuel_label_encoder.pkl"))


def predict(
    lat: float, lon: float,
    wind_speed: float, wind_direction: float,
    fuel: str, slope: float, aspect: float,
    humidity: float, temp: float,
    ffmc: float, dmc: float, dc: float, bui: float, isi: float,
    pcp: float, elev: float,
) -> dict:
    fuel_enc = (
        _encoder.transform([fuel])[0]
        if fuel in _encoder.classes_
        else _encoder.transform(["unknown"])[0]
    )

    row = np.array([[
        lat, lon,
        wind_speed,
        np.sin(np.radians(wind_direction)),
        np.cos(np.radians(wind_direction)),
        fuel_enc,
        slope,
        np.sin(np.radians(aspect)),
        np.cos(np.radians(aspect)),
        humidity, temp,
        ffmc, dmc, dc, bui, isi,
        pcp, elev,
    ]])

    direction, severity = _model.predict(row)[0]
    return {
        "fire_direction": round(float(direction % 360), 1),
        "fire_severity":  round(float(np.clip(severity, 0, 1)), 4),
    }
