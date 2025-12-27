import json
import numpy as np
import pandas as pd

H_JSON = "homografia_H.json"

# Ajusta a tu fichero de detecciones RAW (el del tracker/YOLO con x1,y1,x2,y2)
IN_CSV  = "detecciones_raw_match2.csv"
OUT_CSV = "detecciones_match2_metros.csv"

# Campo estándar
LARGO = 105.0
ANCHO = 68.0

# Columnas esperadas (cambia aquí si tu CSV usa otros nombres)
FRAME_COL = "frame"
ID_COL    = "id"
X1, Y1, X2, Y2 = "x1", "y1", "x2", "y2"

def main():
    with open(H_JSON, "r") as f:
        H = np.array(json.load(f)["H"], dtype=np.float64)

    df = pd.read_csv(IN_CSV)

    # Punto del jugador = bottom-center de la bbox
    cx = (df[X1] + df[X2]) / 2.0
    cy = df[Y2].astype(float)

    pts = np.stack([cx, cy, np.ones(len(df))], axis=1)  # Nx3

    proj = (H @ pts.T).T  # Nx3
    X_m = proj[:, 0] / proj[:, 2]
    Y_m = proj[:, 1] / proj[:, 2]

    df["X_m"] = X_m
    df["Y_m"] = Y_m

    # Filtrar lo que cae fuera del campo
    df = df[(df["X_m"] >= 0) & (df["X_m"] <= LARGO) & (df["Y_m"] >= 0) & (df["Y_m"] <= ANCHO)].copy()

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Guardado: {OUT_CSV} | filas: {len(df)}")

if __name__ == "__main__":
    main()
