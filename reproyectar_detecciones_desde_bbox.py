import json
import numpy as np
import pandas as pd

H_JSON = "homografia_H.json"

IN_CSV  = "detecciones_match2_equipos.csv"
OUT_CSV = "detecciones_match2_equipos_reproj.csv"

# Campo estándar
LARGO = 105.0
ANCHO = 68.0

# Columnas bbox
X1, Y1, X2, Y2 = "x1", "y1", "x2", "y2"

def main():
    with open(H_JSON, "r") as f:
        H = np.array(json.load(f)["H"], dtype=np.float64)

    df = pd.read_csv(IN_CSV)

    # Punto "pie" = bottom-center bbox
    cx = (df[X1] + df[X2]) / 2.0
    cy = df[Y2].astype(float)

    pts = np.stack([cx, cy, np.ones(len(df))], axis=1)  # Nx3

    proj = (H @ pts.T).T
    X_m = proj[:, 0] / proj[:, 2]
    Y_m = proj[:, 1] / proj[:, 2]

    # Guardamos en nuevas columnas (para comparar si quieres)
    df["X_m_new"] = X_m
    df["Y_m_new"] = Y_m

    # Filtrar dentro del campo
    df = df[
        (df["X_m_new"] >= 0) & (df["X_m_new"] <= LARGO) &
        (df["Y_m_new"] >= 0) & (df["Y_m_new"] <= ANCHO)
    ].copy()

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Guardado: {OUT_CSV} | filas: {len(df)}")

if __name__ == "__main__":
    main()
