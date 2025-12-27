import sys
import json
import numpy as np
import pandas as pd
import cv2

def main():
    if len(sys.argv) < 3:
        print("Uso: python calcular_H.py <puntos.csv> <salida.json>")
        sys.exit(1)

    csv_in = sys.argv[1]
    json_out = sys.argv[2]

    df = pd.read_csv(csv_in)

    # Comprobación rápida de columnas necesarias
    required = {"x_img", "y_img", "x_field", "y_field"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"El CSV debe tener columnas {required}. Tiene: {df.columns.tolist()}")

    pts_img = df[["x_img", "y_img"]].astype(np.float32).values
    pts_field = df[["x_field", "y_field"]].astype(np.float32).values

    # Homografía robusta
    H, mask = cv2.findHomography(
        pts_img, pts_field,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5
    )
    if H is None:
        raise RuntimeError("No se pudo calcular H. Revisa los puntos (quizá están mal o colineales).")

    # Guardar en JSON
    with open(json_out, "w") as f:
        json.dump({"H": H.tolist()}, f, indent=2)

    inliers = int(mask.sum()) if mask is not None else len(df)
    print(f"✅ Guardado {json_out} | inliers: {inliers}/{len(df)}")

if __name__ == "__main__":
    main()
