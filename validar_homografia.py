import json
import numpy as np
import pandas as pd
import cv2

IMG_PATH = "IMAGEN PARA ANALISIS PRO.png"
CSV_PATH = "puntos_homografia.csv"

# Tamaño del campo en metros
FIELD_L = 105.0
FIELD_W = 68.0

# Escala para el "top-down" (px por metro)
SCALE = 10  # 10px/m → salida 1050x680

OUT_H_JSON = "homografia_H.json"
OUT_OVERLAY = "validacion_overlay_en_frame.png"
OUT_TOPDOWN = "validacion_topdown.png"

def read_points(csv_path: str):
    df = pd.read_csv(csv_path)
    # Admite csv con o sin columna "name"
    cols = df.columns.str.lower().tolist()
    # Normalizar nombres
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return df.columns[cols.index(n.lower())]
        return None

    x_img = pick("x_img")
    y_img = pick("y_img")
    x_field = pick("x_field")
    y_field = pick("y_field")

    if not all([x_img, y_img, x_field, y_field]):
        raise ValueError(f"CSV debe contener columnas: x_img,y_img,x_field,y_field. Tiene: {df.columns.tolist()}")

    pts_img = df[[x_img, y_img]].astype(np.float32).values
    pts_field = df[[x_field, y_field]].astype(np.float32).values
    names = df["name"].values if "name" in df.columns else np.array([f"P{i+1}" for i in range(len(df))])
    return pts_img, pts_field, names

def reprojection_error(H, pts_img, pts_field):
    # proyectar pts_img -> field
    pts_h = cv2.perspectiveTransform(pts_img.reshape(-1, 1, 2), H).reshape(-1, 2)
    dif = pts_h - pts_field
    err = np.sqrt((dif**2).sum(axis=1))
    return err, pts_h

def draw_field_lines_overlay(img, Hinv):
    """Dibuja líneas del campo (en metros) proyectadas sobre la imagen usando Hinv."""
    overlay = img.copy()

    def proj(poly):
        poly = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        pts = cv2.perspectiveTransform(poly, Hinv).reshape(-1, 2)
        return pts.astype(int)

    # Líneas básicas: contorno, medio campo, círculo central, área derecha grande/pequeña
    # Contorno del campo
    field_outline = [(0,0),(FIELD_L,0),(FIELD_L,FIELD_W),(0,FIELD_W),(0,0)]
    pts = proj(field_outline)
    cv2.polylines(overlay, [pts], False, (0,255,255), 2)

    # Línea de medio campo x=52.5
    midline = [(FIELD_L/2, 0), (FIELD_L/2, FIELD_W)]
    pts = proj(midline)
    cv2.polylines(overlay, [pts], False, (0,255,255), 2)

    # Círculo central (aprox con 72 puntos)
    cx, cy, r = FIELD_L/2, FIELD_W/2, 9.15
    circle = [(cx + r*np.cos(t), cy + r*np.sin(t)) for t in np.linspace(0, 2*np.pi, 73)]
    pts = proj(circle)
    cv2.polylines(overlay, [pts], False, (0,255,255), 2)

    # Área grande derecha
    xA = FIELD_L - 16.5
    y1 = (FIELD_W/2) - 20.15
    y2 = (FIELD_W/2) + 20.15
    area_big = [(FIELD_L,y1),(xA,y1),(xA,y2),(FIELD_L,y2)]
    pts = proj(area_big)
    cv2.polylines(overlay, [pts], False, (0,255,255), 2)

    # Área pequeña derecha
    xS = FIELD_L - 5.5
    y1s = (FIELD_W/2) - 9.16
    y2s = (FIELD_W/2) + 9.16
    area_small = [(FIELD_L,y1s),(xS,y1s),(xS,y2s),(FIELD_L,y2s)]
    pts = proj(area_small)
    cv2.polylines(overlay, [pts], False, (0,255,255), 2)

    # Punto de penalti derecha
    pen = [(FIELD_L-11, FIELD_W/2)]
    pt = proj(pen)[0]
    cv2.circle(overlay, tuple(pt), 6, (0,255,255), -1)

    return overlay

def make_topdown(img, H):
    """Warp a vista cenital en un lienzo 105x68m escalado a píxeles."""
    # Matriz de escala para pasar metros->pixels
    S = np.array([[SCALE, 0, 0],
                  [0, SCALE, 0],
                  [0, 0, 1]], dtype=np.float64)
    Hpx = S @ H  # imagen -> (metros) -> (pixels)
    W = int(FIELD_L * SCALE)
    Hh = int(FIELD_W * SCALE)
    top = cv2.warpPerspective(img, Hpx, (W, Hh))
    return top

def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir {IMG_PATH}")

    pts_img, pts_field, names = read_points(CSV_PATH)

    # Calcular homografía con RANSAC (robusta)
    H, mask = cv2.findHomography(pts_img, pts_field, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if H is None:
        raise RuntimeError("No se pudo calcular la homografía. Revisa los puntos.")

    # Guardar H
    with open(OUT_H_JSON, "w") as f:
        json.dump({"H": H.tolist()}, f, indent=2)

    # Error reproyección (metros)
    err, pts_proj = reprojection_error(H, pts_img, pts_field)

    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(err), dtype=bool)
    err_in = err[inliers] if inliers.any() else err

    print("\n=== Validación numérica (error en metros) ===")
    print(f"Puntos: {len(err)} | Inliers: {inliers.sum()}/{len(err)}")
    print(f"Error medio (inliers):  {err_in.mean():.3f} m")
    print(f"Error mediano (inliers): {np.median(err_in):.3f} m")
    print(f"Error máx (inliers):    {err_in.max():.3f} m\n")

    # Overlay de líneas del campo sobre el frame
    Hinv = np.linalg.inv(H)
    overlay = draw_field_lines_overlay(img, Hinv)

    # Dibujar también los puntos clicados y su proyección en campo (solo para debug)
    for i, (p, name) in enumerate(zip(pts_img.astype(int), names)):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if inliers[i] else (0, 0, 255)
        cv2.circle(overlay, (x, y), 6, color, -1)
        cv2.putText(overlay, str(i+1), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(OUT_OVERLAY, overlay)

    # Vista cenital (top-down)
    topdown = make_topdown(img, H)
    cv2.imwrite(OUT_TOPDOWN, topdown)

    print(f"✅ Guardado: {OUT_H_JSON}")
    print(f"✅ Guardado: {OUT_OVERLAY}")
    print(f"✅ Guardado: {OUT_TOPDOWN}")

if __name__ == "__main__":
    main()
