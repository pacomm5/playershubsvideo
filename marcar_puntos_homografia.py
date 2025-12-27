import sys
import cv2
import pandas as pd

# -------------------------------------------------------------------
# LISTA DE PUNTOS (edítala según el frame)
# -------------------------------------------------------------------
POINTS = [
    ("P1_centro_circulo", 52.5, 34.0),
    ("P2_circulo_arriba_interseccion_medio", 52.5, 24.85),
    ("P3_circulo_abajo_interseccion_medio", 52.5, 43.15),

    # NUEVOS para evitar colinealidad en H0/H1:
    ("P4_circulo_izquierda_extremo", 43.35, 34.0),   # 52.5 - 9.15
    ("P5_circulo_derecha_extremo", 61.65, 34.0),     # 52.5 + 9.15

    ("P6_medio_campo_con_banda_superior", 52.5, 0.0),
    ("P7_medio_campo_con_banda_inferior", 52.5, 68.0),

    # Área derecha (solo si se ve; si no, SALTA con N)
    ("P8_punto_penalti_derecha", 94.0, 34.0),
    ("P9_area_grande_dcha_esquina_interior_sup", 88.5, 13.85),
    ("P10_area_grande_dcha_esquina_interior_inf", 88.5, 54.15),
    ("P11_area_pequena_dcha_esquina_interior_sup", 99.5, 24.84),
    ("P12_area_pequena_dcha_esquina_interior_inf", 99.5, 43.16),
]

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "IMAGEN PARA ANALISIS PRO.png"

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

    h, w = img.shape[:2]
    max_w, max_h = 1600, 900
    scale = min(1.0, max_w / w, max_h / h)

    disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    base = disp.copy()

    idx = 0
    actions = []  # historial: ("click", row) o ("skip", point)
    rows = []

    win = "Marcar homografia | Click=marcar | S=guardar | N=saltar | U=deshacer | ESC=salir"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def redraw():
        disp[:] = base[:]

        # dibuja puntos ya marcados
        for i, r in enumerate(rows, start=1):
            x_d = int(r["x_img"] * scale)
            y_d = int(r["y_img"] * scale)
            cv2.circle(disp, (x_d, y_d), 6, (0, 255, 255), -1)
            cv2.putText(disp, str(i), (x_d + 8, y_d - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # overlay texto arriba
        if idx < len(POINTS):
            name, xf, yf = POINTS[idx]
            txt = f"SIGUIENTE: {name} -> campo({xf:.2f},{yf:.2f})  [{idx+1}/{len(POINTS)}]   (N para saltar)"
        else:
            txt = "FIN LISTA. Pulsa S para guardar o ESC para salir."

        cv2.rectangle(disp, (0, 0), (disp.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(disp, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def click(event, x, y, flags, param):
        nonlocal idx
        if event == cv2.EVENT_LBUTTONDOWN and idx < len(POINTS):
            x_img = int(round(x / scale))
            y_img = int(round(y / scale))

            name, x_field, y_field = POINTS[idx]
            row = {
                "x_img": x_img,
                "y_img": y_img,
                "x_field": float(x_field),
                "y_field": float(y_field),
                "name": name
            }
            rows.append(row)
            actions.append(("click", row))
            print(f"[CLICK {idx+1:02d}] {name}: x_img={x_img}, y_img={y_img} -> ({x_field},{y_field})")
            idx += 1

    cv2.setMouseCallback(win, click)

    while True:
        redraw()
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            break

        if key in (ord('n'), ord('N')) and idx < len(POINTS):
            name, xf, yf = POINTS[idx]
            actions.append(("skip", (name, xf, yf)))
            print(f"[SKIP  {idx+1:02d}] {name} (no visible)")
            idx += 1

        if key in (ord('u'), ord('U')) and actions:
            act, payload = actions.pop()
            if act == "click":
                if rows and rows[-1]["name"] == payload["name"]:
                    rows.pop()
                idx = max(0, idx - 1)
                print("[UNDO] deshecho click")
            elif act == "skip":
                idx = max(0, idx - 1)
                print("[UNDO] deshecho skip")

        if key in (ord('s'), ord('S')):
            if len(rows) < 4:
                print("⚠️ Necesitas al menos 4 puntos para guardar.")
                continue
            df = pd.DataFrame(rows)[["x_img", "y_img", "x_field", "y_field", "name"]]
            df.to_csv("puntos_homografia.csv", index=False)
            cv2.imwrite("puntos_homografia_marcados.png", disp)
            print("✅ Guardado: puntos_homografia.csv")
            print("✅ Guardado: puntos_homografia_marcados.png")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
