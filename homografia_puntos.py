import cv2
import numpy as np

VIDEO = "match2_clip_1m_60s.mp4"

# Abrimos el vídeo y cogemos el primer frame
cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("No se ha podido leer el primer frame del vídeo")

puntos = []

def click_event(event, x, y, flags, param):
    global puntos, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        print(f"Punto {len(puntos)}: ({x}, {y})")
        # Dibujamos un círculo y el índice
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(len(puntos)), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Selecciona 4 puntos del campo", frame)

# Mostramos la imagen y esperamos 4 clics
cv2.namedWindow("Selecciona 4 puntos del campo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Selecciona 4 puntos del campo", 960, 540)
cv2.setMouseCallback("Selecciona 4 puntos del campo", click_event)

print("👉 Haz clic en este orden EXACTO sobre el campo:")
print("1) ESQUINA INFERIOR IZQUIERDA")
print("2) ESQUINA INFERIOR DERECHA")
print("3) ESQUINA SUPERIOR DERECHA")
print("4) ESQUINA SUPERIOR IZQUIERDA")

while True:
    cv2.imshow("Selecciona 4 puntos del campo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break
    if len(puntos) == 4:
        print("✅ Puntos seleccionados:", puntos)
        break

cv2.destroyAllWindows()

# Guardamos los puntos en un .npy para usarlos después
puntos_np = np.array(puntos, dtype=np.float32)
np.save("puntos_campo.npy", puntos_np)
print("Puntos guardados en puntos_campo.npy")
