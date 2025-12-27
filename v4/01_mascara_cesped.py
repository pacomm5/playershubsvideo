import cv2
import numpy as np

VIDEO = "match2_clip_1m_60s.mp4"
OUT_MASK = "mask_cesped.png"

LOW = (35, 40, 40)
HIGH = (90, 255, 255)

cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError("No se pudo leer el primer frame del vídeo")

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, LOW, HIGH)

# limpiar ruido
kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# quedarnos con el componente más grande (campo)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255

# --- RELLENAR AGUJEROS (jugadores) ---
mask_bin = (mask > 0).astype(np.uint8)   # 1 = césped, 0 = resto
inv = 1 - mask_bin

ff = inv.copy()
h, w = ff.shape
ff_mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(ff, ff_mask, (0, 0), 2)

holes = (inv == 1) & (ff != 2)
mask_bin[holes] = 1
mask = (mask_bin * 255).astype(np.uint8)

cv2.imwrite(OUT_MASK, mask)
print("✅ Guardada máscara:", OUT_MASK)
