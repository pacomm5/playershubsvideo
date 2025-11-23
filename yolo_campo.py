from ultralytics import YOLO
import cv2

# 1. Modelo
model = YOLO("yolo11n.pt")
print("Modelo cargado")

# 2. Vídeo de entrada (usa el original de 60s)
INPUT_VIDEO = "VIDEOS/panorama_60s.mp4"
OUTPUT_VIDEO = "VIDEOS/panorama_60s_campo_solo.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"No se puede abrir: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolución: {width}x{height}")

# --- RECORTE BASADO EN TUS LÍNEAS ROJAS ---
# Campo ≈ entre el 12,5% y el 49% de la altura
y1 = int(height * 0.125)   # línea roja superior
y2 = int(height * 0.49)    # línea roja inferior
campo_alto = y2 - y1

print(f"Recorte: y1={y1}, y2={y2} (alto={campo_alto})")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, campo_alto))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # 3. Recortamos SOLO el trozo de campo
    roi = frame[y1:y2, :]

    # 4. Pasamos ese recorte por YOLO
    results = model.predict(
        roi,
        imgsz=1280,
        conf=0.35,
        device="mps",
        verbose=False
    )

    annotated = results[0].plot()
    out.write(annotated)

    if frame_idx % 50 == 0:
        print(f"Frame {frame_idx} procesado...")

cap.release()
out.release()
print(f"✅ Vídeo listo: {OUTPUT_VIDEO}")



