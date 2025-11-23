import cv2

INPUT = "VIDEOS/panorama_60s.mp4"
OUTPUT = "VIDEOS/panorama_60s_upscaled_vertical.mp4"

# En vez de y1=273, y2=800 usamos porcentajes del alto completo
TOP_PCT = 0.18   # 18% desde arriba (menos techo)
BOT_PCT = 0.60   # 96% hasta abajo (dejamos casi todo el campo)

VERT_SCALE = 3.0
MAX_WIDTH = 3840

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)

writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    y1 = int(h * TOP_PCT)
    y2 = int(h * BOT_PCT)

    field = frame[y1:y2, :]          # recorte del campo
    orig_h, orig_w = field.shape[:2]

    new_h = int(orig_h * VERT_SCALE)
    new_w = min(orig_w, MAX_WIDTH)

    field_up = cv2.resize(field, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if writer is None:
        print("Original:", orig_w, "x", orig_h)
        print("Upscaled vertical:", new_w, "x", new_h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (new_w, new_h))

    writer.write(field_up)

cap.release()
if writer:
    writer.release()

print("Vídeo vertical upscaled guardado en:", OUTPUT)

