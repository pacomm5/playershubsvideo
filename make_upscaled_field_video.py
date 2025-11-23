import cv2

INPUT = "VIDEOS/panorama_60s.mp4"
OUTPUT = "VIDEOS/panorama_60s_upscaled.mp4"

# Recorte vertical del campo (ajusta si hace falta)
y1 = 273
y2 = 800   # o el valor que estés usando ya

# Factor de escala deseado
BASE_SCALE = 2.5

# Ancho máximo permitido para que el códec no se queje
MAX_WIDTH = 3840

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)

writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    field = frame[y1:y2, :]  # recorte del campo
    orig_h, orig_w = field.shape[:2]

    # Calculamos el scale real para no pasarnos de MAX_WIDTH
    scale = min(BASE_SCALE, MAX_WIDTH / orig_w)

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    field_up = cv2.resize(
        field,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    if writer is None:
        print(f"Resolución original: {orig_w} x {orig_h}")
        print(f"Factor escala aplicado: {scale:.2f}")
        print(f"Resolución upscaled: {new_w} x {new_h}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (new_w, new_h))

    writer.write(field_up)

cap.release()
if writer is not None:
    writer.release()

print("Video upscaled guardado en:", OUTPUT)

