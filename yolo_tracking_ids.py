from ultralytics import YOLO
import cv2

# 1. Cargar modelo
model = YOLO("yolo11n.pt")

# 2. Vídeo de entrada (el de campo recortado)
INPUT_VIDEO = "VIDEOS/panorama_60s_campo_solo.mp4"
OUTPUT_VIDEO = "VIDEOS/panorama_60s_tracking_ids.mp4"

# 3. Usamos track() en modo stream para procesar frame a frame
results_generator = model.track(
    source=INPUT_VIDEO,
    tracker="bytetrack.yaml",
    conf=0.35,
    device="mps",
    stream=True,    # >>> importante: nos da un result por frame
    persist=True    # mantiene el estado del tracker (IDs)
)

writer = None

for i, r in enumerate(results_generator):
    frame = r.orig_img.copy()   # frame original
    boxes = r.boxes

    if writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, r.fps if hasattr(r, "fps") else 25, (w, h))

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()     # [x1, y1, x2, y2]
        confs = boxes.conf.cpu().numpy()    # confianza
        ids = boxes.id                      # IDs del tracker

        if ids is not None:
            ids = ids.cpu().numpy().astype(int)
        else:
            ids = [-1] * len(xyxy)

        for (x1, y1, x2, y2), conf, track_id in zip(xyxy, confs, ids):
            # dibujar caja
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # texto con ID + conf
            label = f"ID {track_id} ({conf:.2f})" if track_id != -1 else f"({conf:.2f})"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    writer.write(frame)

    if (i + 1) % 30 == 0:
        print(f"Procesados {i+1} frames...")

if writer is not None:
    writer.release()

print(f"🎉 Vídeo con IDs guardado en: {OUTPUT_VIDEO}")
