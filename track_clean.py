from ultralytics import YOLO
import cv2

model = YOLO("yolo11s.pt")

results = model.track(
    source="match2_clip_1m_60s.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    conf=0.25,
    device="mps",
    classes=[0],
    imgsz=1280
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None

for r in results:
    frame = r.orig_img.copy()

    if writer is None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter("tracking_clean.mp4", fourcc, 25, (w, h))

    if r.boxes:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
        
        for (x1, y1, x2, y2), track_id in zip(boxes, ids):
            # cuadro
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # ID limpio
            cv2.putText(
                frame,
                f"id:{track_id}",
                (int(x1), int(y1)-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

    writer.write(frame)

if writer:
    writer.release()

print("✔ tracking_clean.mp4 generado")
