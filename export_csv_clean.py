from ultralytics import YOLO
import csv

model = YOLO("yolo11s.pt")

results = model.track(
    source="match2_clip_1m_60s.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save=False,
    stream=True,
    device="mps",
    classes=[0],
    imgsz=1280
)

with open("positions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "id", "x_center", "y_center", "width", "height"])

    for frame_i, r in enumerate(results):
        if not r.boxes:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        
        for (x1, y1, x2, y2), track_id in zip(xyxy, ids):
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w/2
            cy = y1 + h/2
            writer.writerow([frame_i, track_id, cx, cy, w, h])

print("✔ positions.csv generado")
