from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.track(
    source="match2_clip_1m_60s.mp4",
    tracker="bytetrack.yaml",
    save=True,       # guarda vídeo con tracking
    persist=True,
    classes=[0],     # solo personas
    conf=0.25,
    device="mps",
    imgsz=1280
)

print("✔ Tracking listo en runs/track/")

