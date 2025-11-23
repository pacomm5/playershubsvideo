from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Muy bueno para fútbol

model.predict(
    source="match2_clip_1m_60s.mp4",
    save=True,
    conf=0.35,
    device="mps",   # GPU del Mac M1/M2
    imgsz=1280      # resolución interna más alta = mejor detección
)

print("✔ Detección lista en runs/detect/")
