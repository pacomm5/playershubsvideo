# tracking_upscaled_vertical.py
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.track(
    source="VIDEOS/panorama_60s_upscaled_vertical.mp4",
    tracker="bytetrack.yaml",
    save=True,
    conf=0.25,
    device="mps",
    persist=True
)

print("Revisa runs/track/")
