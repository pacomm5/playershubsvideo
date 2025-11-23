from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.track(
    source="VIDEOS/panorama_clip_10s.mp4",
    tracker="bytetrack.yaml",
    save=True,
    conf=0.25,
    device="mps",
    persist=True
)

print("Revisa runs/track/")
