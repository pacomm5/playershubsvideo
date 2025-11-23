from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.track(
    source="VIDEOS/panorama_60s_campo_solo.mp4",
    tracker="bytetrack.yaml",
    save=True,
    conf=0.35,
    device="mps",
    persist=True,   # mantiene el estado del tracker
    verbose=True
)

print("Listo. Mira en la carpeta runs/track/")

