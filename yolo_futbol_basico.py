from ultralytics import YOLO

# Intentamos primero YOLO11 (nuevo); si falla, YOLOv8
try:
    model = YOLO("yolo11n.pt")   # modelo ligero
    print("Cargado modelo yolo11n.pt")
except Exception as e:
    print("No se pudo cargar yolo11n.pt, probando con yolov8n.pt")
    model = YOLO("yolov8n.pt")
    print("Cargado modelo yolov8n.pt")

# Ruta del vídeo desde la carpeta del proyecto
VIDEO = "VIDEOS/panorama_60s.mp4"

# Ejecutar la predicción sobre el vídeo
results = model.predict(
    source=VIDEO,
    save=True,       # guarda un vídeo nuevo con cajas
    save_txt=False,  # de momento sin labels en txt
    conf=0.35,       # umbral de confianza
    stream=False
)

print("✅ Listo. Revisa la carpeta runs/detect para ver el vídeo procesado.")
