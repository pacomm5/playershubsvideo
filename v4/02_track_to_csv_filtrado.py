from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2

VIDEO = "match2_clip_1m_60s.mp4"
MODEL = "yolo11l.pt"
IMGSZ = 1280
CONF  = 0.25
IOU   = 0.6

OUT_CSV = "detecciones_track_filtrado.csv"

# HSV del césped (mismos que usaste)
LOW = (35, 40, 40)
HIGH = (90, 255, 255)

MIN_BBOX_H = 25      # filtra personas muy pequeñas (banquillo/gradas)
FOOT_OFFSET = 6      # píxeles por debajo del pie
PATCH = 7            # parche 7x7 para evitar fallos

def mask_cesped(frame_bgr):
    """Máscara de césped por frame (robusta a paneos)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LOW, HIGH)

    kernel = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # quedarnos con el componente más grande (campo)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (labels == largest).astype(np.uint8) * 255

    # rellenar agujeros (jugadores) – igual que en tu opción A
    mb = (m > 0).astype(np.uint8)
    inv = 1 - mb
    ff = inv.copy()
    h, w = ff.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 2)
    holes = (inv == 1) & (ff != 2)
    mb[holes] = 1
    return (mb * 255).astype(np.uint8)

def main():
    model = YOLO(MODEL)
    rows = []
    frame_idx = 0

    results = model.track(
        source=VIDEO,
        tracker="bytetrack_soccer.yaml",
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        classes=[0],      # person
        persist=True,
        stream=True,
        verbose=False,
        save=True         # video anotado en runs/track
    )

    for r in results:
        frame_idx += 1

        # frame original del resultado (Ultralytics)
        frame = r.orig_img
        if frame is None:
            continue

        m = mask_cesped(frame)
        h, w = m.shape[:2]

        boxes = r.boxes
        if boxes is None or boxes.xyxy is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        ids = boxes.id.cpu().numpy() if boxes.id is not None else np.full(len(xyxy), -1)

        kept = 0
        for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
            bbox_h = abs(y2 - y1)
            if bbox_h < MIN_BBOX_H:
                continue

            cx = int(round((x1 + x2) / 2))
            cy = int(round(y2)) + FOOT_OFFSET

            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))

            # mirar parche alrededor del pie (evita fallos por calcetín/bota)
            x0, x1p = max(0, cx - PATCH//2), min(w, cx + PATCH//2 + 1)
            y0, y1p = max(0, cy - PATCH//2), min(h, cy + PATCH//2 + 1)
            if m[y0:y1p, x0:x1p].mean() < 128:
                continue

            rows.append({
                "frame": frame_idx - 1,
                "id": int(tid) if tid is not None else -1,
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "conf": float(c),
            })
            kept += 1

        if frame_idx % 200 == 0:
            print(f"Frames {frame_idx} | filas CSV: {len(rows)} | kept último frame={kept}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Guardado {OUT_CSV} con {len(df)} detecciones")
    print(df.head())

if __name__ == "__main__":
    main()
