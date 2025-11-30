import cv2
import numpy as np
import pandas as pd
from pathlib import Path

VIDEO_PATH = Path("match2_clip_1m_60s.mp4")
CSV_IN = Path("detecciones_match2_clip.csv")
CSV_OUT = Path("detecciones_match2_color.csv")


def cargar_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    return cap


def get_frame(cap, cache, idx):
    if idx in cache:
        return cache[idx]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    cache[idx] = frame
    return frame


def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"No se encuentra {CSV_IN}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {VIDEO_PATH}")

    df = pd.read_csv(CSV_IN)
    cap = cargar_video(VIDEO_PATH)
    frame_cache = {}

    h_vals, s_vals, v_vals = [], [], []

    for i, row in df.iterrows():
        frame_idx = int(row["frame"])
        frame = get_frame(cap, frame_cache, frame_idx)
        if frame is None:
            h_vals.append(np.nan)
            s_vals.append(np.nan)
            v_vals.append(np.nan)
            continue

        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        patch = frame[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]

        if patch.size == 0:
            h_vals.append(np.nan)
            s_vals.append(np.nan)
            v_vals.append(np.nan)
            continue

        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        ph, pw = patch_hsv.shape[:2]
        ys, ye = int(ph * 0.25), int(ph * 0.75)
        xs, xe = int(pw * 0.25), int(pw * 0.75)
        center = patch_hsv[ys:ye, xs:xe]

        if center.size == 0:
            h_vals.append(np.nan)
            s_vals.append(np.nan)
            v_vals.append(np.nan)
            continue

        mean_hsv = center.reshape(-1, 3).mean(axis=0)
        h_vals.append(mean_hsv[0])
        s_vals.append(mean_hsv[1])
        v_vals.append(mean_hsv[2])

        if i and i % 1000 == 0:
            print(f"Procesadas {i} filas...")

    df["h"] = h_vals
    df["s"] = s_vals
    df["v"] = v_vals

    df.to_csv(CSV_OUT, index=False)
    print(f"Guardado con color en: {CSV_OUT}")


if __name__ == "__main__":
    main()
