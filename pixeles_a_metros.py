import numpy as np
import csv

H = np.load("H_campo.npy")
print("Homografía cargada:")
print(H)

INPUT_CSV = "positions.csv"
OUTPUT_CSV = "positions_metros.csv"

def pixel_to_meters(x, y, H):
    pt = np.array([x, y, 1.0])
    XM = H @ pt
    X = XM[0] / XM[2]
    Y = XM[1] / XM[2]
    return X, Y

with open(INPUT_CSV, "r") as fin, open(OUTPUT_CSV, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    fieldnames = ["frame", "id", "X_m", "Y_m"]
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        frame = int(row["frame"])
        track_id = int(row["id"])
        cx = float(row["x_center"])
        cy = float(row["y_center"])

        X_m, Y_m = pixel_to_meters(cx, cy, H)
        writer.writerow({
            "frame": frame,
            "id": track_id,
            "X_m": X_m,
            "Y_m": Y_m
        })

print("✅ positions_metros.csv generado")
