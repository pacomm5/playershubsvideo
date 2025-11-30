import csv
from pathlib import Path

PIXEL_CSV = Path("positions.csv")
METROS_CSV = Path("positions_metros.csv")
OUTPUT_CSV = Path("detecciones_match2_clip.csv")


def cargar_metros(path):
    """Devuelve un diccionario {(frame, id): (X_m, Y_m)}."""
    metros = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            pid = int(row["id"])
            X_m = float(row["X_m"])
            Y_m = float(row["Y_m"])
            metros[(frame, pid)] = (X_m, Y_m)
    return metros


def main():
    if not PIXEL_CSV.exists():
        raise FileNotFoundError(f"No se encuentra {PIXEL_CSV}")
    if not METROS_CSV.exists():
        raise FileNotFoundError(f"No se encuentra {METROS_CSV}")

    metros_map = cargar_metros(METROS_CSV)

    out_fields = [
        "frame",
        "id",
        "x1",
        "y1",
        "x2",
        "y2",
        "x_center",
        "y_center",
        "width",
        "height",
        "X_m",
        "Y_m",
    ]

    total = 0
    sin_metros = 0

    with PIXEL_CSV.open(newline="") as fin, OUTPUT_CSV.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            frame = int(row["frame"])
            pid = int(row["id"])
            x_center = float(row["x_center"])
            y_center = float(row["y_center"])
            width = float(row["width"])
            height = float(row["height"])

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            X_m, Y_m = metros_map.get((frame, pid), (None, None))
            if X_m is None:
                sin_metros += 1

            writer.writerow(
                {
                    "frame": frame,
                    "id": pid,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "X_m": X_m if X_m is not None else "",
                    "Y_m": Y_m if Y_m is not None else "",
                }
            )
            total += 1

    print(f"Filas procesadas: {total}")
    print(f"Filas sin coordenadas métricas: {sin_metros}")
    print(f"CSV generado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
