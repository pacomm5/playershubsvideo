import pandas as pd

CSV_DETECCIONES = "detecciones_match2_equipos.csv"
CSV_OUT = "posicion_media_jugador_filtrado.csv"

# Umbral mínimo de detecciones por track para conservarlo
MIN_FRAMES = 150


def main():
    df = pd.read_csv(CSV_DETECCIONES)
    df = df[df["team"].isin(["local", "visitante"])]

    # Contar detecciones por equipo e id
    counts = df.groupby(["team", "id"]).size().reset_index(name="n")
    ids_validos = counts[counts["n"] >= MIN_FRAMES][["team", "id"]]

    if ids_validos.empty:
        raise RuntimeError(f"No hay IDs con al menos {MIN_FRAMES} detecciones")

    df_filtrado = df.merge(ids_validos, on=["team", "id"], how="inner")

    # Posición media por track válido
    pos_filtrado = (
        df_filtrado.groupby(["team", "id"])[["X_m", "Y_m"]]
        .mean()
        .reset_index()
    )

    pos_filtrado.to_csv(CSV_OUT, index=False)
    print(f"Tracks totales: {counts.shape[0]}")
    print(f"Tracks válidos (>= {MIN_FRAMES} detecciones): {ids_validos.shape[0]}")
    print(f"Posiciones medias guardadas en: {CSV_OUT}")


if __name__ == "__main__":
    main()
