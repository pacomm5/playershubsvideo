import pandas as pd

CSV_IN = "detecciones_match2_equipos.csv"
CSV_OUT = "posicion_media_por_jugador_y_equipo.csv"


def main():
    df = pd.read_csv(CSV_IN)
    df = df[df["team"].isin(["local", "visitante"])]

    pos_jugadores = (
        df.groupby(["team", "id"])[["X_m", "Y_m"]]
        .mean()
        .reset_index()
    )

    print("Posición media por jugador y equipo:")
    print(pos_jugadores.head())

    pos_jugadores.to_csv(CSV_OUT, index=False)
    print(f"Guardado en {CSV_OUT}")


if __name__ == "__main__":
    main()
