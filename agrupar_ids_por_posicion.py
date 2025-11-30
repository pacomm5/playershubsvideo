import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

CSV_IN = "posicion_media_jugador_filtrado.csv"
CSV_OUT = "posicion_media_jugadores_agrupado.csv"

# Distancia máxima (en metros) para agrupar IDs del mismo jugador
EPS_METROS = 3.0


def cluster_team(df_team, eps):
    coords = df_team[["X_m", "Y_m"]].values
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
    df_team = df_team.copy()
    df_team["player_cluster"] = clustering.labels_
    return df_team


def main():
    df = pd.read_csv(CSV_IN)
    required = {"team", "id", "X_m", "Y_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas en {CSV_IN}, se esperan: {required}")

    resultados = []
    for team in df["team"].unique():
        df_team = df[df["team"] == team]
        if df_team.empty:
            continue
        clustered = cluster_team(df_team, EPS_METROS)
        # Prefija con el nombre del equipo para que no choquen clusters entre equipos
        clustered["player_cluster"] = clustered["player_cluster"].apply(lambda c: f"{team}_{c}")
        resultados.append(clustered)

    if not resultados:
        raise RuntimeError("No hay datos para agrupar.")

    df_final = pd.concat(resultados, ignore_index=True)
    df_final.to_csv(CSV_OUT, index=False)
    print(f"Guardado agrupado en: {CSV_OUT}")
    print(df_final.head())


if __name__ == "__main__":
    main()
