"""
Pipeline automático:
- Filtra tracks cortos.
- Busca un eps adecuado por equipo para agrupar IDs cercanos con DBSCAN.
- Genera posiciones limpias por jugador (cluster) sin tocar los CSV originales.
"""
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path

CSV_DETECCIONES = Path("detecciones_match2_equipos.csv")

MIN_FRAMES = 120  # mínimo de detecciones por track para conservarlo
TARGET_PLAYERS = 11  # jugadores esperados por equipo
# Valores pequeños separan más (más clusters), grandes fusionan (menos clusters)
EPS_CANDIDATES = [
    1.5, 2, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3,
    3.5, 4, 5, 6, 8, 10, 12, 15
]  # en metros

# Dimensiones del campo para limitar coordenadas
FIELD_LENGTH = 105
FIELD_WIDTH = 68

CSV_TRACKS_FILTRADOS = Path("posicion_media_jugador_filtrado_auto.csv")
CSV_AGRUPADO = Path("posicion_media_jugadores_agrupado_auto.csv")


def filtrar_tracks(df):
    """Filtra tracks con al menos MIN_FRAMES detecciones."""
    df = df[df["team"].isin(["local", "visitante"])].copy()
    # Clip de coordenadas al campo para evitar medias fuera de rango
    df["X_m"] = df["X_m"].clip(0, FIELD_LENGTH)
    df["Y_m"] = df["Y_m"].clip(0, FIELD_WIDTH)
    counts = df.groupby(["team", "id"]).size().reset_index(name="n")
    ids_validos = counts[counts["n"] >= MIN_FRAMES][["team", "id"]]
    if ids_validos.empty:
        raise RuntimeError(f"No hay IDs con al menos {MIN_FRAMES} detecciones")

    df_filtrado = df.merge(ids_validos, on=["team", "id"], how="inner")
    pos_filtrado = (
        df_filtrado.groupby(["team", "id"])[["X_m", "Y_m"]]
        .mean()
        .reset_index()
    )
    pos_filtrado.to_csv(CSV_TRACKS_FILTRADOS, index=False)
    print(f"Tracks totales: {counts.shape[0]}")
    print(f"Tracks válidos (>= {MIN_FRAMES} detecciones): {ids_validos.shape[0]}")
    print(f"Posición media por track guardada en: {CSV_TRACKS_FILTRADOS}")
    return pos_filtrado


def elegir_eps(df_team):
    """Devuelve el eps cuyo número de clusters se acerca más al objetivo."""
    from math import inf

    best_eps = EPS_CANDIDATES[0]
    best_diff = inf
    best_clusters = None

    coords = df_team[["X_m", "Y_m"]].values
    for eps in EPS_CANDIDATES:
        clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
        n_clusters = len(set(clustering.labels_))
        diff = abs(n_clusters - TARGET_PLAYERS)
        # Preferimos el más cercano a 11; si empata, preferimos el que no se quede por debajo
        if diff < best_diff or (
            diff == best_diff and n_clusters >= TARGET_PLAYERS and (best_clusters is None or best_clusters < TARGET_PLAYERS)
        ):
            best_diff = diff
            best_eps = eps
            best_clusters = n_clusters

    print(f"Equipo {df_team.iloc[0]['team']}: mejor eps={best_eps} (clusters={best_clusters})")
    return best_eps


def agrupar_clusters(pos_filtrado):
    """Aplica DBSCAN por equipo con eps auto-seleccionado y guarda CSV final."""
    resultados = []
    for team in pos_filtrado["team"].unique():
        df_team = pos_filtrado[pos_filtrado["team"] == team].copy()
        if df_team.empty:
            continue
        eps = elegir_eps(df_team)
        clustering = DBSCAN(eps=eps, min_samples=1).fit(df_team[["X_m", "Y_m"]].values)
        df_team["player_cluster"] = clustering.labels_.astype(int)
        # Prefijo para que no choquen IDs entre equipos
        df_team["player_cluster"] = df_team["player_cluster"].apply(lambda c: f"{team}_{c}")
        resultados.append(df_team)

    if not resultados:
        raise RuntimeError("No hay datos para agrupar.")

    df_final = pd.concat(resultados, ignore_index=True)
    df_final.to_csv(CSV_AGRUPADO, index=False)
    print(df_final.groupby("team")["player_cluster"].nunique())
    print(f"Clusters por jugador guardados en: {CSV_AGRUPADO}")
    return df_final


def main():
    if not CSV_DETECCIONES.exists():
        raise FileNotFoundError(f"No se encuentra {CSV_DETECCIONES}")

    df = pd.read_csv(CSV_DETECCIONES)
    pos_filtrado = filtrar_tracks(df)
    agrupar_clusters(pos_filtrado)


if __name__ == "__main__":
    main()
