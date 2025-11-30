import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path

CSV_IN = Path("detecciones_match2_color.csv")
CSV_OUT = Path("detecciones_match2_equipos.csv")
CSV_POS_MEDIA = Path("posicion_media_por_equipo_nuevo.csv")

# Ajusta estos labels tras ver los centroides impresos
MAP_CLUSTER_A_TEAM = {
    0: "local",
    1: "visitante",
    2: "otros",
}


def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"No se encuentra {CSV_IN}")

    df = pd.read_csv(CSV_IN)
    df_valid = df.dropna(subset=["h", "s", "v"]).copy()
    if df_valid.empty:
        raise RuntimeError("No hay detecciones con color válido para clusterizar")

    X = df_valid[["h", "s", "v"]].values
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    df_valid["cluster_color"] = kmeans.fit_predict(X)

    print("Centroides HSV por cluster:")
    for i, center in enumerate(kmeans.cluster_centers_):
        h, s, v = center
        print(f"Cluster {i}: H={h:.1f}, S={s:.1f}, V={v:.1f}")

    df_valid["team"] = df_valid["cluster_color"].map(MAP_CLUSTER_A_TEAM)

    df_out = df.merge(
        df_valid[["frame", "id", "cluster_color", "team"]],
        on=["frame", "id"],
        how="left",
    )

    df_out.to_csv(CSV_OUT, index=False)
    print(f"CSV con equipos: {CSV_OUT}")

    df_teams = df_out[df_out["team"].isin(["local", "visitante"])]
    if df_teams.empty:
        print("No hay filas con team asignado a local/visitante. Ajusta MAP_CLUSTER_A_TEAM.")
        return

    pos_media = (
        df_teams.groupby("team")[["X_m", "Y_m"]]
        .mean()
        .reset_index()
    )
    pos_media.to_csv(CSV_POS_MEDIA, index=False)
    print(pos_media)
    print(f"Posición media guardada en: {CSV_POS_MEDIA}")


if __name__ == "__main__":
    main()
