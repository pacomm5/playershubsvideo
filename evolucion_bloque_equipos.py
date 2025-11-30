import pandas as pd
import matplotlib.pyplot as plt

# Configuración
RUTA_CSV = "detecciones_match2_equipos.csv"
X_COL = "X_m"
Y_COL = "Y_m"
TEAM_COL = "team"
FRAME_COL = "frame"
FPS = 25  # ajusta si tu video tiene otro FPS


def main():
    df = pd.read_csv(RUTA_CSV)
    df = df[df[TEAM_COL].isin(["local", "visitante"])].copy()

    grouped = (
        df.groupby([FRAME_COL, TEAM_COL])[[X_COL, Y_COL]]
        .mean()
        .reset_index()
    )
    grouped["segundo"] = grouped[FRAME_COL] / FPS
    grouped["minuto"] = grouped["segundo"] / 60.0

    local = grouped[grouped[TEAM_COL] == "local"]
    visitante = grouped[grouped[TEAM_COL] == "visitante"]

    plt.figure(figsize=(12, 5))
    plt.plot(local["minuto"], local[X_COL], label="local", linewidth=2)
    plt.plot(visitante["minuto"], visitante[X_COL], label="visitante", linewidth=2)
    plt.xlabel("Minuto de partido (aprox.)")
    plt.ylabel("Posición media en longitud X (m)")
    plt.title("Evolución de la altura media del bloque por equipo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("evolucion_bloque_equipos.png", dpi=200)
    print("Gráfico guardado en evolucion_bloque_equipos.png")
    plt.close()


if __name__ == "__main__":
    main()
