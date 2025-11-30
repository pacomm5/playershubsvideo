import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- Configuración ----------
RUTA_CSV = "detecciones_match2_equipos.csv"
X_COL = "X_m"
Y_COL = "Y_m"
TEAM_COL = "team"
LARGO = 105
ANCHO = 68
NBINS_X = 50
NBINS_Y = 32
SAVE_PATH = "heatmap_equipos.png"


def dibujar_campo(ax):
    """Dibuja un campo de fútbol en el eje ax."""
    ax.set_facecolor("#2e7d32")
    campo = patches.Rectangle((0, 0), LARGO, ANCHO,
                              linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(campo)
    ax.plot([0, LARGO, LARGO, 0, 0],
            [0, 0, ANCHO, ANCHO, 0], color="white", linewidth=2)
    ax.plot([LARGO / 2, LARGO / 2], [0, ANCHO], color="white", linewidth=2)
    circulo_centro = plt.Circle((LARGO / 2, ANCHO / 2), 9.15,
                                color="white", fill=False, linewidth=2)
    ax.add_artist(circulo_centro)
    ax.scatter([LARGO / 2], [ANCHO / 2], color="white", s=15)
    area_ancho = 40.3
    area_largo = 16.5
    ax.add_patch(patches.Rectangle((0, (ANCHO - area_ancho) / 2), area_largo, area_ancho,
                                   linewidth=2, edgecolor="white", facecolor="none"))
    ax.add_patch(patches.Rectangle((LARGO - area_largo, (ANCHO - area_ancho) / 2),
                                   area_largo, area_ancho,
                                   linewidth=2, edgecolor="white", facecolor="none"))
    ax.set_xlim(0, LARGO)
    ax.set_ylim(ANCHO, 0)  # mismo orden que en los otros gráficos
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud (m)")
    ax.set_ylabel("Anchura (m)")


def main():
    df = pd.read_csv(RUTA_CSV)
    df = df[df[TEAM_COL].isin(["local", "visitante"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, team in zip(axes, ["local", "visitante"]):
        df_team = df[df[TEAM_COL] == team]
        x = df_team[X_COL].values
        y = df_team[Y_COL].values

        heatmap, _, _ = np.histogram2d(
            x, y,
            bins=[NBINS_X, NBINS_Y],
            range=[[0, LARGO], [0, ANCHO]]
        )
        heatmap = heatmap.T  # imshow espera [y, x]

        dibujar_campo(ax)
        im = ax.imshow(
            heatmap,
            extent=[0, LARGO, ANCHO, 0],
            origin="upper",
            alpha=0.75,
            cmap="hot",
            zorder=2
        )
        ax.set_title(f"Mapa de calor – {team}")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Frecuencia de presencia")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200)
    print(f"Heatmap guardado en {SAVE_PATH}")
    # plt.show()  # descomenta si prefieres ver la ventana interactiva


if __name__ == "__main__":
    main()
