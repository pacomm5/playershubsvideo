import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter

# Configuración
RUTA_CSV = "detecciones_match2_equipos.csv"
X_COL = "X_m"
Y_COL = "Y_m"
TEAM_COL = "team"

LARGO = 105
ANCHO = 68
NBINS_X = 100  # resolución en eje largo
NBINS_Y = 65   # resolución en eje ancho
SIGMA = 1.5    # suavidad gaussiana (prueba 1.5–3)
CMAP = "inferno"  # opciones: "hot", "magma", "turbo", etc.
SAVE_PATH = "heatmap_equipos_suave.png"


def dibujar_campo(ax):
    """Dibuja el campo con líneas básicas."""
    ax.set_facecolor("#2e7d32")
    ax.add_patch(
        patches.Rectangle((0, 0), LARGO, ANCHO, linewidth=2, edgecolor="white", facecolor="none")
    )
    ax.plot([0, LARGO, LARGO, 0, 0], [0, 0, ANCHO, ANCHO, 0], color="white", linewidth=2)
    ax.plot([LARGO / 2, LARGO / 2], [0, ANCHO], color="white", linewidth=2)
    circulo = plt.Circle((LARGO / 2, ANCHO / 2), 9.15, color="white", fill=False, linewidth=2)
    ax.add_artist(circulo)
    ax.scatter([LARGO / 2], [ANCHO / 2], color="white", s=15)
    area_w, area_l = 40.3, 16.5
    ax.add_patch(
        patches.Rectangle((0, (ANCHO - area_w) / 2), area_l, area_w, linewidth=2, edgecolor="white", facecolor="none")
    )
    ax.add_patch(
        patches.Rectangle((LARGO - area_l, (ANCHO - area_w) / 2), area_l, area_w, linewidth=2, edgecolor="white", facecolor="none")
    )
    ax.set_xlim(0, LARGO)
    ax.set_ylim(ANCHO, 0)  # mantiene la orientación usada en otros gráficos
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud (m)")
    ax.set_ylabel("Anchura (m)")


def main():
    df = pd.read_csv(RUTA_CSV)
    df = df[df[TEAM_COL].isin(["local", "visitante"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, team in zip(axes, ["local", "visitante"]):
        df_team = df[df[TEAM_COL] == team]
        x = df_team[X_COL].values
        y = df_team[Y_COL].values

        heatmap, _, _ = np.histogram2d(
            x, y,
            bins=[NBINS_X, NBINS_Y],
            range=[[0, LARGO], [0, ANCHO]]
        )
        heatmap_suave = gaussian_filter(heatmap.T, sigma=SIGMA)

        dibujar_campo(ax)
        im = ax.imshow(
            heatmap_suave,
            extent=[0, LARGO, ANCHO, 0],
            cmap=CMAP,
            alpha=0.7,
            interpolation="bilinear",
            origin="upper",
            zorder=2
        )
        ax.set_title(f"Mapa de calor (suave) – {team}")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("Intensidad de presencia")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200)
    print(f"Heatmap suave guardado en {SAVE_PATH}")
    # plt.show()  # descomenta si quieres ver la ventana interactiva


if __name__ == "__main__":
    main()
