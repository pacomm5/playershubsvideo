import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

CSV_IN = "posicion_media_jugadores_agrupado_auto.csv"
SAVE_PATH = "posicion_media_jugadores_agrupado_auto.png"


def draw_pitch(ax, length=105, width=68):
    """Dibuja un campo 105x68 en metros."""
    ax.set_facecolor("#2e7d32")
    ax.add_patch(
        patches.Rectangle((0, 0), length, width, linewidth=2, edgecolor="white", facecolor="none")
    )
    ax.plot([length / 2, length / 2], [0, width], color="white", linewidth=2)
    centre_circle = patches.Circle((length / 2, width / 2), 9.15, linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(centre_circle)
    ax.scatter(length / 2, width / 2, color="white", s=15)
    ax.add_patch(patches.Rectangle((0, (width - 40.32) / 2), 16.5, 40.32, linewidth=2, edgecolor="white", facecolor="none"))
    ax.add_patch(patches.Rectangle((0, (width - 18.32) / 2), 5.5, 18.32, linewidth=2, edgecolor="white", facecolor="none"))
    ax.add_patch(patches.Rectangle((length - 16.5, (width - 40.32) / 2), 16.5, 40.32, linewidth=2, edgecolor="white", facecolor="none"))
    ax.add_patch(patches.Rectangle((length - 5.5, (width - 18.32) / 2), 5.5, 18.32, linewidth=2, edgecolor="white", facecolor="none"))
    ax.scatter(11, width / 2, color="white", s=15)
    ax.scatter(length - 11, width / 2, color="white", s=15)
    ax.add_patch(patches.Arc((11, width / 2), 18.3, 18.3, angle=0, theta1=310, theta2=50, linewidth=2, color="white"))
    ax.add_patch(patches.Arc((length - 11, width / 2), 18.3, 18.3, angle=0, theta1=130, theta2=230, linewidth=2, color="white"))
    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("Longitud (m)")
    ax.set_ylabel("Anchura (m)")


def main():
    df = pd.read_csv(CSV_IN)
    required = {"team", "player_cluster", "X_m", "Y_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas en {CSV_IN}, se esperan: {required}")

    # Un punto por jugador real: media por cluster y equipo
    df = (
        df.groupby(["team", "player_cluster"])[["X_m", "Y_m"]]
        .mean()
        .reset_index()
    )

    # Filtrar a los que caen dentro del campo (0-105, 0-68) para evitar puntos fuera de vista
    length, width = 105, 68
    in_bounds = df[
        (df["X_m"].between(0, length))
        & (df["Y_m"].between(0, width))
    ]
    print("Clusters totales por equipo:")
    print(df.groupby("team")["player_cluster"].nunique())
    print("Clusters dentro del campo (visibles) por equipo:")
    print(in_bounds.groupby("team")["player_cluster"].nunique())
    df = in_bounds

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax)

    color_map = {"local": "blue", "visitante": "red"}

    for _, row in df.iterrows():
        x, y, team, pid = row["X_m"], row["Y_m"], row["team"], row["player_cluster"]
        color = color_map.get(team, "yellow")
        ax.scatter(x, y, s=70, color=color)
        ax.text(x + 0.5, y + 0.5, str(pid), fontsize=8, color="white")

    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="", label="local", markerfacecolor="blue"),
        Line2D([0], [0], marker="o", linestyle="", label="visitante", markerfacecolor="red"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title("Posición media por jugador (IDs agrupados)")
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200)
    print(f"Figura guardada en {SAVE_PATH}")
    # plt.show()  # descomenta si quieres ver la ventana interactiva


if __name__ == "__main__":
    main()
