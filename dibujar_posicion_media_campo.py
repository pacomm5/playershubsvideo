import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CSV_POS_MEDIA = "posicion_media_por_equipo_nuevo.csv"


def draw_pitch(ax, length=105, width=68):
    """Dibuja un campo 105x68 en metros."""
    ax.set_facecolor("#3f9243")
    ax.add_patch(
        patches.Rectangle((0, 0), length, width, linewidth=2, edgecolor="white", facecolor="none")
    )
    ax.plot([length / 2, length / 2], [0, width], color="white", linewidth=2)
    centre_circle = patches.Circle((length / 2, width / 2), 9.15, linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(centre_circle)
    ax.scatter(length / 2, width / 2, color="white", s=15)
    # Áreas grandes y pequeñas
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
    df = pd.read_csv(CSV_POS_MEDIA)
    required = {"team", "X_m", "Y_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas en {CSV_POS_MEDIA}, se esperan: {required}")

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax)

    # Colores simples por equipo; ajusta si usas otros nombres
    color_map = {"local": "blue", "visitante": "red"}

    for _, row in df.iterrows():
        x, y, team = row["X_m"], row["Y_m"], row["team"]
        color = color_map.get(team, "yellow")
        ax.scatter(x, y, color=color, s=80, label=team)
        ax.text(x + 1, y + 1, team, fontsize=10, color=color)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right")
    ax.set_title("Posición media por equipo")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
