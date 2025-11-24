import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= CONFIGURACIÓN =================
CSV_FILE = "positions_metros.csv"
PLAYER_ID = 18       # <- CAMBIA AQUÍ EL JUGADOR
BINS_X = 30          # resolución heatmap en eje largo
BINS_Y = 20          # resolución heatmap en eje ancho
SAVE_FIG = True      # True para guardar PNG, False solo mostrar
# =================================================


def draw_pitch(ax, length=105, width=68):
    """
    Dibuja un campo de fútbol estándar 105x68 en coordenadas métricas.
    """
    # Fondo verde
    ax.set_facecolor("#3f9243")

    # Límites del campo
    pitch = patches.Rectangle((0, 0), length, width,
                              linewidth=2, edgecolor="white",
                              facecolor="none")
    ax.add_patch(pitch)

    # Línea de medio campo
    ax.plot([length/2, length/2], [0, width], color="white", linewidth=2)

    # Círculo central
    centre_circle = patches.Circle((length/2, width/2), 9.15,
                                   linewidth=2, edgecolor="white",
                                   facecolor="none")
    ax.add_patch(centre_circle)

    # Punto central
    ax.scatter(length/2, width/2, color="white", s=15)

    # Áreas y áreas pequeñas
    # Área izquierda
    ax.add_patch(patches.Rectangle((0, (width-40.32)/2),
                                   16.5, 40.32,
                                   linewidth=2, edgecolor="white",
                                   facecolor="none"))
    # Área pequeña izquierda
    ax.add_patch(patches.Rectangle((0, (width-18.32)/2),
                                   5.5, 18.32,
                                   linewidth=2, edgecolor="white",
                                   facecolor="none"))

    # Área derecha
    ax.add_patch(patches.Rectangle((length-16.5, (width-40.32)/2),
                                   16.5, 40.32,
                                   linewidth=2, edgecolor="white",
                                   facecolor="none"))
    # Área pequeña derecha
    ax.add_patch(patches.Rectangle((length-5.5, (width-18.32)/2),
                                   5.5, 18.32,
                                   linewidth=2, edgecolor="white",
                                   facecolor="none"))

    # Puntos de penalti
    ax.scatter(11, width/2, color="white", s=15)
    ax.scatter(length-11, width/2, color="white", s=15)

    # Arcos de área (simplificados como segmentos de círculo)
    left_arc = patches.Arc((11, width/2), 18.3, 18.3,
                           angle=0, theta1=310, theta2=50,
                           linewidth=2, color="white")
    right_arc = patches.Arc((length-11, width/2), 18.3, 18.3,
                            angle=0, theta1=130, theta2=230,
                            linewidth=2, color="white")
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # Ejes
    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # opcional: para que se vea como en TV (arriba=portería rival)
    ax.set_xlabel("Longitud (m)")
    ax.set_ylabel("Anchura (m)")


def main():
    # Cargar datos
    df = pd.read_csv(CSV_FILE)

    # Filtrar jugador
    player = df[df["id"] == PLAYER_ID]
    if player.empty:
        print(f"No hay datos para el jugador ID {PLAYER_ID}")
        return

    print(f"Frames para el jugador {PLAYER_ID}: {len(player)}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1) Dibujar el campo
    draw_pitch(ax)

    # 2) Heatmap encima del campo
    h = ax.hist2d(
        player["X_m"],
        player["Y_m"],
        bins=[BINS_X, BINS_Y],
        range=[[0, 105], [0, 68]],
        cmap="hot",
        alpha=0.7
    )

    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label("Densidad de presencia")

    ax.set_title(f"Heatmap jugador ID {PLAYER_ID}")

    plt.tight_layout()

    if SAVE_FIG:
        out_name = f"heatmap_id_{PLAYER_ID}.png"
        plt.savefig(out_name, dpi=200)
        print(f"✅ Imagen guardada como {out_name}")

    plt.show()


if __name__ == "__main__":
    main()
