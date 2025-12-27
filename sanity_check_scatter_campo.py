import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CSV_IN = "detecciones_match2_equipos_reproj.csv"

# Columnas reproyectadas
X_COL = "X_m_new"
Y_COL = "Y_m_new"
TEAM_COL = "team"   # si existe

LARGO = 105
ANCHO = 68

def dibujar_campo(ax):
    # Fondo campo
    campo = patches.Rectangle((0, 0), LARGO, ANCHO,
                              linewidth=2, edgecolor="white",
                              facecolor="#2e7d32")
    ax.add_patch(campo)

    # Límites
    ax.plot([0, LARGO, LARGO, 0, 0],
            [0, 0, ANCHO, ANCHO, 0], color="white", linewidth=2)

    # Medio campo
    ax.plot([LARGO/2, LARGO/2], [0, ANCHO], color="white", linewidth=2)

    # Círculo central
    centro = plt.Circle((LARGO/2, ANCHO/2), 9.15, color="white", fill=False, linewidth=2)
    ax.add_artist(centro)
    ax.scatter([LARGO/2], [ANCHO/2], color="white", s=15)

    # Áreas
    area_w = 40.3
    area_l = 16.5
    ax.add_patch(patches.Rectangle((0, (ANCHO-area_w)/2),
                                   area_l, area_w, linewidth=2,
                                   edgecolor="white", fill=False))
    ax.add_patch(patches.Rectangle((LARGO-area_l, (ANCHO-area_w)/2),
                                   area_l, area_w, linewidth=2,
                                   edgecolor="white", fill=False))

    # Ajustes
    ax.set_xlim(0, LARGO)
    ax.set_ylim(ANCHO, 0)   # misma convención que tus gráficos anteriores
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud (m)")
    ax.set_ylabel("Anchura (m)")

def main():
    df = pd.read_csv(CSV_IN)

    fig, ax = plt.subplots(figsize=(12, 7))
    dibujar_campo(ax)

    # Si existe columna team, pintamos por equipo
    if TEAM_COL in df.columns:
        df_local = df[df[TEAM_COL] == "local"]
        df_visit = df[df[TEAM_COL] == "visitante"]
        df_otros = df[~df[TEAM_COL].isin(["local", "visitante"])]

        ax.scatter(df_local[X_COL], df_local[Y_COL], s=6, alpha=0.35, label="local")
        ax.scatter(df_visit[X_COL], df_visit[Y_COL], s=6, alpha=0.35, label="visitante")
        if len(df_otros) > 0:
            ax.scatter(df_otros[X_COL], df_otros[Y_COL], s=6, alpha=0.25, label="otros")

        ax.legend(loc="upper right")
        ax.set_title("Sanity check: puntos reproyectados por equipo")
    else:
        ax.scatter(df[X_COL], df[Y_COL], s=6, alpha=0.35)
        ax.set_title("Sanity check: puntos reproyectados")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
