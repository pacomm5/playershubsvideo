import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

# ---------- CONFIGURACIÓN ----------
RUTA_CSV = "detecciones_match2_equipos.csv"
FRAME_COL = "frame"
X_COL = "X_m"
Y_COL = "Y_m"
TEAM_COL = "team"

LARGO = 105
ANCHO = 68

FPS_ORIG_VIDEO = 25
FRAME_STEP = 3  # usar 1 de cada 3 frames para acortar
SALIDA_MP4 = "animacion_partido_trail.mp4"

N_FRAMES_TRAIL = 20  # estela de los últimos N frames (~N/FPS seg)


def dibujar_campo(ax):
    ax.set_facecolor("#2e7d32")
    ax.add_patch(patches.Rectangle((0, 0), LARGO, ANCHO, linewidth=2, edgecolor="white", facecolor="none"))
    ax.plot([0, LARGO, LARGO, 0, 0], [0, 0, ANCHO, ANCHO, 0], "w", linewidth=2)
    ax.plot([LARGO / 2, LARGO / 2], [0, ANCHO], "w", linewidth=2)
    circulo = plt.Circle((LARGO / 2, ANCHO / 2), 9.15, color="white", fill=False, linewidth=2)
    ax.add_artist(circulo)
    ax.scatter([LARGO / 2], [ANCHO / 2], s=15, color="white")
    area_w, area_l = 40.3, 16.5
    ax.add_patch(patches.Rectangle((0, (ANCHO - area_w) / 2), area_l, area_w, linewidth=2, edgecolor="white", facecolor="none"))
    ax.add_patch(patches.Rectangle((LARGO - area_l, (ANCHO - area_w) / 2), area_l, area_w, linewidth=2, edgecolor="white", facecolor="none"))
    ax.set_xlim(0, LARGO)
    ax.set_ylim(ANCHO, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud (m)")
    ax.set_ylabel("Anchura (m)")


def main():
    df = pd.read_csv(RUTA_CSV)
    df = df[df[TEAM_COL].isin(["local", "visitante"])].copy()
    df = df.sort_values(FRAME_COL)
    df[X_COL] = df[X_COL].clip(0, LARGO)
    df[Y_COL] = df[Y_COL].clip(0, ANCHO)

    frames_unicos = sorted(df[FRAME_COL].unique())
    frames_sampleados = frames_unicos[::FRAME_STEP]

    print(f"Frames totales: {len(frames_unicos)}")
    print(f"Frames usados en animación: {len(frames_sampleados)}")
    duracion_seg = len(frames_sampleados) / (FPS_ORIG_VIDEO / FRAME_STEP)
    print(f"Duración aproximada de la animación: {duracion_seg:.1f} s")

    if not frames_sampleados:
        print("No hay frames después del filtrado.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    dibujar_campo(ax)

    scatter_local = ax.scatter([], [], s=60, color="blue", label="local", zorder=3)
    scatter_visit = ax.scatter([], [], s=60, color="red", label="visitante", zorder=3)

    trail_local, = ax.plot([], [], color="blue", linewidth=2, alpha=0.6, zorder=2)
    trail_visit, = ax.plot([], [], color="red", linewidth=2, alpha=0.6, zorder=2)

    texto_tiempo = ax.text(
        5, 5, "", color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.4, boxstyle="round"), zorder=4
    )
    ax.legend(loc="upper right")

    hist_local = []
    hist_visit = []

    def actualizar(frame_idx):
        frame_actual = frames_sampleados[frame_idx]
        df_frame = df[df[FRAME_COL] == frame_actual]

        df_local = df_frame[df_frame[TEAM_COL] == "local"]
        df_visit = df_frame[df_frame[TEAM_COL] == "visitante"]

        xs_local = df_local[X_COL].values
        ys_local = df_local[Y_COL].values
        xs_visit = df_visit[X_COL].values
        ys_visit = df_visit[Y_COL].values

        scatter_local.set_offsets(np.column_stack([xs_local, ys_local]) if len(xs_local) else np.empty((0, 2)))
        scatter_visit.set_offsets(np.column_stack([xs_visit, ys_visit]) if len(xs_visit) else np.empty((0, 2)))

        if len(xs_local) > 0:
            hist_local.append((np.mean(xs_local), np.mean(ys_local)))
        elif hist_local:
            hist_local.append(hist_local[-1])

        if len(xs_visit) > 0:
            hist_visit.append((np.mean(xs_visit), np.mean(ys_visit)))
        elif hist_visit:
            hist_visit.append(hist_visit[-1])

        hist_local_rec = hist_local[-N_FRAMES_TRAIL:]
        hist_visit_rec = hist_visit[-N_FRAMES_TRAIL:]

        if hist_local_rec:
            xs_tl, ys_tl = zip(*hist_local_rec)
            trail_local.set_data(xs_tl, ys_tl)
        if hist_visit_rec:
            xs_tv, ys_tv = zip(*hist_visit_rec)
            trail_visit.set_data(xs_tv, ys_tv)

        segundo = frame_actual / FPS_ORIG_VIDEO
        minuto = int(segundo // 60)
        seg_rest = int(segundo % 60)
        texto_tiempo.set_text(f"{minuto:02d}:{seg_rest:02d}")

        return scatter_local, scatter_visit, trail_local, trail_visit, texto_tiempo

    fps_anim = FPS_ORIG_VIDEO / FRAME_STEP
    anim = FuncAnimation(
        fig,
        actualizar,
        frames=len(frames_sampleados),
        interval=1000 / fps_anim,
        blit=True
    )

    writer = FFMpegWriter(fps=fps_anim, bitrate=2500)
    print(f"Guardando animación en {SALIDA_MP4} ...")
    anim.save(SALIDA_MP4, writer=writer)
    print("Listo.")


if __name__ == "__main__":
    main()
