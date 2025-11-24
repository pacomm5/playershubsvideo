import numpy as np
import cv2

# Cargamos los puntos que has clicado
src_pts = np.load("puntos_campo.npy").astype(np.float32)
print("Puntos en imagen (pixeles):")
print(src_pts)

# Definimos las coordenadas reales del campo (metros)
# Correspondencia con el orden de clic:
# 1) esquina inf. izq  -> (0, 0)
# 2) esquina inf. der  -> (105, 0)
# 3) esquina sup. der  -> (105, 68)
# 4) esquina sup. izq  -> (0, 68)
dst_pts = np.array([
    [52.5, 24.85],   # punto 1: círculo abajo
    [61.65, 34.0],   # punto 2: derecha
    [52.5, 43.15],   # punto 3: arriba
    [43.35, 34.0]    # punto 4: izquierda
], dtype=np.float32)

print("Puntos en campo (metros):")
print(dst_pts)

# Calculamos la homografía
H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
print("Matriz de homografía H:")
print(H)

# Guardamos para usarla luego
np.save("H_campo.npy", H)
print("✅ Homografía guardada en H_campo.npy")
