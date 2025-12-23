import pandas as pd

df = pd.read_csv("puntos_homografia.csv")

# Ajusta estos nombres si en tu CSV son distintos:
p2 = df[df["name"].str.contains("P2")].iloc[0]
p3 = df[df["name"].str.contains("P3")].iloc[0]

x_mid = int(round((p2["x_img"] + p3["x_img"]) / 2))
y_mid = int(round((p2["y_img"] + p3["y_img"]) / 2))

mask_p1 = df["name"].str.contains("P1")
df.loc[mask_p1, "x_img"] = x_mid
df.loc[mask_p1, "y_img"] = y_mid

df.to_csv("puntos_homografia.csv", index=False)
print("✅ P1 actualizado al punto medio entre P2 y P3:", x_mid, y_mid)
