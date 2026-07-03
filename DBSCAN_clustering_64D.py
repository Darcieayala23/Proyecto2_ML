import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
import matplotlib.patches as mpatches
import time


random_state = 42

# ============================================================
# Cargar el dataset PCA (11D) que acabamos de guardar
# ============================================================
df_pca = pd.read_csv("eco_acoustic_train.csv")

pc_cols = [c for c in df_pca.columns if c.startswith("mel_")]
X = df_pca[pc_cols].values          # solo las 11 dimensiones, sin metadata
y_true = df_pca["species_id"].values
is_tp = df_pca["is_tp"].values


print(f"Shape de X para clustering: {X.shape}")

# ============================================================
# Paso 1: estimar 'eps' con el k-distance plot
# ============================================================
# Regla práctica: min_samples ~ 2 * n_dimensiones (Sander et al.)
min_samples = 2 * X.shape[1]  # 2 * 64 = 128, para minPts en DBSCAN

neighbors = NearestNeighbors(n_neighbors=min_samples) # NearestNeighbors incluye al propio
                                                        # Por lo tanto, de las 128 columnas devueltas:
                                                        #   columna 0        -> el punto mismo (dist = 0)
                                                        #   columnas 1 a 127   -> los 127 vecinos reales más cercanos
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
k_distances = np.sort(distances[:, -1]) # nos quedamos con la distancia al vecino más lejano (columna -1), y la ordenamos de menor a mayor

# Usamos Kneedle para encontrar el "codo" automáticamente
kl = KneeLocator(
    range(len(k_distances)), 
    k_distances, 
    curve="convex", 
    direction="increasing",
    #S=5  # prueba 3, 5, 10... más alto = menos sensible a ruido
)
eps_kneedle = k_distances[kl.knee]
print(f"Eps sugerido por Kneedle: {eps_kneedle:.3f} (en el punto ordenado #{kl.knee})")

# ============================================================
# Gráfico del k-distance plot
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(k_distances, label="k-distance")

k_vecino_real = min_samples - 1  # 21

# Marcar el punto que identificó Kneedle
plt.scatter([kl.knee], [eps_kneedle], color="red", zorder=5, s=60,
            label=f"eps sugerido = {eps_kneedle:.3f} ({k_vecino_real}-ésimo vecino)")

plt.xlabel("Puntos ordenados")
plt.ylabel(f"Distancia al {min_samples}-ésimo vecino")
plt.title("K-distance plot para estimar eps (DBSCAN)")
plt.legend(loc="best")
plt.grid(True)
plt.savefig("K-distance_eps_estimado.png", dpi=150)
plt.show()

# ============================================================
# Analisis de sensibilidad: probar eps en un rango amplio y ver cómo cambian n_clusters, %ruido y silhouette
# ============================================================

resultados = []
for eps_test in np.arange(0.50, 7.5, 0.25):
    db = DBSCAN(eps=eps_test, min_samples=128).fit(X)
    labels = db.labels_
    n_c = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    pct_noise = 100 * n_noise / len(labels)

    if n_c >= 2:
        mask = labels != -1
        sil = silhouette_score(X[mask], labels[mask])
    else:
        sil = np.nan
        
    resultados.append({
        "eps": eps_test, "n_clusters": n_c,
        "pct_noise": pct_noise, "silhouette": sil
    })

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("dbscan_sensibilidad_eps.csv", index=False)
print(df_resultados)


# ============================================================
# Graficar el analisis de sensibilidad: silhouette vs eps, coloreado por n_clusters, y %ruido en panel inferior
# ============================================================
color_map = {
    0: "#D9D9D9",   # todo ruido - gris muy claro, casi invisible a propósito
    1: "#E8998D",   # 1 cluster (colapso) - salmón apagado
    2: "#264653",   # 2 clusters (zona de interés) - azul petróleo oscuro
    3: "#8AB0AB",   # verde-grisáceo suave
    4: "#E9C46A",   # mostaza suave (como acento, no compite)
}
colores_puntos = [color_map.get(k, "black") for k in df_resultados["n_clusters"]]

# Para graficar: donde silhouette es NaN, usamos y=0 como posición visual
# (pero NO lo tratamos como un score real, solo como marcador de "no aplica")
y_para_graficar = df_resultados["silhouette"].fillna(0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(df_resultados["eps"], y_para_graficar,
          color="gray", linewidth=1, alpha=0.4, zorder=1)  # línea guía tenue

# Puntos válidos (k>=2, silhouette real) -> círculo normal
mask_valido = df_resultados["silhouette"].notna()
ax1.scatter(df_resultados.loc[mask_valido, "eps"], y_para_graficar[mask_valido],
             c=[color_map.get(k, "black") for k in df_resultados.loc[mask_valido, "n_clusters"]],
             s=70, zorder=2, edgecolor="black", linewidth=0.5, marker="o")

# Puntos sin silhouette (k=0 o k=1) -> marcador o en y=0, para no confundir con score real
mask_nan = ~mask_valido
ax1.scatter(df_resultados.loc[mask_nan, "eps"], y_para_graficar[mask_nan],
             c=[color_map.get(k, "black") for k in df_resultados.loc[mask_nan, "n_clusters"]],
             s=90, zorder=3, edgecolor="black", linewidth=0.8, marker="o")

eps_elegido_final = 2.75
ax1.axvline(eps_elegido_final, color="black", linestyle="--", linewidth=1.2,
            label=f"eps elegido = {eps_elegido_final}")

ax1.set_ylabel("Silhouette Score")
ax1.set_title("Silhouette Score vs eps (color = número de clusters, X = sin score real)")
ax1.grid(True, alpha=0.3)

# Leyenda manual de colores (categórica)
handles = [mpatches.Patch(color=c, label=f"{k} cluster(s)" if k != 0 else "0 (todo ruido)")
           for k, c in color_map.items() if k in df_resultados["n_clusters"].values]
handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label=f"eps elegido={eps_elegido_final}"))
ax1.legend(handles=handles, loc="upper right", fontsize=9)

ax2.bar(df_resultados["eps"], df_resultados["pct_noise"],
         width=0.2, color="tab:red", alpha=0.6)
ax2.axvline(eps_elegido_final, color="black", linestyle="--", linewidth=1.2)
ax2.set_xlabel("eps")
ax2.set_ylabel("% Ruido")
ax2.grid(True, alpha=0.3)

fig.suptitle("Sensibilidad de DBSCAN al parámetro eps", fontsize=13)
fig.tight_layout()
plt.savefig("silhouette_vs_eps_v3.png", dpi=150)
plt.show()


# =====================================================
# Resultados de: Eps elegido: 2.75, min_samples = 128 (2 * n_dimensiones)
# =====================================================

tiempos = []
for _ in range(10):
    t0 = time.time()
    DBSCAN(eps=2.75, min_samples=128).fit(X)
    tiempos.append(time.time() - t0)

print(tiempos)  # <- agrega esto, imprime la lista completa


t_dbscan_64d = np.mean(tiempos)
print(f"Tiempo promedio DBSCAN (64D, 10 corridas): {t_dbscan_64d:.4f} ± {np.std(tiempos):.4f} segundos")

# Ahora sí, la corrida "real" que usamos para el análisis (labels, crosstabs, etc.)
db_final = DBSCAN(eps=2.75, min_samples=128).fit(X)
labels = db_final.labels_
print(pd.Series(labels).value_counts())  # incluye -1 (ruido)

df_pca["cluster_dbscan"] = labels

# Versión 1: dataset completo (todas las etiquetas, incluyendo ruidosas)
# ============================================================
print("=== Crosstab completo (todas las filas) ===")
ct_completo = pd.crosstab(df_pca["cluster_dbscan"], df_pca["species_id"])
print(ct_completo)

# Versión 2: solo filas con etiqueta confiable (is_tp == 1)
# ============================================================
print("\n=== Crosstab solo con etiquetas confiables (is_tp=1) ===")
mask_confiable = df_pca["is_tp"] == 1
ct_confiable = pd.crosstab(
    df_pca.loc[mask_confiable, "cluster_dbscan"], 
    df_pca.loc[mask_confiable, "species_id"]
)
print(ct_confiable)

print("\n=== Crosstab confiable, normalizado por cluster (%) ===")
print(pd.crosstab(
    df_pca.loc[mask_confiable, "cluster_dbscan"], 
    df_pca.loc[mask_confiable, "species_id"],
    normalize='index'
).round(3))


# ============================================================
# Validación externa: ARI y NMI entre clusters de DBSCAN y especies reales
# ============================================================

# --- Con el dataset completo (todas las filas, incluyendo is_tp=0) ---
ari_completo = adjusted_rand_score(df_pca["species_id"], df_pca["cluster_dbscan"])
nmi_completo = normalized_mutual_info_score(df_pca["species_id"], df_pca["cluster_dbscan"])

print(f"\n=== Validación externa (todas las filas) ===")
print(f"ARI: {ari_completo:.3f}")
print(f"NMI: {nmi_completo:.3f}")

# --- Solo con etiquetas confiables (is_tp == 1) ---
ari_confiable = adjusted_rand_score(
    df_pca.loc[mask_confiable, "species_id"],
    df_pca.loc[mask_confiable, "cluster_dbscan"]
)
nmi_confiable = normalized_mutual_info_score(
    df_pca.loc[mask_confiable, "species_id"],
    df_pca.loc[mask_confiable, "cluster_dbscan"]
)

print(f"\n=== Validación externa (solo is_tp=1) ===")
print(f"ARI: {ari_confiable:.3f}")
print(f"NMI: {nmi_confiable:.3f}")




