import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import time

random_state = 42

# ============================================================
# Cargar el dataset PCA (11D)
# ============================================================
df_pca = pd.read_csv("eco_acoustic_train_95.csv")

pc_cols = [c for c in df_pca.columns if c.startswith("PC")]
X = df_pca[pc_cols].values
y_true = df_pca["species_id"].values
is_tp = df_pca["is_tp"].values

print(f"Shape de X para clustering: {X.shape}")

# ============================================================
# Paso 1: barrer distintos números de componentes (clusters)
# y evaluar con BIC, AIC y Silhouette
# ============================================================


resultados_gmm = []

for cov_type in ["full", "diag", "tied", "spherical"]:
    for k in range(2, 16):
        gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                                random_state=random_state, n_init=5)
        gmm.fit(X)
        cluster_labels = gmm.predict(X)

        bic = gmm.bic(X)
        aic = gmm.aic(X)
        sil = silhouette_score(X, cluster_labels)

        resultados_gmm.append({
            "covariance_type": cov_type, "k": k,
            "bic": bic, "aic": aic, "silhouette": sil
        })

df_gmm = pd.DataFrame(resultados_gmm)
df_gmm.to_csv("gmm_sensibilidad_completa.csv", index=False)
print(df_gmm)

# df_gmm ya debería tener las columnas: covariance_type, k, bic, aic, silhouette (11D)

colores_cov = {
    "full": "#264653",
    "diag": "#8AB0AB",
    "tied": "#E76F51",
    "spherical": "#E9C46A",
}

fig, axes = plt.subplots(2, 1, figsize=(6, 9))

for cov_type, color in colores_cov.items():
    subset = df_gmm[df_gmm["covariance_type"] == cov_type]
    axes[0].plot(subset["k"], subset["bic"], marker="o", color=color, label=cov_type)
    axes[1].plot(subset["k"], subset["silhouette"], marker="o", color=color, label=cov_type)

axes[0].set_title("BIC vs k")
axes[0].set_xlabel("k (n_components)")
axes[0].set_ylabel("BIC")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best", fontsize=9)

axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("k (n_components)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axhline(0, color="gray", linewidth=0.5, linestyle=":")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best", fontsize=9)

fig.suptitle("Sensibilidad de GMM al número de componentes (k) — Espacio PCA (11D)", fontsize=12)
fig.tight_layout()
plt.savefig("gmm_sensibilidad_11d_final.png", dpi=150)
plt.show()

# ============================================================
# Ajustar el modelo final: tied, k=2
# ============================================================
gmm_final = GaussianMixture(n_components=2, covariance_type="tied",
                              random_state=random_state, n_init=5)
gmm_final.fit(X)
cluster_labels_gmm = gmm_final.predict(X)

df_pca["cluster_gmm"] = cluster_labels_gmm

# ============================================================
# Crosstab: cluster_gmm vs species_id (todas las filas)
# ============================================================
print("=== Crosstab GMM (tied, k=2) vs species_id (todas las filas) ===")
print(pd.crosstab(df_pca["cluster_gmm"], df_pca["species_id"]))

mask_confiable = df_pca["is_tp"] == 1

print("\n=== Crosstab GMM (tied, k=2) vs species_id (is_tp=1) ===")
print(pd.crosstab(
    df_pca.loc[mask_confiable, "cluster_gmm"],
    df_pca.loc[mask_confiable, "species_id"]
))

# ============================================================
# Validación externa: ARI y NMI vs species_id
# ============================================================

# --- Todas las filas ---
ari_todas = adjusted_rand_score(df_pca["species_id"], df_pca["cluster_gmm"])
nmi_todas = normalized_mutual_info_score(df_pca["species_id"], df_pca["cluster_gmm"])
print(f"\nARI (todas): {ari_todas:.3f}, NMI: {nmi_todas:.3f}")

# --- Solo etiquetas confiables (is_tp=1) ---
ari_confiable = adjusted_rand_score(
    df_pca.loc[mask_confiable, "species_id"],
    df_pca.loc[mask_confiable, "cluster_gmm"]
)
nmi_confiable = normalized_mutual_info_score(
    df_pca.loc[mask_confiable, "species_id"],
    df_pca.loc[mask_confiable, "cluster_gmm"]
)
print(f"ARI (is_tp=1): {ari_confiable:.3f}, NMI: {nmi_confiable:.3f}")