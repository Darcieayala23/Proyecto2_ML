import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import time

random_state = 42

# ============================================================
# Cargar el dataset PCA (64D)
# ============================================================
df_pca = pd.read_csv("eco_acoustic_train.csv")

pc_cols = [c for c in df_pca.columns if c.startswith("mel_")]
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

# Ajustar el modelo final de 64D: spherical, k=2
gmm_64d = GaussianMixture(n_components=2, covariance_type="spherical",
                            random_state=random_state, n_init=5)
gmm_64d.fit(X)
cluster_labels_64d = gmm_64d.predict(X)

df_pca["cluster_gmm_64d"] = cluster_labels_64d

# Crosstab
print("=== Crosstab GMM 64D vs species_id (todas las filas) ===")
print(pd.crosstab(df_pca["cluster_gmm_64d"], df_pca["species_id"]))

mask_confiable = df_pca["is_tp"] == 1

print("\n=== Crosstab GMM 64D vs species_id (is_tp=1) ===")
print(pd.crosstab(
    df_pca.loc[mask_confiable, "cluster_gmm_64d"],
    df_pca.loc[mask_confiable, "species_id"]
))

# ============================================================
# Validación externa: ARI y NMI vs species_id
# ============================================================

# --- Todas las filas ---
ari_64d_todas = adjusted_rand_score(df_pca["species_id"], df_pca["cluster_gmm_64d"])
nmi_64d_todas = normalized_mutual_info_score(df_pca["species_id"], df_pca["cluster_gmm_64d"])
print(f"\nARI (todas): {ari_64d_todas:.3f}, NMI: {nmi_64d_todas:.3f}")

# --- Solo etiquetas confiables (is_tp=1) ---
ari_64d_confiable = adjusted_rand_score(
    df_pca.loc[mask_confiable, "species_id"],
    df_pca.loc[mask_confiable, "cluster_gmm_64d"]
)
nmi_64d_confiable = normalized_mutual_info_score(
    df_pca.loc[mask_confiable, "species_id"],
    df_pca.loc[mask_confiable, "cluster_gmm_64d"]
)
print(f"ARI (is_tp=1): {ari_64d_confiable:.3f}, NMI: {nmi_64d_confiable:.3f}")