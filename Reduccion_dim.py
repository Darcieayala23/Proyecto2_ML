import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
import umap
import warnings
import time
import matplotlib.pyplot as plt

plt.ion()
random_state = 42


# Cargar datos
df = pd.read_csv("eco_acoustic_train.csv")
# Filtrar solo las filas con is_tp == 1 antes de crear X_train e y_train
if "is_tp" in df.columns:
    df = df[df["is_tp"] == 1].copy()
else:
    raise KeyError("La columna 'is_tp' no existe en eco_acoustic_train.csv")

X_train = df[[f"mel_{i}" for i in range(64)]].values
y_train = df["species_id"].values

X_scaler = StandardScaler().fit(X_train)
X_train_sc = X_scaler.transform(X_train)

# ── PCA ─────────────────────────────────────────────────────────────

t0 = time.time()
pca_2d = PCA(n_components=2)
Z_pca2d = pca_2d.fit_transform(X_train_sc)
tiempo_pca_2d = time.time() - t0
var_ret_2d = sum(pca_2d.explained_variance_ratio_) * 100

print("\n--- Resultados PCA 2D ---")
print(f"Eigenvectores (U) shape: {pca_2d.components_.shape}")
print(f"Eigenvectores (U):\n{pca_2d.components_}")
print(f"\nEigenvalores (diagonal Σ):\n{pca_2d.explained_variance_}")
print(f"\n% Varianza explicada por componente:\n{pca_2d.explained_variance_ratio_}")
print(f"\nVarianza total explicada: {var_ret_2d:.2f}%")
print(f"Tiempo PCA 2D: {tiempo_pca_2d:.4f} s")

Z_check2d = pca_2d.transform(X_train_sc)
print(f"\nShape original:    {X_train_sc.shape}")
print(f"Shape proyectado:  {Z_check2d.shape}")

X_reconstructed_2d = pca_2d.inverse_transform(Z_check2d)
print(f"Shape reconstruido: {X_reconstructed_2d.shape}")

error_2d = np.mean((X_train_sc - X_reconstructed_2d) ** 2)
print(f"\nError de reconstrucción (MSE) 2D: {error_2d:.6f}")

plt.figure(figsize=(8, 5))
scatter = plt.scatter(Z_pca2d[:, 0], Z_pca2d[:, 1], c=y_train, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Proyección PCA — {var_ret_2d:.2f}% varianza retenida')
plt.colorbar(scatter, label='Clase')
plt.grid(True)
plt.draw()
plt.pause(0.1)

t0 = time.time()
pca_3d = PCA(n_components=3)
Z_pca3d = pca_3d.fit_transform(X_train_sc)
tiempo_pca_3d = time.time() - t0
var_ret_3d = sum(pca_3d.explained_variance_ratio_) * 100

print("\n--- Resultados PCA 3D ---")
print(f"Eigenvectores (U) shape: {pca_3d.components_.shape}")
print(f"Eigenvectores (U):\n{pca_3d.components_}")
print(f"\nEigenvalores (diagonal Σ):\n{pca_3d.explained_variance_}")
print(f"\n% Varianza explicada por componente:\n{pca_3d.explained_variance_ratio_}")
print(f"\nVarianza total explicada: {var_ret_3d:.2f}%")
print(f"Tiempo PCA 3D: {tiempo_pca_3d:.4f} s")

Z_check3d = pca_3d.transform(X_train_sc)
print(f"\nShape original:    {X_train_sc.shape}")
print(f"Shape proyectado:  {Z_check3d.shape}")

X_reconstructed_3d = pca_3d.inverse_transform(Z_check3d)
print(f"Shape reconstruido: {X_reconstructed_3d.shape}")

error_3d = np.mean((X_train_sc - X_reconstructed_3d) ** 2)
print(f"\nError de reconstrucción (MSE) 3D: {error_3d:.6f}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(Z_pca3d[:, 0], Z_pca3d[:, 1], Z_pca3d[:, 2], c=y_train, cmap='tab10', s=50)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_zlabel('PC3', fontsize=12)
ax.set_title(f'PCA 3D — Varianza: {var_ret_3d:.1f}%', fontsize=14)
plt.colorbar(scatter, label='Clase', shrink=0.8)
plt.draw()
plt.pause(0.1)


# ── t-SNE ───────────────────────────────────────────────────────────

print("\n--- t-SNE 2D (perplexity=30) ---")
t0 = time.time()
tsne_2d = TSNE(n_components=2, perplexity=45, max_iter=1000, random_state=42, init='pca')
Z_tsne2 = tsne_2d.fit_transform(X_train_sc)
t_tsne2 = time.time() - t0
print(f"Tiempo t-SNE 2D: {t_tsne2:.2f} s")

plt.figure(figsize=(8, 5))
scatter = plt.scatter(Z_tsne2[:, 0], Z_tsne2[:, 1], c=y_train, cmap='tab10', s=15, alpha=0.8)
plt.colorbar(scatter, label='Clase')
plt.title(f't-SNE 2D — perplexity=30 (t={t_tsne2:.1f}s)', fontsize=14)
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)
plt.grid(True)
plt.draw()
plt.pause(0.1)

print("\n--- t-SNE 3D (perplexity=30) ---")
t0 = time.time()
tsne_3d = TSNE(n_components=3, perplexity=45, max_iter=1000, random_state=42, init='pca')
Z_tsne3 = tsne_3d.fit_transform(X_train_sc)     
t_tsne3 = time.time() - t0
print(f"Tiempo t-SNE 3D: {t_tsne3:.2f} s")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(Z_tsne3[:, 0], Z_tsne3[:, 1], Z_tsne3[:, 2], c=y_train, cmap='tab10', s=50)
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_zlabel('t-SNE 3', fontsize=12)
ax.set_title(f't-SNE 3D — perplexity=30 (t={t_tsne3:.1f}s)', fontsize=14)
plt.colorbar(scatter, label='Clase', shrink=0.8)
plt.draw()
plt.pause(0.1)

print("\n--- t-SNE 2D (variación de perplexity) ---")
perplexities = [5, 15, 30, 45, 50, 80]
fig, axes = plt.subplots(1, len(perplexities), figsize=(6 * len(perplexities), 5))
fig.suptitle('t-SNE 2D — Variación de Perplexity', fontsize=15)

for i, perp in enumerate(perplexities):
    t0 = time.time()
    tsne_p = TSNE(n_components=2, perplexity=perp, max_iter=1000, random_state=42, init='pca')
    Z_tsne_p = tsne_p.fit_transform(X_train_sc)
    t_tsne_p = time.time() - t0
    print(f"  perplexity={perp:>2d} | Tiempo: {t_tsne_p:.2f} s")

    sc = axes[i].scatter(Z_tsne_p[:, 0], Z_tsne_p[:, 1], c=y_train, cmap='tab10', s=15, alpha=0.8)
    plt.colorbar(sc, ax=axes[i], label='Clase')
    axes[i].set_title(f't-SNE 2D — perplexity={perp}\n(t={t_tsne_p:.1f}s)', fontsize=13)
    axes[i].set_xlabel('t-SNE 1', fontsize=12)
    axes[i].set_ylabel('t-SNE 2', fontsize=12)

plt.tight_layout()
plt.draw()
plt.pause(0.1)


# ── UMAP ────────────────────────────────────────────────────────────

print("\nUMAP 2D")
inicio = time.time()
umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
Z_umap_2d = umap_2d.fit_transform(X_train_sc)
tiempo_umap_2d = time.time() - inicio
sil_2d = silhouette_score(Z_umap_2d, y_train)
print(f"Tiempo: {tiempo_umap_2d:.4f} s")
print(f"Silhouette score: {sil_2d:.4f}")

plt.figure(figsize=(8, 6))
sc = plt.scatter(Z_umap_2d[:, 0], Z_umap_2d[:, 1], c=y_train, cmap='tab10', s=20)
plt.colorbar(sc, label='Clase')
plt.title(f"UMAP 2D (n_neighbors=15, min_dist=0.1)\nsilhouette={sil_2d:.3f}", fontsize=14)
plt.xlabel("Dim 1", fontsize=14)
plt.ylabel("Dim 2", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.draw()
plt.pause(0.1)

print("\nUMAP 3D")
inicio = time.time()
umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
Z_umap_3d = umap_3d.fit_transform(X_train_sc)
tiempo_umap_3d = time.time() - inicio
sil_3d = silhouette_score(Z_umap_3d, y_train)
print(f"Tiempo: {tiempo_umap_3d:.4f} s")
print(f"Silhouette score: {sil_3d:.4f}")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(Z_umap_3d[:, 0], Z_umap_3d[:, 1], Z_umap_3d[:, 2], c=y_train, cmap='tab10', s=20)
plt.colorbar(sc, label='Clase')
ax.set_title(f"UMAP 3D (n_neighbors=15, min_dist=0.1)\nsilhouette={sil_3d:.3f}", fontsize=14)
ax.set_xlabel("Dim 1", fontsize=14)
ax.set_ylabel("Dim 2", fontsize=14)
ax.set_zlabel("Dim 3", fontsize=14)
ax.tick_params(labelsize=12)
plt.draw()
plt.pause(0.1)

print("\nPruebas de hiperparámetros")
experimentos = [
    (5, 0.1),
    (15, 0.1),
    (30, 0.1),
    (15, 0.5),
    (15, 0.8)
]
resultados = []

fig, axes = plt.subplots(1, len(experimentos), figsize=(24, 4.5),
                          num="UMAP — Comparación de hiperparámetros")

for ax_i, (vecinos, dist) in zip(axes, experimentos):
    inicio_var = time.time()
    modelo = umap.UMAP(n_components=2, n_neighbors=vecinos, min_dist=dist, random_state=42)
    Z_var = modelo.fit_transform(X_train_sc)
    tiempo_var = time.time() - inicio_var
    sil_var = silhouette_score(Z_var, y_train)
    resultados.append((vecinos, dist, tiempo_var, sil_var))

    ax_i.scatter(Z_var[:, 0], Z_var[:, 1], c=y_train, cmap='tab10', s=15)
    ax_i.set_title(f'n_neighbors={vecinos}, min_dist={dist}\nsil={sil_var:.3f}, t={tiempo_var:.2f}s', fontsize=13)
    ax_i.set_xticks([])
    ax_i.set_yticks([])

    print(
        f"n_neighbors={vecinos:>3}, "
        f"min_dist={dist:.1f} | "
        f"tiempo={tiempo_var:6.3f}s | "
        f"silhouette={sil_var:.4f}"
    )

fig.suptitle('UMAP 2D — Comparación de hiperparámetros', fontsize=16)
plt.tight_layout()
plt.draw()
plt.pause(0.1)

mejor = max(resultados, key=lambda r: r[3])
print(f"\n>> Mejor combinación según silhouette: "
      f"n_neighbors={mejor[0]}, min_dist={mejor[1]} "
      f"(silhouette={mejor[3]:.4f})")

tabla = pd.DataFrame(resultados, columns=['n_neighbors', 'min_dist', 'Tiempo (s)', 'Silhouette'])
tabla['n_neighbors'] = tabla['n_neighbors'].astype(int)
tabla['Tiempo (s)'] = tabla['Tiempo (s)'].round(3)
tabla['Silhouette'] = tabla['Silhouette'].round(4)

print("\n" + "=" * 50)
print("TABLA COMPARATIVA — UMAP hiperparámetros")
print("=" * 50)
print(tabla.to_string(index=False))

tabla.to_csv('tabla_hiperparametros_umap.csv', index=False)

celdas_texto = [
    [str(int(row['n_neighbors'])), f"{row['min_dist']:.1f}",
     f"{row['Tiempo (s)']:.3f}", f"{row['Silhouette']:.4f}"]
    for _, row in tabla.iterrows()
]

fig_tabla, ax_tabla = plt.subplots(figsize=(7, 1.2 + 0.4 * len(tabla)))
ax_tabla.axis('off')
celdas = ax_tabla.table(cellText=celdas_texto, colLabels=tabla.columns, cellLoc='center', loc='center')
celdas.auto_set_font_size(False)
celdas.set_fontsize(12)
celdas.scale(1, 1.8)

idx_mejor = tabla['Silhouette'].idxmax()
for col in range(len(tabla.columns)):
    celdas[(idx_mejor + 1, col)].set_facecolor('#d4edda')

for col in range(len(tabla.columns)):
    celdas[(0, col)].set_facecolor('#003087')
    celdas[(0, col)].set_text_props(color='white', fontweight='bold')

ax_tabla.set_title('UMAP — Comparación de hiperparámetros', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('tabla_hiperparametros_umap.png', dpi=200, bbox_inches='tight')
plt.draw()
plt.pause(0.1)


# ── Comparación final ──────────────────────────────────────────────
def safe_trustworthiness(X_original: np.ndarray, Z_reduced: np.ndarray, k: int = 5) -> float:
    """Calcula trustworthiness de forma segura para la comparación final."""
    if X_original.shape[0] <= k:
        return float('nan')
    try:
        return float(trustworthiness(X_original, Z_reduced, n_neighbors=k))
    except Exception:
        return float('nan')


# calcular métricas de confianza y preparar títulos con tiempos
trust_pca2 = safe_trustworthiness(X_train_sc, Z_pca2d, k=5)
trust_tsne2 = safe_trustworthiness(X_train_sc, Z_tsne2, k=5)
trust_umap2 = safe_trustworthiness(X_train_sc, Z_umap_2d, k=5)

trust_pca3 = safe_trustworthiness(X_train_sc, Z_pca3d, k=5)
trust_tsne3 = safe_trustworthiness(X_train_sc, Z_tsne3, k=5)
trust_umap3 = safe_trustworthiness(X_train_sc, Z_umap_3d, k=5)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].scatter(Z_pca2d[:, 0], Z_pca2d[:, 1], c=y_train, cmap='tab10', s=20)
ax[0].set_title(f'PCA 2D — Varianza: {var_ret_2d:.1f}%\n(t={tiempo_pca_2d:.2f}s)')

ax[1].scatter(Z_tsne2[:, 0], Z_tsne2[:, 1], c=y_train, cmap='tab10', s=20)
ax[1].set_title(f't-SNE 2D — trustworthiness: {trust_tsne2:.3f}\n(t={t_tsne2:.2f}s)')

ax[2].scatter(Z_umap_2d[:, 0], Z_umap_2d[:, 1], c=y_train, cmap='tab10', s=20)
ax[2].set_title(f'UMAP 2D — trustworthiness: {trust_umap2:.3f}\n(t={tiempo_umap_2d:.2f}s)')

plt.suptitle('Comparación PCA vs t-SNE vs UMAP (2D)')
plt.tight_layout()
plt.draw()
plt.pause(0.1)

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(Z_pca3d[:, 0], Z_pca3d[:, 1], Z_pca3d[:, 2], c=y_train, cmap='tab10', s=20)
ax1.set_title(f'PCA 3D — Varianza: {var_ret_3d:.1f}%\n(t={tiempo_pca_3d:.2f}s)')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(Z_tsne3[:, 0], Z_tsne3[:, 1], Z_tsne3[:, 2], c=y_train, cmap='tab10', s=20)
ax2.set_title(f't-SNE 3D — trustworthiness: {trust_tsne3:.3f}\n(t={t_tsne3:.2f}s)')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(Z_umap_3d[:, 0], Z_umap_3d[:, 1], Z_umap_3d[:, 2], c=y_train, cmap='tab10', s=20)
ax3.set_title(f'UMAP 3D — trustworthiness: {trust_umap3:.3f}\n(t={tiempo_umap_3d:.2f}s)')

plt.suptitle('Comparación PCA vs t-SNE vs UMAP (3D)')
plt.tight_layout()
plt.draw()
plt.pause(0.1)

input('\nPresiona Enter para cerrar todos los plots...')