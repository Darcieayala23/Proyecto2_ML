import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
import time
import matplotlib.pyplot as plt


random_state = 42

# ============================================================
# Carga de datos
# ============================================================
# IMPORTANTE: para PCA (exploración no supervisada)
# se usa el dataset COMPLETO, sin filtrar is_tp.
# is_tp solo se filtra/pondera más adelante, en la clasificación
# supervisada (MLP / XGBoost), porque ahí sí se usa species_id
# como target y su confiabilidad importa.

# Cargar datos
df = pd.read_csv("eco_acoustic_train.csv")

X_train = df[[f"mel_{i}" for i in range(64)]].values
y_train = df["species_id"].values 
# NO se usa para ajustar PCA, solo se guarda para
# colorear visualizaciones y validar clustering después

X_scaler = StandardScaler().fit(X_train)
X_train_sc = X_scaler.transform(X_train)

print(df["species_id"].value_counts())

## COMPARACION PCA

t0 = time.time()
pca_full = PCA() # crea el objeto PCA sin especificar n_components para obtener todas las componentes
pca_full.fit(X_train_sc) # ajusta PCA a los datos escalados
t_pca = time.time() - t0
print(f"Tiempo de ejecución PCA: {t_pca:.4f} segundos")

cumsum = np.cumsum(pca_full.explained_variance_ratio_) # varianza ordenado de mayor a menor (es un array acumulado de las varianzas de cada componente)
M = np.argmax(cumsum >= 0.95) + 1 # busca en ese array acumulado la primera posición donde ya superaste el 95%, y ese es tu M=11.
print(f"Componentes para 95% de varianza: {M}")

for i, v in enumerate(pca_full.explained_variance_ratio_, start=1):
    print(f"Componente {i}: varianza = {v:.4f}, acumulada = {pca_full.explained_variance_ratio_[:i].sum():.4f}")

# ============================================================
# Gráfico de varianza acumulada (Sección 3.2 del informe)
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
plt.axvline(x=M, color='g', linestyle='--', label=f'M={M} componentes')
plt.legend()
plt.title('Varianza acumulada por componentes')
plt.grid(True)
plt.show()

# ============================================================
# Dataset reducido a M componentes (95% varianza) para usar
# en clustering (GMM / DBSCAN) y como una de las combinaciones
# de features a comparar en el MLP/ensamble más adelante.
# ============================================================
pca_M = PCA(n_components=M, random_state=random_state)
X_train_pca = pca_M.fit_transform(X_train_sc) # calcula las 11 direcciones (eigenvectores) que mejor capturan la varianza de tus datos escalados
                                              # proyecta cada una las 1906 filas originales (64 columnas) sobre esas 11 nuevas direcciones.
print(f"\nShape del dataset reducido: {X_train_pca.shape}")

# Guardamos con nombres que ordenan bien numérica Y alfabéticamente
# (evita el problema de PC1, PC10, PC11, PC2... en Excel/CSV)
columnas_pca = [f"PC{str(i+1).zfill(2)}" for i in range(M)]
df_pca = pd.DataFrame(X_train_pca, columns=columnas_pca)

# Traemos las columnas de metadata desde el df original
# (mismo orden de filas, así que el alineamiento es directo)
df_pca["recording_id"] = df["recording_id"].values
df_pca["species_id"]   = df["species_id"].values # se guarda aparte, solo para validar clustering después
df_pca["songtype_id"]  = df["songtype_id"].values
df_pca["is_tp"]        = df["is_tp"].values

# Reordenamos: metadata primero, luego las PCs
orden_columnas = ["recording_id", "species_id", "songtype_id", "is_tp"] + columnas_pca
df_pca = df_pca[orden_columnas]

df_pca.to_csv("eco_acoustic_train_95.csv", index=False)

