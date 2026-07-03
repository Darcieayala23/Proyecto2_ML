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


# Cargar datos
df = pd.read_csv("eco_acoustic_train.csv")
# Filtrar solo las filas con is_tp == 1 antes de crear X_train e y_train


X_train = df[[f"mel_{i}" for i in range(64)]].values
y_train = df["species_id"].values

X_scaler = StandardScaler().fit(X_train)
X_train_sc = X_scaler.transform(X_train)


## COMPARACION PCA

t0 = time.time()
pca_full = PCA()

pca_full.fit(X_train_sc)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
M = np.argmax(cumsum >= 0.95) + 1
print(f"Componentes para 95% de varianza: {M}")

for i, v in enumerate(pca_full.explained_variance_ratio_, start=1):
    print(f"Componente {i}: varianza = {v:.4f}, acumulada = {pca_full.explained_variance_ratio_[:i].sum():.4f}")


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

