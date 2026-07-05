# Proyecto2_ML
Proyecto 2: Clasificación de Señales Eco-Acústicas

PARTE 1: Reducción de dimensionalidad

Para la primera parte se exploro los modelos de reducción de dimensionalidad utilizando una menor cantidad de datos para una mejor visualización de los datos reducidos. Los codigos utilizados para la exploración geometrica son:
1. Reduccion_dim.py
2. PCA_11D.py

PARTE 2: Clustering

Se aplicaron algoritmos de clustering no supervisado (DBSCAN y GMM) sobre datos MFCC reducidos con PCA a 11 dimensiones, comparando un enfoque basado en densidad frente a uno probabilístico. Se seleccionaron hiperparámetros mediante heurísticas y análisis de sensibilidad (k-distance, Silhouette, BIC), y se validó la correspondencia entre clústeres y especies reales usando ARI y NMI.

4. DBSCAN_clustering_11D.py
5. DBSCAN_clustering_64.py
6. GMM_clustering_11D.py
7. GMM_clustering_64D.py

PARTE 3:


6. clasificacion.py
7. GMM_clustering_64D.py


