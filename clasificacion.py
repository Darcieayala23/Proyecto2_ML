import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

SEED = 42
# --- Escenario A: 64 variables originales ---
df_original = pd.read_csv("eco_acoustic_train.csv")
df_orig_filtered = df_original[df_original["is_tp"] == 1].copy()

X_orig = df_orig_filtered[[f"mel_{i}" for i in range(64)]].values
y_orig = df_orig_filtered["species_id"].values

# --- Escenario B: 11 variables de PCA ---
df_pca = pd.read_csv("eco_acoustic_train_95.csv")
df_pca_filtered = df_pca[df_pca["is_tp"] == 1].copy()

X_pca = df_pca_filtered[[f"PC{str(i+1).zfill(2)}" for i in range(11)]].values
y_pca = df_pca_filtered["species_id"].values 

# Mapeo de las etiquetas:
mapeo_clases = {10: 0, 12: 1, 17: 2, 18: 3, 23: 4}
y_orig_mapped = np.vectorize(mapeo_clases.get)(y_orig)
y_pca_mapped = np.vectorize(mapeo_clases.get)(y_pca)

# Particion de datos en Entrenamiento (80%) y Validacion/Test (20%)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_orig, y_orig_mapped, test_size=0.2, random_state=SEED, stratify=y_orig_mapped
)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_pca_mapped, test_size=0.2, random_state=SEED, stratify=y_pca_mapped
)

print("\n=== CONFIGURACION DE LOS MODELOS (SECCION 3.4) ===")

# Definicion de la Red Neuronal (MLP)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    random_state=SEED
)

# Definicion del Modelo de Ensamble (XGBoost)
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=SEED,
    eval_metric='mlogloss'
)

resultados = []
def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, nombre_modelo, escenario):
    """Funcion auxiliar para entrenar, medir tiempo y calcular métricas."""
    # Medir tiempo de entrenamiento
    t0 = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - t0
    
    # Prediccion
    y_pred = modelo.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    resultados.append({
        "Escenario": escenario,
        "Modelo": nombre_modelo,
        "Variables": X_train.shape[1],
        "Accuracy": f"{acc*100:.2f}%",
        "F1-Score": f"{f1*100:.2f}%",
        "Tiempo (s)": f"{tiempo_entrenamiento:.4f}s"
    })
    
    print(f"\nReporte de Clasificacion para {nombre_modelo} ({escenario}):")
    print(classification_report(y_test, y_pred, target_names=["Especie 10", "Especie 12", "Especie 17", "Especie 18", "Especie 23"]))

# --- EXPERIMENTO ---
# Seccion 3.4: Entrenamiento con datos originales (64 features)
evaluar_modelo(mlp_model, X_train_orig, X_test_orig, y_train_orig, y_test_orig, "Red Neuronal (MLP)", "3.4 Original")
evaluar_modelo(xgb_model, X_train_orig, X_test_orig, y_train_orig, y_test_orig, "XGBoost Ensamble", "3.4 Original")

# Seccion 3.5: Entrenamiento con datos reducidos por PCA (11 features)
evaluar_modelo(mlp_model, X_train_pca, X_test_pca, y_train_pca, y_test_pca, "Red Neuronal (MLP)", "3.5 Reducido (PCA)")
evaluar_modelo(xgb_model, X_train_pca, X_test_pca, y_train_pca, y_test_pca, "XGBoost Ensamble", "3.5 Reducido (PCA)")

print("\n=== CUADRO COMPARATIVO FINAL (SECCION 3.5) ===")
df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16})
especies = ["Especie 10", "Especie 12", "Especie 17", "Especie 18", "Especie 23"]

# 1. Matriz de Confusión para XGBoost - Espacio Original (64D)
xgb_model.fit(X_train_orig, y_train_orig)
y_pred_o = xgb_model.predict(X_test_orig)
cm_o = confusion_matrix(y_test_orig, y_pred_o)

plt.figure(figsize=(7, 6))
sns.heatmap(cm_o, annot=True, fmt='d', cmap='Blues', xticklabels=especies, yticklabels=especies, cbar=False)
plt.title("XGBoost - Espacio Original (64D)")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")
plt.tight_layout()
plt.savefig("matriz_original.png", dpi=300)
plt.close()

# 2. Matriz de Confusión para XGBoost - Espacio Reducido (11D PCA)
xgb_model.fit(X_train_pca, y_train_pca)
y_pred_p = xgb_model.predict(X_test_pca)
cm_p = confusion_matrix(y_test_pca, y_pred_p)

plt.figure(figsize=(7, 6))
sns.heatmap(cm_p, annot=True, fmt='d', cmap='Greens', xticklabels=especies, yticklabels=especies, cbar=False)
plt.title("XGBoost - Espacio Reducido (11D PCA)")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")
plt.tight_layout()
plt.savefig("matriz_reducida.png", dpi=300)
plt.close()