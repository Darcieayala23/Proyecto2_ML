"""
Políticas de Moderación (Umbrales Operativos)

Entrena el modelo XGBoost sobre las características reducidas mediante PCA,
evalúa el conjunto de prueba y aplica una política de moderación basada en
umbrales probabilísticos. Además, genera las figuras y el artefacto utilizado
por la aplicación Streamlit.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

# CONFIG

SEED = 42
P_HIGH = 0.85          # umbral zona de Confianza
P_LOW = 0.40           # umbral zona de Rechazo
N_COMPONENTS = 11      # componentes PCA (95% varianza)
TRAIN_ON_TP_ONLY = True
EVAL_ON_TP_ONLY = True

BASE_DIR = os.getcwd()
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
})

MAPEO_CLASES = {10: 0, 12: 1, 17: 2, 18: 3, 23: 4}
INV_MAPEO = {v: k for k, v in MAPEO_CLASES.items()}
ESPECIES = ["Esp. 10", "Esp. 12", "Esp. 17", "Esp. 18", "Esp. 23"]

#Ingesta de datos

df_train = pd.read_csv("eco_acoustic_train.csv")
df_test = pd.read_csv("eco_acoustic_test.csv")

mel_cols = [f"mel_{i}" for i in range(64)]

X_train_raw = df_train[mel_cols].values
X_test_raw = df_test[mel_cols].values

# Preprocesamiento

scaler = StandardScaler().fit(X_train_raw)
X_train_sc = scaler.transform(X_train_raw)
X_test_sc = scaler.transform(X_test_raw)

pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(X_train_sc)
X_train_pca = pca.transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

var_ret = pca.explained_variance_ratio_.sum()
print(f"Varianza retenida por PCA({N_COMPONENTS}): {var_ret*100:.2f}%")

# Entrenamiento

if TRAIN_ON_TP_ONLY:
    mask_tr = df_train["is_tp"].values == 1
else:
    mask_tr = np.ones(len(df_train), dtype=bool)

X_fit = X_train_pca[mask_tr]
y_fit = np.vectorize(MAPEO_CLASES.get)(df_train.loc[mask_tr, "species_id"].values)

print(f"Muestras de entrenamiento (is_tp only={TRAIN_ON_TP_ONLY}): {X_fit.shape[0]}")
print("Clases presentes en train:", sorted(set(y_fit)))
assert set(y_fit) == set(MAPEO_CLASES.values()), "Faltan clases en el train filtrado"

# Entrenamiento del modelo

xgb = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    random_state=SEED, eval_metric="mlogloss",
)
t0 = time.time()
xgb.fit(X_fit, y_fit)
t_fit = time.time() - t0
print(f"Tiempo de entrenamiento XGBoost: {t_fit:.4f} s")

# Evaluación

if EVAL_ON_TP_ONLY:
    mask_te = df_test["is_tp"].values == 1
else:
    mask_te = np.ones(len(df_test), dtype=bool)

df_eval = df_test.loc[mask_te].reset_index(drop=True)
X_eval_pca = X_test_pca[mask_te]
y_true = np.vectorize(MAPEO_CLASES.get)(df_eval["species_id"].values)
print(f"Detecciones evaluadas (is_tp only={EVAL_ON_TP_ONLY}): {len(df_eval)}")

t0 = time.time()
proba = xgb.predict_proba(X_eval_pca)          # matriz
t_pred = time.time() - t0
y_pred = proba.argmax(axis=1)
P = proba.max(axis=1)                          # confianza

acc_global = accuracy_score(y_true, y_pred)
f1_global = f1_score(y_true, y_pred, average="macro")
print(f"\n[SIN moderación] Accuracy global: {acc_global*100:.2f}% | "
      f"F1-macro: {f1_global*100:.2f}% | tiempo inferencia: {t_pred*1000:.1f} ms "
      f"({t_pred/len(y_true)*1000:.3f} ms/detección)")

#Asignación de zonas operativas

def asignar_zona(p, p_high=P_HIGH, p_low=P_LOW):
    if p >= p_high:
        return "Confianza"
    elif p >= p_low:
        return "Incertidumbre"
    else:
        return "Rechazo"

zonas = np.array([asignar_zona(p) for p in P])
acierto = (y_pred == y_true)

df_det = pd.DataFrame({
    "recording_id": df_eval["recording_id"].values,
    "species_true": df_eval["species_id"].values,
    "species_pred": [INV_MAPEO[c] for c in y_pred],
    "P": P.round(4),
    "zona": zonas,
    "acierto": acierto,
})
df_det.to_csv("moderacion_resultados_por_deteccion.csv", index=False)

#Métricas por zona

orden_zonas = ["Confianza", "Incertidumbre", "Rechazo"]
filas = []
N = len(df_det)
for z in orden_zonas:
    sub = df_det[df_det["zona"] == z]
    n = len(sub)
    acc_z = sub["acierto"].mean() if n > 0 else np.nan
    filas.append({
        "Zona": z,
        "n": n,
        "% del total": f"{100*n/N:.1f}%",
        "Accuracy en zona": f"{acc_z*100:.1f}%" if n > 0 else "-",
    })
df_zonas = pd.DataFrame(filas)
df_zonas.to_csv("moderacion_resumen_zonas.csv", index=False)

print("\n=== RESUMEN POR ZONA OPERATIVA ===")
print(df_zonas.to_string(index=False))

# --- Indicadores operativos del sistema ---
n_verde = (zonas == "Confianza").sum()
n_amar = (zonas == "Incertidumbre").sum()
n_rojo = (zonas == "Rechazo").sum()
acc_verde = df_det.loc[df_det.zona == "Confianza", "acierto"].mean() if n_verde else np.nan

print("\n=== INDICADORES OPERATIVOS DEL SISTEMA ===")
print(f"Cobertura automática (zona verde)      : {100*n_verde/N:.1f}%  ({n_verde}/{N})")
print(f"Precisión auto-aceptada (control FP)   : {acc_verde*100:.1f}%  "
      f"vs {acc_global*100:.1f}% sin moderación  "
      f"(+{(acc_verde-acc_global)*100:.1f} pp)")
print(f"Carga de auditoría humana (zona amar.) : {100*n_amar/N:.1f}%  ({n_amar}/{N})")
print(f"Descartes automáticos (zona roja)      : {100*n_rojo/N:.1f}%  ({n_rojo}/{N})")

# Visualizaciones

COL_VERDE, COL_AMAR, COL_ROJO = "#2A9D8F", "#E9C46A", "#E76F51"

# --- Fig A: distribución de P con zonas sombreadas ---
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(0, 1, 41)
ax.axvspan(0, P_LOW, color=COL_ROJO, alpha=0.12)
ax.axvspan(P_LOW, P_HIGH, color=COL_AMAR, alpha=0.15)
ax.axvspan(P_HIGH, 1.0, color=COL_VERDE, alpha=0.15)
ax.hist(P, bins=bins, color="#264653", edgecolor="white", linewidth=0.6)
ax.axvline(P_LOW, color=COL_ROJO, linestyle="--", linewidth=2)
ax.axvline(P_HIGH, color=COL_VERDE, linestyle="--", linewidth=2)
ymax = ax.get_ylim()[1]
ax.text(P_LOW/2, ymax*0.92, f"Rechazo\n{n_rojo}", ha="center", va="top", color=COL_ROJO, fontweight="bold")
ax.text((P_LOW+P_HIGH)/2, ymax*0.92, f"Incertidumbre\n{n_amar}", ha="center", va="top", color="#B8860B", fontweight="bold")
ax.text((P_HIGH+1)/2, ymax*0.92, f"Confianza\n{n_verde}", ha="center", va="top", color=COL_VERDE, fontweight="bold")
ax.set_xlabel("Probabilidad predictiva máxima  P")
ax.set_ylabel("Número de detecciones")
ax.set_title("Distribución de la confianza P y zonas operativas")
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "umbral_distribucion_P.png"), dpi=200)
plt.close(fig)

# --- Fig B: accuracy por zona vs global ---
accs, cols, labels = [], [], []
for z, c in zip(orden_zonas, [COL_VERDE, COL_AMAR, COL_ROJO]):
    sub = df_det[df_det.zona == z]
    accs.append(sub["acierto"].mean()*100 if len(sub) else 0)
    cols.append(c)
    labels.append(f"{z}\n(n={len(sub)})")
accs.append(acc_global*100)
cols.append("#264653")
labels.append(f"GLOBAL\n(sin moderar)")

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, accs, color=cols, edgecolor="black", linewidth=0.6)
for b, a in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, a+1.5, f"{a:.1f}%", ha="center", fontweight="bold")
ax.axhline(acc_global*100, color="gray", linestyle=":", linewidth=1.5)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy por zona operativa vs. sistema sin moderación")
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "umbral_accuracy_por_zona.png"), dpi=200)
plt.close(fig)

# --- Fig C: sweep umbral -> cobertura vs precisión 
ths = np.linspace(0.30, 0.99, 70)
cobertura, precision = [], []
for t in ths:
    m = P >= t
    cobertura.append(100*m.mean())
    precision.append(100*acierto[m].mean() if m.any() else np.nan)

fig, ax1 = plt.subplots(figsize=(10, 6))
l1, = ax1.plot(ths, precision, color=COL_VERDE, linewidth=2.5, label="Precisión zona auto-aceptada")
ax1.set_xlabel("Umbral de confianza  P_high")
ax1.set_ylabel("Precisión auto-aceptada (%)", color=COL_VERDE)
ax1.tick_params(axis="y", labelcolor=COL_VERDE)
ax2 = ax1.twinx()
l2, = ax2.plot(ths, cobertura, color="#264653", linewidth=2.5, linestyle="--", label="Cobertura automática")
ax2.set_ylabel("Cobertura automática (%)", color="#264653")
ax2.tick_params(axis="y", labelcolor="#264653")
ax1.axvline(P_HIGH, color=COL_ROJO, linestyle=":", linewidth=2)
ax1.text(P_HIGH+0.005, ax1.get_ylim()[0]+3, f"P_high={P_HIGH}", color=COL_ROJO, fontweight="bold")
ax1.set_title("Trade-off: precisión vs. cobertura según el umbral")
ax1.legend(handles=[l1, l2], loc="lower center")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "umbral_sweep_cobertura_accuracy.png"), dpi=200)
plt.close(fig)

# Guardar modelo

joblib.dump({
    "scaler": scaler, "pca": pca, "modelo": xgb,
    "mapeo_clases": MAPEO_CLASES, "inv_mapeo": INV_MAPEO,
    "P_HIGH": P_HIGH, "P_LOW": P_LOW, "mel_cols": mel_cols,
}, "modelo_moderacion.joblib")

print("\nArtefactos guardados:")
print("  - moderacion_resultados_por_deteccion.csv")
print("  - moderacion_resumen_zonas.csv")
print("  - modelo_moderacion.joblib")
print("  - figures/umbral_distribucion_P.png")
print("  - figures/umbral_accuracy_por_zona.png")
print("  - figures/umbral_sweep_cobertura_accuracy.png")
