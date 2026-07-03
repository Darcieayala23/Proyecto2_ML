"""
==============================================================================
Interfaz Web Informativa — Clasificación Eco-Acústica (Sección 3.5 / bonus)
==============================================================================
Simula la inferencia del modelo optimizado (XGBoost + 11D-PCA) sobre escenarios
precargados (partición de prueba) y aplica la política de mitigación de riesgos
basada en umbrales probabilísticos (tres zonas operativas).

Ejecutar:   streamlit run app_streamlit.py
Requiere:   modelo_moderacion.joblib  (generado por umbrales_operativos.py)
            eco_acoustic_test.csv
==============================================================================
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ============================================================
# Configuración de página
# ============================================================
st.set_page_config(page_title="Clasificador Eco-Acústico", layout="wide")

# Catálogo taxonómico (Documentación del dataset, Cuadro 1)
TAXONOMIA = {
    10: ("Leptodactylus discodactylus", "Anfibio", "🐸"),
    12: ("Osteocephalus taurinus", "Anfibio", "🐸"),
    17: ("Chiroxiphia lineata", "Ave", "🐦"),
    18: ("Saltator grossus", "Ave", "🐦"),
    23: ("Pheucticus chrysopeplus", "Ave", "🐦"),
}

COL_VERDE, COL_AMAR, COL_ROJO = "#2A9D8F", "#E9C46A", "#E76F51"


# Carga de artefactos 

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_moderacion.joblib")


@st.cache_data
def cargar_test():
    return pd.read_csv("eco_acoustic_test.csv")


art = cargar_modelo()
df_test = cargar_test()

scaler, pca, modelo = art["scaler"], art["pca"], art["modelo"]
inv_mapeo, mel_cols = art["inv_mapeo"], art["mel_cols"]


# Inferencia

def inferir(row):
    x = row[mel_cols].to_numpy(dtype=float).reshape(1, -1)
    proba = modelo.predict_proba(pca.transform(scaler.transform(x)))[0]
    idx = int(proba.argmax())
    return proba, inv_mapeo[idx], float(proba.max())


def zona_de(p, p_high, p_low):
    if p >= p_high:
        return "Confianza", COL_VERDE
    elif p >= p_low:
        return "Incertidumbre", COL_AMAR
    return "Rechazo", COL_ROJO


# Barra lateral

st.sidebar.header("Política de moderación")
p_high = st.sidebar.slider("Umbral de Confianza  (P_high)", 0.50, 0.99,
                           float(art["P_HIGH"]), 0.01)
p_low = st.sidebar.slider("Umbral de Rechazo  (P_low)", 0.05, 0.60,
                          float(art["P_LOW"]), 0.01)
if p_low >= p_high:
    st.sidebar.error("P_low debe ser menor que P_high.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Escenario precargado")

solo_fiables = st.sidebar.checkbox("Solo detecciones fiables (is_tp=1)", value=True)
df_scen = df_test[df_test["is_tp"] == 1] if solo_fiables else df_test
ids = df_scen["recording_id"].tolist()

if "sel_idx" not in st.session_state:
    st.session_state.sel_idx = 0
if st.sidebar.button("Escenario aleatorio"):
    st.session_state.sel_idx = random.randrange(len(ids))

st.session_state.sel_idx = min(st.session_state.sel_idx, len(ids) - 1)
rid = st.sidebar.selectbox("recording_id", ids, index=st.session_state.sel_idx)
row = df_scen[df_scen["recording_id"] == rid].iloc[0]


# Cuerpo principal
st.title("Sistema de Clasificación de Señales Eco-Acústicas")
st.caption("Inferencia del modelo optimizado (XGBoost · 11D-PCA) con política de "
           "mitigación de riesgos por umbrales probabilísticos.")

proba, especie_pred, P = inferir(row)
zona, color = zona_de(P, p_high, p_low)
nombre_sci, fauna, emoji = TAXONOMIA[especie_pred]

#Banner de decisión operativa
if zona == "Confianza":
    st.success(f"### 🟢 Zona de Confianza — Clasificación automática\n"
               f"Se confirma la detección de **{nombre_sci}** con alta fiabilidad "
               f"(P = {P:.1%}).")
elif zona == "Incertidumbre":
    st.warning(f"### 🟡 Zona de Incertidumbre — Clasificación asistida\n"
               f"Evento catalogado como **dudoso** (P = {P:.1%}). Se sugiere enviar el "
               f"registro a la **cola de auditoría** de un experto humano.")
else:
    st.error(f"### 🔴 Zona de Rechazo — Descarte automático\n"
             f"Evento **omitido** por baja confianza (P = {P:.1%}) para mitigar el "
             f"impacto del ruido ambiental.")

#Métricas principales
c1, c2, c3, c4 = st.columns(4)
c1.metric("Especie predicha", f"{emoji} {especie_pred}", nombre_sci)
c2.metric("Tipo de fauna", fauna)
c3.metric("Confianza (P)", f"{P:.1%}")
c4.metric("Estado operativo", zona)

st.markdown("---")
col_izq, col_der = st.columns([3, 2])

#Gráfico del vector de probabilidades
with col_izq:
    st.subheader("Vector de probabilidades por especie")
    especies = [f"{sid}\n{TAXONOMIA[sid][0]}" for sid in inv_mapeo.values()]
    orden = np.argsort(proba)
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [color if inv_mapeo[i] == especie_pred else "#B8C4C2" for i in orden]
    ax.barh([especies[i] for i in orden], proba[orden] * 100, color=cols)
    for i, v in enumerate(proba[orden] * 100):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=11)
    ax.set_xlabel("Probabilidad (%)")
    ax.set_xlim(0, 100)
    ax.axvline(p_high * 100, color=COL_VERDE, ls="--", lw=1.5)
    ax.axvline(p_low * 100, color=COL_ROJO, ls="--", lw=1.5)
    fig.tight_layout()
    st.pyplot(fig)

# Información del escenario

with col_der:
    st.subheader("Escenario seleccionado")
    st.write(f"**recording_id:** `{rid}`")
    st.write(f"**songtype_id:** {row['songtype_id']}")
    st.write(f"**is_tp (fiabilidad):** {'✔️ True Positive' if row['is_tp'] == 1 else '➖ no confirmado'}")

    st.markdown("**Validación (ground-truth del escenario)**")
    especie_real = int(row["species_id"])
    real_sci = TAXONOMIA[especie_real][0]
    if especie_real == especie_pred:
        st.info(f"✅ Predicción correcta — especie real: {especie_real} ({real_sci})")
    else:
        st.error(f"Predicción incorrecta — especie real: {especie_real} ({real_sci})")
    st.caption("La etiqueta real solo se muestra con fines demostrativos; no interviene "
               "en la inferencia ni en la decisión de moderación.")

# Política de decisión

with st.expander("ℹ️ Definición de las zonas operativas"):
    st.markdown(f"""
    | Zona | Rango de confianza | Acción |
    |---|---|---|
    | 🟢 **Confianza** | P ≥ {p_high:.0%} | Clasificación automática |
    | 🟡 **Incertidumbre** | {p_low:.0%} ≤ P < {p_high:.0%} | Cola de auditoría humana |
    | 🔴 **Rechazo** | P < {p_low:.0%} | Descarte automático |
    """)
