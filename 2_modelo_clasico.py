"""
=============================================================================
PROYECTO FINAL INTEGRADOR DE MACHINE LEARNING - SCRIPT 2
Archivo: 2_modelo_clasico.py
Descripción:
    Este script implementa, entrena y evalúa un modelo tradicional de 
    Machine Learning utilizando Scikit-Learn a partir de los datos
    preprocesados en la carpeta 'data/' o 'datos/'.
    
    Responsabilidades:
    1. Cargar conjuntos preprocesados (X_train, X_test, y_train, y_test).
    2. Detectar automáticamente o configurar el tipo de problema (Clasificación o Regresión).
    3. Instanciar y entrenar un modelo clásico (Random Forest / Logistic Regression / Regresión Lineal).
    4. Generar predicciones sobre el conjunto de prueba no visto (X_test).
    5. Calcular e imprimir métricas estándar (Accuracy, Precision, Recall, F1-Score / MSE, MAE, R²).
    6. Exportar el modelo entrenado (modelo_clasico.joblib) para el script comparativo final.
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Modelos Scikit-Learn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text

# Métricas de evaluación
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =============================================================================
# CONFIGURACIÓN Y RUTAS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def obtener_directorio_datos():
    """Busca los datos en 'data' o 'datos' de forma dinámica."""
    for carpeta in ["data", "datos"]:
        ruta = os.path.join(BASE_DIR, carpeta)
        if os.path.exists(os.path.join(ruta, "X_train.csv")):
            return ruta
    return os.path.join(BASE_DIR, "data")

DATOS_DIR = obtener_directorio_datos()
RUTA_MODELO_GUARDADO = os.path.join(BASE_DIR, "modelo_clasico.joblib")
RUTA_METRICAS_GUARDADAS = os.path.join(BASE_DIR, "metricas_clasico.json")

# Configuración del tipo de problema: 'auto', 'clasificacion' o 'regresion'
TIPO_PROBLEMA = "auto"

# Modelo a utilizar:
# Para Clasificación: 'decision_tree' (recomendado para sustentación e interpretabilidad), 'random_forest', 'logistic_regression'
# Para Regresión: 'random_forest', 'linear_regression'
ALGORITMO_ELEGIDO = "decision_tree"
RANDOM_STATE = 42


# =============================================================================
# FUNCIONES MODULARES
# =============================================================================

def cargar_datos():
    """
    Carga los conjuntos de datos preprocesados generados por el Script 1.
    """
    rutas = {
        "X_train": os.path.join(DATOS_DIR, "X_train.csv"),
        "X_test": os.path.join(DATOS_DIR, "X_test.csv"),
        "y_train": os.path.join(DATOS_DIR, "y_train.csv"),
        "y_test": os.path.join(DATOS_DIR, "y_test.csv"),
    }

    print("\n" + "=" * 70)
    print(f" 📂 PASO 1: CARGA DE CONJUNTOS PREPROCESADOS (desde '{os.path.basename(DATOS_DIR)}/')")
    print("=" * 70)

    # Verificación de existencia de archivos
    faltantes = [nombre for nombre, ruta in rutas.items() if not os.path.exists(ruta)]
    if faltantes:
        raise FileNotFoundError(
            f"\n❌ ERROR: Faltan archivos en '{DATOS_DIR}': {', '.join(faltantes)}.\n"
            f"   Asegúrate de haber ejecutado previamente '1_preprocesamiento.py'."
        )

    X_train = pd.read_csv(rutas["X_train"])
    X_test = pd.read_csv(rutas["X_test"])
    
    df_y_train = pd.read_csv(rutas["y_train"])
    df_y_test = pd.read_csv(rutas["y_test"])
    
    y_train = df_y_train.iloc[:, 0] if isinstance(df_y_train, pd.DataFrame) else df_y_train
    y_test = df_y_test.iloc[:, 0] if isinstance(df_y_test, pd.DataFrame) else df_y_test

    print(f" ✔ X_train cargado exitosamente: {X_train.shape[0]} filas, {X_train.shape[1]} columnas")
    print(f" ✔ X_test  cargado exitosamente: {X_test.shape[0]} filas, {X_test.shape[1]} columnas")
    print(f" ✔ y_train cargado exitosamente: {len(y_train)} etiquetas")
    print(f" ✔ y_test  cargado exitosamente: {len(y_test)} etiquetas")

    return X_train, X_test, y_train, y_test


def inferir_tipo_problema(y_train):
    """
    Determina si la variable objetivo corresponde a un problema de
    Clasificación o Regresión de forma automática si TIPO_PROBLEMA == 'auto'.
    """
    if TIPO_PROBLEMA in ["clasificacion", "regresion"]:
        return TIPO_PROBLEMA

    num_unicos = y_train.nunique()
    es_flotante = pd.api.types.is_float_dtype(y_train)

    if not es_flotante or num_unicos <= 15:
        return "clasificacion"
    else:
        return "regresion"


def instanciar_modelo(tipo_problema, algoritmo="random_forest"):
    """
    Crea e inicializa el estimador clásico de Scikit-Learn según el tipo de problema.
    """
    if tipo_problema == "clasificacion":
        if algoritmo == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ), "Random Forest Classifier"
        elif algoritmo == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ), "Regresión Logística"
        elif algoritmo == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=5,
                criterion="gini",
                random_state=RANDOM_STATE
            ), "Árbol de Decisión Classifier"
        else:
            raise ValueError(f"Algoritmo desconocido para clasificación: {algoritmo}")

    elif tipo_problema == "regresion":
        if algoritmo == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ), "Random Forest Regressor"
        elif algoritmo == "linear_regression":
            return LinearRegression(), "Regresión Lineal Múltiple"
        else:
            raise ValueError(f"Algoritmo desconocido para regresión: {algoritmo}")
    else:
        raise ValueError(f"Tipo de problema inválido: {tipo_problema}")


def evaluar_clasificacion(modelo, X_test, y_test, nombre_modelo):
    """
    Calcula métricas detalladas para problemas de clasificación.
    """
    y_pred = modelo.predict(X_test)

    # Identificar si es binaria o multiclase
    promedio_metricas = "binary" if y_test.nunique() == 2 else "weighted"

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=promedio_metricas, zero_division=0)
    rec = recall_score(y_test, y_pred, average=promedio_metricas, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=promedio_metricas, zero_division=0)
    matriz_conf = confusion_matrix(y_test, y_pred)

    metricas = {
        "tipo": "clasificacion",
        "modelo": nombre_modelo,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "promedio_utilizado": promedio_metricas
    }

    print("\n" + "=" * 70)
    print(f" 📊 PASO 3: EVALUACIÓN DE DESEMPEÑO - {nombre_modelo.upper()}")
    print("=" * 70)
    print(f" 🎯 Accuracy  (Exactitud): {acc:.4f}  ({acc * 100:.2f}%)")
    print(f" 🎯 Precision (Precisión): {prec:.4f}")
    print(f" 🎯 Recall    (Sensibilidad): {rec:.4f}")
    print(f" 🎯 F1-Score  (Balance):   {f1:.4f}")

    print("\n 🔍 Matriz de Confusión:")
    print(matriz_conf)

    print("\n 📋 Reporte Detallado por Clase:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metricas, y_pred


def evaluar_regresion(modelo, X_test, y_test, nombre_modelo):
    """
    Calcula métricas detalladas para problemas de regresión.
    """
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metricas = {
        "tipo": "regresion",
        "modelo": nombre_modelo,
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }

    print("\n" + "=" * 70)
    print(f" 📊 PASO 3: EVALUACIÓN DE DESEMPEÑO - {nombre_modelo.upper()}")
    print("=" * 70)
    print(f" 🎯 MSE  (Error Cuadrático Medio):  {mse:.4f}")
    print(f" 🎯 RMSE (Raíz del MSE):             {rmse:.4f}")
    print(f" 🎯 MAE  (Error Absoluto Medio):    {mae:.4f}")
    print(f" 🎯 R²   (Coeficiente Determinación): {r2:.4f}  ({r2 * 100:.2f}%)")

    return metricas, y_pred


def guardar_artefactos(modelo, metricas, feature_names):
    """
    Serializa el modelo entrenado con joblib y guarda las métricas en JSON.
    """
    print("\n" + "=" * 70)
    print(" 💾 PASO 4: EXPORTACIÓN DEL MODELO Y METADATOS")
    print("=" * 70)

    # 1. Guardar modelo
    joblib.dump(modelo, RUTA_MODELO_GUARDADO)
    print(f" ✔ Modelo clásico guardado en: '{RUTA_MODELO_GUARDADO}'")

    # 2. Guardar métricas y características
    info = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "caracteristicas": list(feature_names),
        "num_caracteristicas": len(feature_names),
        "metricas": metricas
    }
    with open(RUTA_METRICAS_GUARDADAS, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print(f" ✔ Métricas exportadas a:     '{RUTA_METRICAS_GUARDADAS}'")


def ejecutar_entrenamiento_clasico():
    """
    Ejecuta el ciclo de vida completo del modelo clásico de Scikit-Learn.
    """
    print("=" * 70)
    print(" 🚀 INICIANDO SCRIPT 2: MODELO CLÁSICO (SCIKIT-LEARN)")
    print("=" * 70)

    # 1. Cargar datos preprocesados
    X_train, X_test, y_train, y_test = cargar_datos()

    # 2. Determinar tipo de problema
    tipo_problema = inferir_tipo_problema(y_train)
    clases_detectadas = y_train.unique() if tipo_problema == "clasificacion" else None
    print(f"\n 📌 Tipo de problema detectado: [{tipo_problema.upper()}]")
    if clases_detectadas is not None:
        print(f" 📌 Clases detectadas ({len(clases_detectadas)}): {list(clases_detectadas)}")

    # 3. Instanciar modelo
    modelo, nombre_modelo = instanciar_modelo(tipo_problema, ALGORITMO_ELEGIDO)
    print("\n" + "=" * 70)
    print(f" ⚙️  PASO 2: ENTRENANDO MODELO CLÁSICO [{nombre_modelo}]")
    print("=" * 70)

    inicio = datetime.now()
    modelo.fit(X_train, y_train)
    duracion = (datetime.now() - inicio).total_seconds()
    print(f" ✔ Entrenamiento completado en {duracion:.3f} segundos.")

    # 4. Evaluación sobre el conjunto de prueba no visto
    if tipo_problema == "clasificacion":
        metricas, y_pred = evaluar_clasificacion(modelo, X_test, y_test, nombre_modelo)
    else:
        metricas, y_pred = evaluar_regresion(modelo, X_test, y_test, nombre_modelo)

    # 5. Análisis de importancia de características y reglas del árbol
    if isinstance(modelo, DecisionTreeClassifier):
        print("\n" + "=" * 70)
        print(" 🌲 REGLAS PRINCIPALES DEL ÁRBOL DE DECISIÓN (Primeros niveles)")
        print("=" * 70)
        reglas = export_text(modelo, max_depth=2, feature_names=list(X_train.columns))
        print(reglas)

    if hasattr(modelo, "feature_importances_"):
        print("\n" + "=" * 70)
        print(" 🌟 IMPORTANCIA DE CARACTERÍSTICAS (Top 5 más influyentes)")
        print("=" * 70)
        importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
        top5 = importancias.sort_values(ascending=False).head(5)
        for col, val in top5.items():
            barra = "█" * int(val * 40)
            print(f" • {col:<30}: {val:.4f} {barra}")

    # 6. Guardar modelo y métricas
    guardar_artefactos(modelo, metricas, X_train.columns)

    print("\n" + "=" * 70)
    print(" ✅ SCRIPT 2 FINALIZADO CON ÉXITO")
    print("    El modelo se encuentra listo para ser comparado en '4_comparacion.py'.")
    print("=" * 70 + "\n")

    return modelo, metricas


if __name__ == "__main__":
    ejecutar_entrenamiento_clasico()
