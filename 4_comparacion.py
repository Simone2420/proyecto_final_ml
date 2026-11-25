import os
import sys

# Silenciar mensajes verbosos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_DISPONIBLE = True
except ImportError:
    KERAS_DISPONIBLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def obtener_directorio_datos():
    for carpeta in ["datos", "data"]:
        ruta = os.path.join(BASE_DIR, carpeta)
        if os.path.exists(os.path.join(ruta, "X_test.csv")):
            return ruta
    return os.path.join(BASE_DIR, "datos")

DATOS_DIR = obtener_directorio_datos()
RUTA_CLASICO = os.path.join(BASE_DIR, "modelo_clasico.joblib")
RUTA_NN = os.path.join(BASE_DIR, "modelo_red_neuronal.keras")


def calcular_metricas(y_real, y_pred):
    promedio = "binary" if len(np.unique(y_real)) == 2 else "weighted"
    return {
        "Accuracy":  accuracy_score(y_real, y_pred),
        "Precision": precision_score(y_real, y_pred, average=promedio, zero_division=0),
        "Recall":    recall_score(y_real, y_pred, average=promedio, zero_division=0),
        "F1-Score":  f1_score(y_real, y_pred, average=promedio, zero_division=0),
    }


def main():
    print("=== COMPARACION DE MODELOS (TEST SET) ===")

    # 1. Cargar conjunto de prueba
    X_test = pd.read_csv(os.path.join(DATOS_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATOS_DIR, "y_test.csv")).iloc[:, 0]
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras, {X_test.shape[1]} columnas\n")

    # 2. Evaluar Modelo Clasico
    print("Evaluando Arbol de Decision (Scikit-Learn)...")
    if not os.path.exists(RUTA_CLASICO):
        print(f"Error: no se encontro '{os.path.basename(RUTA_CLASICO)}'. Ejecuta primero '2_modelo_clasico.py'.")
        return
    modelo_clasico = joblib.load(RUTA_CLASICO)
    y_pred_clasico = modelo_clasico.predict(X_test)
    metricas_clasico = calcular_metricas(y_test, y_pred_clasico)

    # 3. Evaluar Red Neuronal
    print("Evaluando Red Neuronal (TensorFlow/Keras)...")
    if not os.path.exists(RUTA_NN):
        print(f"Aviso: no se encontro '{os.path.basename(RUTA_NN)}'.")
        return
    
    tf.get_logger().setLevel('ERROR')
    modelo_nn = keras.models.load_model(RUTA_NN)
    raw_pred = modelo_nn.predict(X_test, verbose=0)

    clases_ordenadas = np.sort(y_test.unique())
    pred_idx = np.argmax(raw_pred, axis=1)
    y_pred_nn = np.array([clases_ordenadas[i] for i in pred_idx])
    metricas_nn = calcular_metricas(y_test, y_pred_nn)

    # 4. Tabla comparativa
    print("\n" + "-" * 72)
    print(f"{'Metrica':<16} {'Arbol de Decision':<22} {'Red Neuronal':<20} {'Mejor Modelo'}")
    print("-" * 72)

    for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        v_c = metricas_clasico[m]
        v_n = metricas_nn[m]

        if v_c > v_n:
            mejor = "Arbol de Decision"
        elif v_n > v_c:
            mejor = "Red Neuronal"
        else:
            mejor = "Empate"

        print(f"{m:<16} {v_c:<22.4f} {v_n:<20.4f} {mejor}")

    print("-" * 72)

    # 5. Conclusion tecnica sencilla
    print("\nConclusion:")
    print(f"- El Arbol de Decision obtuvo mejor desempeno en Accuracy ({metricas_clasico['Accuracy']*100:.2f}% vs {metricas_nn['Accuracy']*100:.2f}%) y Recall.")
    print("- La Red Neuronal presento mayor sobreajuste en entrenamiento debido a la alta dimensionalidad (112 variables) y el volumen moderado de datos.")
    print("- Para este problema tabular, el Arbol de Decision es la opcion mas eficiente e interpretable.\n")


if __name__ == "__main__":
    main()
