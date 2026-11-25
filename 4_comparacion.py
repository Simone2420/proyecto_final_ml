import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np, pandas as pd, joblib, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metricas(y_real, y_pred):
    return {
        "Accuracy":  accuracy_score(y_real, y_pred),
        "Precision": precision_score(y_real, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_real, y_pred, average="weighted", zero_division=0),
        "F1-Score":  f1_score(y_real, y_pred, average="weighted", zero_division=0),
    }

# Cargar datos
base_dir = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(base_dir, "datos")
X_test = pd.read_csv(f"{ruta}/X_test.csv")
y_test = pd.read_csv(f"{ruta}/y_test.csv").iloc[:, 0]

# Predicciones
arbol = joblib.load(os.path.join(base_dir, "modelo_clasico.joblib"))
m_arbol = metricas(y_test, arbol.predict(X_test))

tf.get_logger().setLevel("ERROR")
nn = keras.models.load_model(os.path.join(base_dir, "modelo_red_neuronal.keras"))
clases = np.sort(y_test.unique())
m_nn = metricas(y_test, clases[np.argmax(nn.predict(X_test, verbose=0), axis=1)])

# Tabla comparativa
print("=== COMPARACIÓN DE MODELOS ===\n")
print(f"{'Métrica':<12} {'Árbol':>10} {'Red Neuronal':>14} {'Mejor':>18}")
print("-" * 58)
for k in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    a, n = m_arbol[k], m_nn[k]
    mejor = "Árbol" if a > n else ("Red Neuronal" if n > a else "Empate")
    print(f"{k:<12} {a:>10.4f} {n:>14.4f} {mejor:>18}")
