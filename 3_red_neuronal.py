import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import LabelEncoder

# Fijar semilla para reproducibilidad
tf.random.set_seed(412)

# 1. Cargar los conjuntos preprocesados desde la carpeta datos/
ruta_datos = os.path.join(os.path.dirname(__file__), "datos")
X_train = pd.read_csv(os.path.join(ruta_datos, "X_train.csv"))
X_test = pd.read_csv(os.path.join(ruta_datos, "X_test.csv"))
y_train = pd.read_csv(os.path.join(ruta_datos, "y_train.csv"))["Category"]
y_test = pd.read_csv(os.path.join(ruta_datos, "y_test.csv"))["Category"]

# 2. Codificar las etiquetas de texto (Category) a valores numéricos (0, 1, 2, 3)
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)
num_clases = len(encoder.classes_)

# 3. Definir la arquitectura secuencial (Sequential)
modelo = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(num_clases, activation="softmax"),
])

# 4. Compilar el modelo con optimizador Adam y pérdida para multiclase
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 5. Entrenar el modelo monitoreando pérdida y precisión
print("Entrenando Red Neuronal...")
modelo.fit(
    X_train,
    y_train_enc,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# 6. Evaluar sobre el conjunto de prueba
loss_test, acc_test = modelo.evaluate(X_test, y_test_enc, verbose=0)
print("\n=== Evaluación Final (Red Neuronal - Test) ===")
print(f"Pérdida (Loss):     {loss_test:.4f}")
print(f"Precisión (Acc):    {acc_test:.4f}")

# 7. Guardar el modelo entrenado en la raiz del proyecto para uso futuro
modelo.save(os.path.join(os.path.dirname(__file__), "modelo_red_neuronal.keras"))
print("Modelo guardado como 'modelo_red_neuronal.keras' en la raíz del proyecto.")