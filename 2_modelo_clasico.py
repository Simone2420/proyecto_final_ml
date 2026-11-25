import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Rutas de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def obtener_directorio_datos():
    for carpeta in ["datos", "data"]:
        ruta = os.path.join(BASE_DIR, carpeta)
        if os.path.exists(os.path.join(ruta, "X_train.csv")):
            return ruta
    return os.path.join(BASE_DIR, "datos")

DATOS_DIR = obtener_directorio_datos()
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_clasico.joblib")


def main():
    print("=== MODELO CLASICO: ARBOL DE DECISION ===")
    
    # 1. Cargar datos preprocesados
    X_train = pd.read_csv(os.path.join(DATOS_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATOS_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATOS_DIR, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(DATOS_DIR, "y_test.csv")).iloc[:, 0]

    print(f"Datos cargados desde '{os.path.basename(DATOS_DIR)}/':")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Clases:  {list(y_train.unique())}\n")

    # 2. Entrenar modelo
    print("Entrenando DecisionTreeClassifier(max_depth=5)...")
    modelo = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
    modelo.fit(X_train, y_train)

    # 3. Predicciones y metricas sobre test
    y_pred = modelo.predict(X_test)
    promedio = "binary" if y_test.nunique() == 2 else "weighted"

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=promedio, zero_division=0)
    rec = recall_score(y_test, y_pred, average=promedio, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=promedio, zero_division=0)

    print("\n--- Metricas en Conjunto de Prueba ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nMatriz de Confusion:")
    print(confusion_matrix(y_test, y_pred))

    # 4. Importancia de caracteristicas
    importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
    imp_activas = importancias[importancias > 0].sort_values(ascending=False)

    print("\n--- Importancia de Caracteristicas (Top) ---")
    for i, (col, val) in enumerate(imp_activas.head(10).items(), 1):
        print(f" {i:2d}. {col:<32}: {val * 100:5.2f}%")

    # 5. Guardar modelo
    joblib.dump(modelo, RUTA_MODELO)
    print(f"\nModelo guardado en '{os.path.basename(RUTA_MODELO)}'.\n")

    # Generar graficas de apoyo si matplotlib esta disponible
    try:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        # Grafica de importancia
        plt.figure(figsize=(9, 5))
        imp_plot = imp_activas.sort_values(ascending=True)
        plt.barh(imp_plot.index, imp_plot.values * 100, color="#3470a3")
        plt.xlabel("Importancia Relativa (%)")
        plt.title("Importancia de Caracteristicas - Arbol de Decision")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "importancia_caracteristicas.png"), dpi=200)
        plt.close()

        # Diagrama del arbol
        plt.figure(figsize=(16, 8))
        plot_tree(
            modelo,
            max_depth=2,
            feature_names=list(X_train.columns),
            class_names=[str(c) for c in modelo.classes_],
            filled=True,
            rounded=True,
            fontsize=8
        )
        plt.title("Arbol de Decision (Niveles Principales)")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "arbol_decision_visual.png"), dpi=200)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
