import os, joblib, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

base_dir = os.path.dirname(os.path.abspath(__file__))
ruta     = os.path.join(base_dir, "datos")

# Gasto promedio por categoría
df = pd.read_csv(f"{ruta}/shopping_trends.csv")
print("=== GASTO PROMEDIO POR CATEGORÍA ===")
print(df.groupby("Category")["Purchase Amount (USD)"].mean().apply(lambda x: f"${x:.2f} USD").to_string())

# Cargar datos procesados
X_train = pd.read_csv(f"{ruta}/X_train.csv")
X_test  = pd.read_csv(f"{ruta}/X_test.csv")
y_train = pd.read_csv(f"{ruta}/y_train.csv").iloc[:, 0]
y_test  = pd.read_csv(f"{ruta}/y_test.csv").iloc[:, 0]

# Entrenar y predecir
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Métricas
kw = dict(average="weighted", zero_division=0)
print("\n=== ÁRBOL DE DECISIÓN ===")
for nombre, valor in [
    ("Accuracy",  accuracy_score(y_test, y_pred)),
    ("Precision", precision_score(y_test, y_pred, **kw)),
    ("Recall",    recall_score(y_test, y_pred, **kw)),
    ("F1-Score",  f1_score(y_test, y_pred, **kw)),
]:
    print(f"{nombre:<10} {valor:.4f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Top 10 variables
top = pd.Series(modelo.feature_importances_, index=X_train.columns)
top = top[top > 0].sort_values(ascending=False).head(10)
print("\nTop 10 variables:")
for col, val in top.items():
    print(f"  {col:<32} {val*100:5.2f}%")

# Guardar
joblib.dump(modelo, f"{base_dir}/modelo_clasico.joblib")
print("\nModelo guardado: modelo_clasico.joblib")
