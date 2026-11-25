import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Cargar el dataset original desde la carpeta datos/
ruta_datos = os.path.join(os.path.dirname(__file__), "datos")
df = pd.read_csv(os.path.join(ruta_datos, "shopping_trends.csv"))

print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")

# 2. Definir variable objetivo (y) y variables de entrada (X)
# Se eliminan ID, Item Purchased (evitar fuga de datos) y redundantes para mantener 14 variables (rango 6-15)
columnas_excluir = [
    "Customer ID",
    "Item Purchased",
    "Preferred Payment Method",
    "Promo Code Used",
    "Category",
]
X = df.drop(columns=columnas_excluir)
y = df["Category"]

# Identificar variables numéricas y categóricas
cols_numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cols_categoricas = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

# 3. Imputación de valores nulos (mediana para numéricos, moda para categóricos)
imputador_num = SimpleImputer(strategy="median")
X_num = imputador_num.fit_transform(X[cols_numericas])

imputador_cat = SimpleImputer(strategy="most_frequent")
X_cat = imputador_cat.fit_transform(X[cols_categoricas])

# 4. Codificación One-Hot Encoding para variables categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat_enc = encoder.fit_transform(X_cat)
df_cat = pd.DataFrame(X_cat_enc, columns=encoder.get_feature_names_out(cols_categoricas))

# 5. Escalado de variables numéricas con StandardScaler
escalador = StandardScaler()
X_num_scaled = escalador.fit_transform(X_num)
df_num = pd.DataFrame(X_num_scaled, columns=cols_numericas)

# Unir variables numéricas escaladas y categóricas codificadas
X_procesado = pd.concat([df_num, df_cat], axis=1)

# 6. Dividir el dataset en 80% entrenamiento y 20% prueba con semilla fija (random_state=412)
X_train, X_test, y_train, y_test = train_test_split(
    X_procesado, y, test_size=0.2, random_state=412, stratify=y
)

# 7. Exportar los cuatro archivos procesados a datos/
X_train.to_csv(os.path.join(ruta_datos, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(ruta_datos, "X_test.csv"), index=False)
y_train.to_frame(name="Category").to_csv(os.path.join(ruta_datos, "y_train.csv"), index=False)
y_test.to_frame(name="Category").to_csv(os.path.join(ruta_datos, "y_test.csv"), index=False)

print("Preprocesamiento completado con éxito:")
print(f" - X_train: {X_train.shape} | Nulos: {X_train.isnull().sum().sum()}")
print(f" - X_test:  {X_test.shape} | Nulos: {X_test.isnull().sum().sum()}")
print(f" - y_train: {y_train.shape}")
print(f" - y_test:  {y_test.shape}")
