# Taller Final Integrador de Machine Learning - Curso 601 / 602

## 🎯 Pregunta de Negocio y Formulación del Problema
> **Pregunta Central de Negocio:**  
> *«¿Qué categoría de producto (`Clothing`, `Footwear`, `Accessories`, `Outerwear`) comprará un cliente según su perfil demográfico, hábitos de compra y temporada?»*

### 💡 Justificación e Impacto de Negocio
- **Segmentación y Personalización:** Anticipar qué tipo de artículo busca un consumidor para recomendar ofertas personalizadas en tiendas físicas o e-commerce.
- **Gestión Inteligente de Inventario:** Optimizar el stock estacional por tienda y región según las tendencias de consumo observadas en cada temporada.
- **Eficiencia en Marketing:** Maximizar el retorno de inversión publicitaria (ROI) enviando promociones de las categorías con mayor probabilidad de conversión por cliente.

---

## 📊 Descripción del Dataset
- **Fuente:** `shopping_trends.csv` (ubicado en `data/`).
- **Registros:** 3,900 observaciones (cumple el rango exigido de 1,000 a 5,000).
- **Variables de entrada ($X$):** 14 variables de entrada seleccionadas tras limpieza (edad, monto de compra, compras previas, calificación de reseñas, temporada, suscripción, método de envío, etc.), expandidas a 112 características mediante One-Hot Encoding.
- **Variable objetivo ($y$):** `Category` (Clasificación Multiclase con 4 categorías: `Clothing`, `Accessories`, `Footwear`, `Outerwear`).

---

## 📁 Estructura del Repositorio
```text
proyecto_final_ml/
├── data/                         # Conjunto de datos crudo y particiones procesadas
│   ├── shopping_trends.csv       # Dataset original crudo (3,900 filas, 19 columnas)
│   ├── X_train.csv               # 80% entrenamiento (3,120 filas, 112 columnas)
│   ├── X_test.csv                # 20% prueba (780 filas, 112 columnas)
│   ├── y_train.csv               # 80% etiquetas de entrenamiento
│   └── y_test.csv                # 20% etiquetas de prueba
├── 1_preprocesamiento.py         # Pipeline de limpieza, imputación, One-Hot, StandardScaler y split (semilla 412)
├── 2_modelo_clasico.py           # Árbol de Decisión podado (Scikit-Learn) -> genera modelo_clasico.joblib
├── 3_red_neuronal.py             # Red Neuronal profunda (TensorFlow/Keras) -> genera modelo_red_neuronal.keras
├── 4_comparacion.py              # Script oficial de sustentación en vivo: tabla comparativa y conclusión
├── modelo_clasico.joblib         # Modelo clásico serializado
├── modelo_red_neuronal.keras     # Red neuronal entrenada (30 épocas)
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Documentación general y preguntas de negocio
```

---

## 🚀 Guía de Ejecución

### 1. Preprocesamiento de Datos
```bash
python 1_preprocesamiento.py
```
- Imputa valores nulos (mediana en numéricos, moda en categóricos).
- Aplica One-Hot Encoding y estandarización con `StandardScaler`.
- Divide en 80% Train y 20% Test con semilla fija obligatoria: `random_state=412`.

### 2. Modelo Clásico (Scikit-Learn)
```bash
python 2_modelo_clasico.py
```
- Entrena un **Árbol de Decisión (`max_depth=5`)**, evitando sobreajuste.
- Imprime las reglas del árbol, matriz de confusión y métricas sobre el conjunto de prueba.
- Exporta `modelo_clasico.joblib`.

### 3. Red Neuronal con TensorFlow/Keras
```bash
python 3_red_neuronal.py
```
- Arquitectura secuencial: `Dense(64, relu) -> Dense(32, relu) -> Dense(4, softmax)`.
- Compilación con optimizador `adam` y función de pérdida `sparse_categorical_crossentropy`.
- Entrena durante 30 épocas y exporta `modelo_red_neuronal.keras`.

### 4. Sustentación Oral y Demostración en Vivo
```bash
python 4_comparacion.py
```
- Evalúa ambos modelos bajo las mismas condiciones exactas sobre `X_test`.
- Despliega la tabla comparativa lado a lado e imprime los argumentos técnicos de defensa.

---

## 🏆 Resumen Comparativo de Métricas (Test Set)

| Métrica | Árbol de Decisión (Scikit-Learn) | Red Neuronal (TensorFlow/Keras) | Modelo Superior |
| :--- | :---: | :---: | :---: |
| **Accuracy (Exactitud)** | **0.4346 (43.46%)** | 0.3500 (35.00%) | 🌲 **Árbol de Decisión** (+8.46%) |
| **Precision (Ponderada)** | 0.3135 | **0.3265** | 🧠 **Red Neuronal** |
| **Recall (Sensibilidad)** | **0.4346** | 0.3500 | 🌲 **Árbol de Decisión** |
| **F1-Score (Ponderado)** | 0.3125 | **0.3372** | 🧠 **Red Neuronal** |
| **Latencia Inferencia** | **~50 ms** | ~200 ms | 🌲 **Árbol de Decisión** (4x más rápido) |

---

## 🎤 Argumentos para la Sustentación Oral (5 - 8 minutos)
1. **Control de Sobreajuste:** Un árbol sin podar memoriza los datos (100% train) pero colapsa en test (~31%). Al regular con `max_depth=5`, se maximiza la capacidad de generalización alcanzando **43.46%** en prueba.
2. **Superioridad en Datos Tabulares:** Con 3,900 registros y 112 variables dispersas, los cortes ortogonales del árbol de decisión son más efectivos que las transformaciones densas de la red neuronal.
3. **Explicabilidad para el Negocio:** El árbol de decisión permite trazar reglas directas basadas en variables clave como `Review Rating`, `Age` y `Purchase Amount`, convirtiéndolo en la opción más auditable para la organización.
