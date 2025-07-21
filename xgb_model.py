# --------------------------------------------------------
#     Lectura de módulos necesarios para la ejecución
# --------------------------------------------------------
# Manejo de datos estructurados
import pandas as pd  # Manipulación de datos: lectura, filtrado, transformación y análisis de DataFrames
import numpy as np   # Operaciones matemáticas y algebraicas con arrays, funciones estadísticas, etc.
# --------------------------------------------------------
# Visualización de datos
import matplotlib.pyplot as plt  # Generación de gráficos básicos (líneas, barras, histogramas, etc.)
import seaborn as sns            # Visualización estadística avanzada con estética mejorada sobre matplotlib
# --------------------------------------------------------
# División y preprocesamiento de los datos
from sklearn.model_selection import train_test_split  # División del conjunto de datos en entrenamiento y prueba
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Codificación categórica y escalado de variables
from sklearn.compose import ColumnTransformer  # Aplicar diferentes transformaciones a columnas específicas
from sklearn.preprocessing import LabelEncoder  # Codificación de etiquetas categóricas a numéricas
# --------------------------------------------------------
# Creación de flujos de procesamiento
from sklearn.pipeline import Pipeline  # Encadenamiento de transformaciones y modelo en un solo flujo (pipeline)
# --------------------------------------------------------
# Modelo predictivo
from sklearn.linear_model import LogisticRegression  # Modelo de regresión logística (clasificación multiclase o binaria)
from xgboost import XGBClassifier # Modelo de clasificación basado en árboles de decisión (XGBoost)
# --------------------------------------------------------
# Evaluación del rendimiento del modelo
from sklearn.metrics import (
    classification_report,  # Reporte con métricas como precisión, recall y F1
    confusion_matrix,       # Matriz de confusión para evaluar errores de clasificación
    roc_auc_score,          # Área bajo la curva ROC, útil para evaluación binaria o multiclase con binarización
    roc_curve,              # Cálculo de la curva ROC (TPR vs FPR)
    f1_score,               # Métrica F1: armónica entre precisión y recall
    accuracy_score          # Exactitud del modelo (proporción de predicciones correctas)
)
# --------------------------------------------------------
# Optimización de hiperparámetros
import optuna  # Framework de optimización bayesiana para búsqueda eficiente de hiperparámetros
# --------------------------------------------------------
# Utilidades para clasificación multiclase
from sklearn.preprocessing import label_binarize  # Conversión de etiquetas multiclase a formato binarizado
from sklearn.multiclass import OneVsRestClassifier  # Clasificador multiclase: un modelo por clase contra el resto
# --------------------------------------------------------
import support  # Módulo personalizado para utilidades de visualización y análisis

# --------------------------------------------------------
#       1. Carga de datos y procesamiento inicial
# --------------------------------------------------------
path = r'C:\Users\sacah\OneDrive - Universidad Nacional de Colombia\BOOTCAMP\data\BD_DEPURADA.xlsx'
data = pd.read_excel(path)
data = data.iloc[:, 1:]
data = data.dropna(subset = ["MINERALES"])

# Creación variable objetivo (codificada)
# Esto es, el departamento a predecir
le = LabelEncoder()
data["DEPARTAMENTO_LABEL"] = le.fit_transform(data["DEPARTAMEN"])

# Eliminar clases con 1 muestra:
# Pues no tendría sentido entrenar un modelo con una única observación
counts = data["DEPARTAMENTO_LABEL"].value_counts()
valid_classes = counts[counts >= 2].index
data = data[data["DEPARTAMENTO_LABEL"].isin(valid_classes)]

# Codificación de una nueva variable temporal útil
data["FECHA_ANTIGUEDAD"] = (pd.Timestamp.today() - data["FECHA_DE_I"]).dt.days

# Determinación de variables predictoras
# El municipio se remueve pues no tendría mucho sentido incluirlo
num_features = ["CANTIDAD_MINERALES_TITULO", "AREA_DIVISION", "FECHA_ANTIGUEDAD"]
cat_features = ["CATEGORIA_MINERAL"]
target = "DEPARTAMENTO_LABEL"
X = data[num_features + cat_features]
y = data[target]
# --------------------------------------------------------
#      2. Estandardización y codificación de variables
# --------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), cat_features)])
# --------------------------------------------------------
#      3. División de datos para entrenamiento y prueba
# --------------------------------------------------------
# Se realiza una división estratificada para mantener la proporción de clases en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 7)

# --------------------------------------------------------
#         4. Definición del modelo XGBoost
# --------------------------------------------------------
clf = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(
        n_estimators = 100,
        max_depth = 6,
        learning_rate = 0.1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder = False,
        eval_metric = "mlogloss",
        objective = "multi:softprob",
        num_class = len(np.unique(y)),
        random_state = 7))])
clf.fit(X_train, y_train)

# --------------------------------------------------------
#            5. Evaluación del modelo
# --------------------------------------------------------
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("F1 Macro:", round(f1_score(y_test, y_pred, average = "macro"), 4))

# --------------------------------------------------------
#               7. Matriz de confusión
# --------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
# --------------------------------------------------------
# Crear figura y ejes
fig, ax = plt.subplots(figsize=(10, 8))
# --------------------------------------------------------
# Dibujar heatmap sobre los ejes
sns.heatmap(cm, annot=False, fmt='d', cmap="Blues", ax=ax)
# --------------------------------------------------------
# Aplicar plantilla personalizada
support.custom_theme(ax, title="Matriz de Confusión", subtitle="Clasificación del modelo")
# --------------------------------------------------------
# Mostrar gráfico
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Ver el mapeo de códigos numéricos a nombres de departamentos
label_mapping = dict(enumerate(le.classes_))
for k, v in label_mapping.items():
    print(f"{k}: {v}")