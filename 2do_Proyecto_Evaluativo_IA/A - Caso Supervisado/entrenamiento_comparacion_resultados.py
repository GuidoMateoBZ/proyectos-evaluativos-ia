import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Cargar el dataset

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target # 0: Maligno, 1: Benigno

# 4. Separar entrenamiento y prueba con stratify

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar los datos ( para Regresión Logística, MLP y KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenar cinco modelos

modelos = {
    "Regresión Logística": LogisticRegression(random_state=42, max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Red Neuronal (MLP)": MLPClassifier(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier()
}

# 6. Comparar accuracy, precision, recall y F1

print("--- 6. Desempeño de Modelos ---")
resultados = []

for nombre, modelo in modelos.items():
    # Usar datos escalados para los que dependen de distancias/gradientes
    if nombre in ["Regresión Logística", "Red Neuronal (MLP)", "KNN"]:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

df_resultados = pd.DataFrame(resultados).set_index("Modelo")
print(df_resultados.round(4))
print("\n")

# 7. Determinar el mejor modelo empíricamente (mayor F1-Score)
#    Y mostrar Matriz de confusión
mejor_modelo_nombre = df_resultados['F1-Score'].idxmax()
mejor_modelo = modelos[mejor_modelo_nombre]

print(f"Mejor Moodelo según F1: {mejor_modelo_nombre}")

print(f"--- 7. Matriz de Confusión del Modelo: {mejor_modelo_nombre} ---")
X_test_cm = X_test_scaled if mejor_modelo_nombre in ["Regresión Logística", "Red Neuronal (MLP)", "KNN"] else X_test

disp = ConfusionMatrixDisplay.from_estimator(
    mejor_modelo, X_test_cm, y_test, 
    display_labels=data.target_names, cmap=plt.cm.Blues
)
plt.title(f"Matriz de Confusión - {mejor_modelo_nombre}")
plt.show()

# 8. Optimizar hiperparámetros con GridSearchCV (Random Forest)

print("--- 8. Optimización de hiperparámetros (Random Forest) ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor F1-Score CV: {grid_search.best_score_:.4f}\n")

# 9. Analizar importancia de variables en Random Forest

print("--- 9. Importancia de Variables (Top 10) ---")
mejor_rf = grid_search.best_estimator_
importancias = pd.Series(mejor_rf.feature_importances_, index=X.columns)
importancias_top10 = importancias.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
importancias_top10.plot(kind='barh', color='skyblue')
plt.title("Top 10 Variables más importantes (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

# 10. Entrenar árbol interpretable reducido 

print("--- 10. Árbol de Decisión Interpretable (Reducido) ---")
# profundidad a 2 para facilitar la lectura de reglas
arbol_reducido = DecisionTreeClassifier(max_depth=2, random_state=42)
arbol_reducido.fit(X_train, y_train)

reglas = export_text(arbol_reducido, feature_names=list(X.columns))
print("Reglas Principales:\n", reglas)

plt.figure(figsize=(12, 8))
plot_tree(arbol_reducido, feature_names=list(X.columns), class_names=list(data.target_names), filled=True, rounded=True)
plt.title("Estructura del Árbol Reducido")
plt.show()