
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Recargar dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 4.1.
# 4 Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*45)
print("DIVISIÓN DEL DATASET")
print("="*45)
print(f"  Total de registros : {len(X)}")
print(f"  Entrenamiento (80%): {X_train.shape[0]} registros")
print(f"  Prueba        (20%): {X_test.shape[0]} registros")
print("="*45)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.1 
# 5 Entrenar cinco modelos

modelos = {
    "Regresión Logística": (LogisticRegression(max_iter=10000, random_state=42), True),
    "Árbol de Decisión":   (DecisionTreeClassifier(random_state=42), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "Red Neuronal (MLP)":  (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42), True),
    "SVM":                 (SVC(random_state=42), True),
}

print("\n" + "="*55)
print("ENTRENAMIENTO Y ACCURACY POR MODELO")
print("="*55)

resultados = {}

for nombre, (modelo, escalar) in modelos.items():
    if escalar:
        modelo.fit(X_train_scaled, y_train)
        acc = modelo.score(X_test_scaled, y_test)
    else:
        modelo.fit(X_train, y_train)
        acc = modelo.score(X_test, y_test)
    resultados[nombre] = acc

mejor = max(resultados, key=resultados.get)

for nombre, acc in resultados.items():
    marca = " ◄ MEJOR" if nombre == mejor else ""
    print(f"  {nombre:<25} {acc*100:>6.2f}%{marca}")

print("="*55)