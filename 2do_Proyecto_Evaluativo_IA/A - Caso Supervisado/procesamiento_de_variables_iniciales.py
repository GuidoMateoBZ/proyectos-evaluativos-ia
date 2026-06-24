
#4.1. 
#1 Cargar el dataset y describir cantidad de registros, variables y clases.
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Cargar dataset
cancer = load_breast_cancer()

# DataFrame
df = load_breast_cancer(as_frame=True).frame

print("Cantidad de registros:", df.shape[0])
print("Cantidad de variables:", df.shape[1] - 1)  # sin contar la clase

print(f"Cantidad de clases: {len(cancer.target_names)}")
print("Clases:", ", ".join(cancer.target_names))

conteo = df['target'].value_counts().sort_index()
print("\n")

for i, nombre_clase in enumerate(cancer.target_names):
    print(f"{nombre_clase}: {conteo[i]}")

print("\nPrimeras filas:")
print(df.head())


#4.1
#2 Analizar distribucion de clases e identificar si existe desbalance. 

conteo = df["target"].value_counts().sort_index()
porcentajes = df["target"].value_counts(normalize=True).sort_index() * 100

print("Distribución de clases:")
for i, nombre_clase in enumerate(cancer.target_names):
    print(f"{nombre_clase}: {conteo[i]} registros ({porcentajes[i]:.2f}%)")

ratio = conteo.max() / conteo.min()

print(f"\nRelación mayoritaria/minoritaria: {ratio:.2f} : 1")


# Para identificar el desblance 
if ratio > 3:
    print("Existe un desbalance importante.")
elif ratio > 1.5:
    print("Existe un desbalance moderado.")
else:
    print("Las clases están relativamente balanceadas.")


# Grafico
import matplotlib.pyplot as plt

conteo.index = cancer.target_names

conteo.plot(kind='bar')
plt.title('Distribución de clases')
plt.ylabel('Cantidad de registros')
plt.xlabel('Clase')
plt.show()


#4.1
#3 Calcular estadisticos descriptivos de las variables.

variables = [
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean compactness",
    "mean smoothness",
    "worst compactness",
    "worst fractal dimension"
]

print("\n" + "="*70)
print("ESTADÍSTICOS DESCRIPTIVOS")
print("="*70)

for v in variables:
    print(f"\n{v.upper()}")
    print("-"*50)
    print(f"{'Media:':<20}{df[v].mean():>10.2f}")
    print(f"{'Mediana:':<20}{df[v].median():>10.2f}")
    print(f"{'Desvío estándar:':<20}{df[v].std():>10.2f}")

print("\n" + "="*70)
print("ASIMETRÍA")
print("="*70)

for v in variables:
    print(f"{v:<30} {df[v].skew():>8.2f}")

print("\n" + "="*70)
print("COEFICIENTE DE VARIACIÓN")
print("="*70)

for v in variables:
    cv = df[v].std() / df[v].mean()
    print(f"{v:<30} {cv:>8.2f}")

print("\n" + "="*70)
print("OUTLIERS DETECTADOS (MÉTODO IQR)")
print("="*70)

for v in variables:
    q1 = df[v].quantile(0.25)
    q3 = df[v].quantile(0.75)
    iqr = q3 - q1

    inferiores = q1 - 1.5 * iqr
    superiores = q3 + 1.5 * iqr

    outliers = df[(df[v] < inferiores) | (df[v] > superiores)]

    print(f"{v:<30} {len(outliers):>5}")

print("\n" + "="*70)
print("HISTOGRAMA: MEAN AREA")
print("="*70)

df["mean area"].hist(bins=20)
plt.title("Distribución de mean area")
plt.xlabel("mean area")
plt.ylabel("Frecuencia")
plt.show()