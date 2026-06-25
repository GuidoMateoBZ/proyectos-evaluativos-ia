import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["axes.grid"] = True

##ACTIVIDAD 11 Cargar el dataset Iris y describir variables. 

iris = load_iris(as_frame=True)
X = iris.data.copy()                 # 4 variables numericas
y_true = iris.target.copy()          # etiquetas reales (solo para comparar al final)
especies = list(iris.target_names)

print("Cantidad de registros :", X.shape[0])
print("Cantidad de variables :", X.shape[1])
print("Variables             :", list(X.columns))
print("Clases reales         :", especies)
X.head()
# Estadisticos descriptivos de las 4 variables
X.describe().round(3)

# Distribucion de clases reales 
pd.Series(y_true).map(dict(enumerate(especies))).value_counts()

##ACTIVIDAD 12   Estandarizar los datos 

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

resumen = pd.DataFrame({
    "media_original": X.mean().values,
    "desvio_original": X.std().values,
    "media_estandarizada": X_std.mean(axis=0).round(3),
    "desvio_estandarizado": X_std.std(axis=0).round(3),
}, index=X.columns)
resumen

##ACTIVIDAD 13 Aplicar PCA a dos componentes para visualiza
## se reduce de 4 a 2 dimensiones con PCA para poder graficar en un plano. 

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_std)

var = pca.explained_variance_ratio_
print(f"Varianza explicada PC1 : {var[0]*100:.2f}%")
print(f"Varianza explicada PC2 : {var[1]*100:.2f}%")
print(f"Varianza acumulada     : {var.sum()*100:.2f}%")

# pesos de cada componente, cuanto aporta cada variable original
pesos = pd.DataFrame(pca.components_, columns=X.columns, index=["PC1", "PC2"])
pesos.round(3)

# Proyeccion en el plano PCA SIN DIFERENCIAR POR CLASE
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c="steelblue", edgecolor="k", alpha=0.7)
plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
plt.title("Iris proyectado en 2 componentes principales")
plt.tight_layout()
plt.show()

#  version diferenciada por especie real, SOLO como referencia visual.
# Las etiquetas no participan del entrenamiento, se usan formalmente recien al final.
plt.figure()
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="viridis",
                 edgecolor="k", alpha=0.8)
plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
plt.title("Iris en PCA coloreado por especie real (referencia)")
plt.legend(sc.legend_elements()[0], especies, title="especie")
plt.tight_layout()
plt.show()