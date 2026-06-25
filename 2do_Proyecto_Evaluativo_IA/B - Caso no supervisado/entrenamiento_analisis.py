import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RANDOM_STATE = 42

from tratamiento_datos import X_std, y_true, especies, X_pca, var, pca
##ACTIVIDAD 14 14. Entrenar KMeans con k entre 2 y 7.

##Se entrena un modelo por cada valor de k y se guardan la inercia y el
##silhouette para analizarlos después.
ks = range(2, 8)
inercias = []
silhouettes = []
modelos = {}

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_std)
    modelos[k] = km
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_std, labels))

tabla = pd.DataFrame({
    "k": list(ks),
    "inercia": np.round(inercias, 2),
    "silhouette": np.round(silhouettes, 4),
})
print(tabla)

##ACTIVIDAD 15 Analizar inercia y metodo del codo.

plt.plot(list(ks), inercias, "o-", color="darkorange")
plt.xlabel("k (numero de clusters)")
plt.ylabel("Inercia")
plt.title("Metodo del codo")
plt.axvline(3, color="gray", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

## ACTIVIDAD 16 Analizar silhouette para cada k

plt.figure()
plt.plot(list(ks), silhouettes, "s-", color="seagreen")
plt.xlabel("k (numero de clusters)")
plt.ylabel("Coeficiente de silhouette")
plt.title("Silhouette promedio segun k")
plt.tight_layout()
plt.show()

mejor_sil = list(ks)[int(np.argmax(silhouettes))]
print("k con mayor silhouette:", mejor_sil)

##Actividad 17 Visualizar clusters en el plano PCA.
##se grafican los clusters  de kmeans sobre las dos componentes principales, para
##k = 3 (estructura real) y k = 2 (mejor silhouette), con los centroides proyectados.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, k in zip(axes, [3, 2]):
    km = modelos[k]
    labels = km.labels_
    cent_pca = pca.transform(km.cluster_centers_)
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis",
                    edgecolor="k", alpha=0.8)
    ax.scatter(cent_pca[:, 0], cent_pca[:, 1], c="red", marker="X",
               s=220, edgecolor="white", label="centroides")
    ax.set_title(f"KMeans con k = {k}")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.legend()

plt.tight_layout()
plt.show()


