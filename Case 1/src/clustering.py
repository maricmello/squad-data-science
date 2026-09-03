"""
Funções de clusterização usadas pelos notebooks 04 a 06.

A ideia aqui é isolar a parte "matemática" (rodar K-Means, calcular
métricas, comparar métodos) para os notebooks focarem na leitura e
interpretação dos resultados.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

RANDOM_STATE = 42


def avaliar_faixa_de_k(X_escalado, k_range=range(2, 9)):
    """
    Roda K-Means para cada k em k_range e devolve um DataFrame com
    inertia e silhouette score de cada um, a base para o método do
    cotovelo e para a escolha do número de clusters.
    """
    linhas = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_escalado)
        linhas.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_escalado, labels),
        })
    return pd.DataFrame(linhas)


def rodar_kmeans(X_escalado, k, random_state=RANDOM_STATE):
    """Roda o K-Means final com k clusters. Retorna (modelo, labels)."""
    modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = modelo.fit_predict(X_escalado)
    return modelo, labels


def rodar_hierarquico(X_escalado, k, linkage="ward"):
    """Roda clusterização hierárquica aglomerativa. Retorna os labels."""
    modelo = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return modelo.fit_predict(X_escalado)


def comparar_clusterizacoes(labels_a, labels_b):
    """
    Adjusted Rand Index entre duas partições, usado para validar se o
    K-Means e a clusterização hierárquica concordam sobre os grupos.
    1.0 = concordância perfeita, 0.0 = concordância ao acaso.
    """
    return adjusted_rand_score(labels_a, labels_b)


def score_de_atratividade(df, colunas_boas, colunas_ruins):
    z = df.copy()
    termos = []

    for col in colunas_ruins:
        z[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
        termos.append(z[col + "_z"])

    for col in colunas_boas:
        z[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
        termos.append(-z[col + "_z"])

    z["score_atratividade"] = np.mean(termos, axis=0)
    return z
