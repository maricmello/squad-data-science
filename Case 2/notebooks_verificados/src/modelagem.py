"""Grade comparativa de modelos x representações, usada pelo notebook 03.

- `construir_modelos_candidatos`: os cinco modelos comparados (Logistic
  Regression + quatro modelos baseados em árvores, de naturezas
  diferentes entre si). É uma função, não um dicionário no nível do
  módulo, para que cada chamada devolva instâncias novas — evita
  compartilhar estado mutável entre notebooks/testes que importem este
  módulo mais de uma vez na mesma sessão.
- `carregar_representacao`: carrega X_train/X_test para uma combinação
  (redução, dimensão), tratando TF-IDF (esparso, arquivo próprio) à
  parte de PCA/UMAP (densos, um .npz por dimensão).
- `rodar_grade_comparativa`: laço principal, treina e avalia cada
  combinação (redução x dimensão x modelo) com `avaliar_modelo`,
  pulando HistGradientBoosting para TF-IDF (não aceita entrada esparsa).
"""

import os

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from xgboost import XGBClassifier

from avaliacao import avaliar_modelo

REDUCOES_DISPONIVEIS = {
    'PCA': [10, 20, 30],
    'UMAP': [10, 20, 30],
    'TF-IDF': [5000],
}


def construir_modelos_candidatos():
    """Retorna os cinco modelos comparados no notebook 03.

    Logistic Regression, simples e linear, e quatro modelos baseados em
    árvores, com naturezas diferentes entre si (bagging, extra-random e
    dois tipos de boosting), para não comparar só variações de uma mesma
    família.
    """
    return {
        'Logistic': LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=2,
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=2,
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            random_state=42,
        ),
        'XGBoost': XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=2,
        ),
    }


def carregar_representacao(reducao, dim, pasta_embeddings):
    """Carrega X_train/X_test para uma combinação (redução, dimensão).

    TF-IDF é esparso e fica salvo em dois arquivos próprios
    (`tfidf_train.npz`/`tfidf_test.npz`, formato `scipy.sparse`); PCA e
    UMAP são densos, um `.npz` por dimensão (`{reducao}_{dim}.npz`,
    chaves `X_train`/`X_test`, formato `numpy`).
    """
    if reducao == 'TF-IDF':
        X_tr = sp.load_npz(os.path.join(pasta_embeddings, 'tfidf_train.npz'))
        X_te = sp.load_npz(os.path.join(pasta_embeddings, 'tfidf_test.npz'))
    else:
        dados = np.load(os.path.join(pasta_embeddings, f'{reducao.lower()}_{dim}.npz'))
        X_tr = dados['X_train']
        X_te = dados['X_test']
    return X_tr, X_te


def rodar_grade_comparativa(y_train, y_test,
                             pasta_embeddings='data/processed/embeddings_reducoes',
                             reducoes_disponiveis=None, modelos=None):
    """Treina e avalia cada combinação (redução x dimensão x modelo).

    Para cada representação disponível (PCA/UMAP nas dimensões testadas,
    TF-IDF), carrega X_train/X_test uma única vez e reaproveita entre
    todos os modelos daquela combinação. HistGradientBoosting é pulado
    para TF-IDF, que não aceita entrada esparsa.

    `reducoes_disponiveis` e `modelos` têm como padrão
    `REDUCOES_DISPONIVEIS` e `construir_modelos_candidatos()`
    respectivamente, mas podem ser sobrescritos (ex.: nos testes, com uma
    grade menor).

    Retorna:
    - `resultados`: lista de dicionários de métricas (uma por combinação
      redução x dimensão x modelo), prontos para `pd.DataFrame`.
    - `predicoes_cache`: dict `(reducao, dim, nome_modelo) -> y_pred`.
    - `embeddings_cache`: dict `(reducao, dim) -> (X_train, X_test)`.
    """
    if reducoes_disponiveis is None:
        reducoes_disponiveis = REDUCOES_DISPONIVEIS
    if modelos is None:
        modelos = construir_modelos_candidatos()

    resultados = []
    predicoes_cache = {}
    embeddings_cache = {}

    for reducao, dims in reducoes_disponiveis.items():

        for dim in dims:

            X_tr, X_te = carregar_representacao(reducao, dim, pasta_embeddings)
            embeddings_cache[(reducao, dim)] = (X_tr, X_te)

            for nome, modelo in modelos.items():

                if reducao == 'TF-IDF' and nome == 'HistGradientBoosting':
                    print("[AVISO] HistGradientBoosting pulado para TF-IDF (não suporta entrada esparsa).")
                    continue

                metricas, y_pred = avaliar_modelo(
                    nome,
                    modelo,
                    X_tr,
                    y_train,
                    X_te,
                    y_test,
                )

                metricas.update({
                    'reducao': reducao,
                    'dim': dim,
                })

                resultados.append(metricas)
                predicoes_cache[(reducao, dim, nome)] = y_pred

    return resultados, predicoes_cache, embeddings_cache
