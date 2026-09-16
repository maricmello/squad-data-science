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
