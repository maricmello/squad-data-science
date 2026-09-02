"""Testes unitários para src/avaliacao.py."""
import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from avaliacao import (
    avaliar_modelo,
    comparar_modelos_bootstrap,
    top_features_por_classe,
    cobertura_por_confianca,
    pr_auc_macro,
)


def _dados_sinteticos(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_avaliar_modelo_retorna_metricas_esperadas():
    X, y = _dados_sinteticos()
    X_tr, X_te, y_tr, y_te = X[:150], X[150:], y[:150], y[150:]
    metricas, y_pred = avaliar_modelo('logistic', LogisticRegression(), X_tr, y_tr, X_te, y_te)
    for chave in ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro',
                  'f1_macro', 'f1_weighted']:
        assert chave in metricas
        assert 0.0 <= metricas[chave] <= 1.0
    assert len(y_pred) == len(y_te)


def test_bootstrap_modelo_identico_nao_e_significativo():
    X, y = _dados_sinteticos()
    y_pred = y.copy()  
    resultado = comparar_modelos_bootstrap(y, y_pred, y_pred, n_boot=200, random_state=0)
    assert resultado['diferenca_observada'] == pytest.approx(0.0)
    assert resultado['significativo_5pct'] is False


def test_bootstrap_detecta_modelo_claramente_pior():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=500)
    y_pred_bom = y_true.copy()
    
    idx_erro = rng.choice(len(y_true), size=int(0.45 * len(y_true)), replace=False)
    y_pred_ruim = y_true.copy()
    y_pred_ruim[idx_erro] = 1 - y_pred_ruim[idx_erro]

    resultado = comparar_modelos_bootstrap(y_true, y_pred_bom, y_pred_ruim, n_boot=500, random_state=0)
    assert resultado['diferenca_observada'] > 0
    assert resultado['significativo_5pct'] is True
    assert resultado['p_valor'] < 0.05


def test_top_features_por_classe_formato():
    from sklearn.feature_extraction.text import TfidfVectorizer
    textos = ['gato preto', 'cachorro late', 'gato mia alto', 'cachorro corre rapido']
    classes_texto = ['gato', 'cachorro', 'gato', 'cachorro']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(textos)
    y = np.array([0 if c == 'gato' else 1 for c in classes_texto])
    modelo = LogisticRegression()
    modelo.fit(X, y)

    resultado = top_features_por_classe(modelo, vectorizer, classes=['gato', 'cachorro'], top_n=3)
    assert set(resultado.keys()) == {'gato', 'cachorro'}
    assert 'a_favor' in resultado['gato']
    assert 'contra' in resultado['gato']
    assert len(resultado['gato']['a_favor']) <= 3


def test_cobertura_por_confianca_thresholds_mais_altos_cobrem_menos():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    probas = np.array([
        [0.99, 0.01],
        [0.60, 0.40],
        [0.10, 0.90],
        [0.55, 0.45],
        [0.85, 0.15],
    ])
    tabela = cobertura_por_confianca(y_true, y_pred, probas, thresholds=(0.5, 0.9))
    cobertura_50 = tabela.loc[tabela['threshold'] == 0.5, 'cobertura_autoclassificacao'].iloc[0]
    cobertura_90 = tabela.loc[tabela['threshold'] == 0.9, 'cobertura_autoclassificacao'].iloc[0]
    assert cobertura_90 <= cobertura_50


def test_pr_auc_macro_modelo_perfeito_da_1():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    resultado = pr_auc_macro(y_true, y_proba, classes=['a', 'b'])
    assert resultado['macro'] == pytest.approx(1.0)
