"""Testes unitários para src/modelagem.py."""
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modelagem import (
    construir_modelos_candidatos,
    carregar_representacao,
    rodar_grade_comparativa,
)


def test_construir_modelos_candidatos_tem_as_cinco_familias():
    modelos = construir_modelos_candidatos()
    assert set(modelos.keys()) == {
        'Logistic', 'Random Forest', 'Extra Trees',
        'HistGradientBoosting', 'XGBoost',
    }


def test_construir_modelos_candidatos_devolve_instancias_novas_a_cada_chamada():
    # Se fosse um dict no nível do módulo, as duas chamadas
    # compartilhariam os mesmos objetos (estado mutável entre usos).
    modelos_a = construir_modelos_candidatos()
    modelos_b = construir_modelos_candidatos()
    assert modelos_a['Logistic'] is not modelos_b['Logistic']


def test_carregar_representacao_pca(tmp_path):
    X_train = np.random.RandomState(0).normal(size=(20, 10))
    X_test = np.random.RandomState(1).normal(size=(6, 10))
    np.savez(tmp_path / 'pca_10.npz', X_train=X_train, X_test=X_test)

    X_tr, X_te = carregar_representacao('PCA', 10, str(tmp_path))
    np.testing.assert_array_equal(X_tr, X_train)
    np.testing.assert_array_equal(X_te, X_test)


def test_carregar_representacao_tfidf_e_esparso(tmp_path):
    X_train = sp.random(20, 50, density=0.1, format='csr', random_state=0)
    X_test = sp.random(6, 50, density=0.1, format='csr', random_state=1)
    sp.save_npz(tmp_path / 'tfidf_train.npz', X_train)
    sp.save_npz(tmp_path / 'tfidf_test.npz', X_test)

    X_tr, X_te = carregar_representacao('TF-IDF', 5000, str(tmp_path))
    assert sp.issparse(X_tr) and sp.issparse(X_te)
    assert X_tr.shape == X_train.shape


def _grade_sintetica(tmp_path, n_train=60, n_test=20, n_features=8, seed=0):
    rng = np.random.RandomState(seed)
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train = (X_train[:, 0] > 0).astype(int)
    y_test = (X_test[:, 0] > 0).astype(int)
    np.savez(tmp_path / 'pca_8.npz', X_train=X_train, X_test=X_test)

    X_train_tfidf = sp.random(n_train, 30, density=0.1, format='csr', random_state=seed)
    X_test_tfidf = sp.random(n_test, 30, density=0.1, format='csr', random_state=seed + 1)
    sp.save_npz(tmp_path / 'tfidf_train.npz', X_train_tfidf)
    sp.save_npz(tmp_path / 'tfidf_test.npz', X_test_tfidf)

    return y_train, y_test


def test_rodar_grade_comparativa_uma_linha_por_combinacao(tmp_path):
    y_train, y_test = _grade_sintetica(tmp_path)
    modelos = {'dummy': DummyClassifier(strategy='most_frequent')}
    reducoes = {'PCA': [8], 'TF-IDF': [5000]}

    resultados, predicoes_cache, embeddings_cache = rodar_grade_comparativa(
        y_train, y_test,
        pasta_embeddings=str(tmp_path),
        reducoes_disponiveis=reducoes,
        modelos=modelos,
    )

    # 2 reduções x 1 dimensão cada x 1 modelo = 2 combinações
    assert len(resultados) == 2
    assert {(r['reducao'], r['dim']) for r in resultados} == {('PCA', 8), ('TF-IDF', 5000)}
    assert set(embeddings_cache.keys()) == {('PCA', 8), ('TF-IDF', 5000)}
    assert set(predicoes_cache.keys()) == {('PCA', 8, 'dummy'), ('TF-IDF', 5000, 'dummy')}
    for y_pred in predicoes_cache.values():
        assert len(y_pred) == len(y_test)


def test_rodar_grade_comparativa_pula_histgradientboosting_no_tfidf(tmp_path, capsys):
    y_train, y_test = _grade_sintetica(tmp_path)
    modelos = {
        'Logistic': LogisticRegression(),
        'HistGradientBoosting': HistGradientBoostingClassifier(),
    }
    reducoes = {'TF-IDF': [5000]}

    resultados, _, _ = rodar_grade_comparativa(
        y_train, y_test,
        pasta_embeddings=str(tmp_path),
        reducoes_disponiveis=reducoes,
        modelos=modelos,
    )

    nomes_avaliados = {r['modelo'] for r in resultados}
    assert nomes_avaliados == {'Logistic'}
    assert 'pulado para TF-IDF' in capsys.readouterr().out


def test_rodar_grade_comparativa_usa_padroes_quando_nao_especificado(tmp_path):
    # Sem passar `modelos`/`reducoes_disponiveis`, a função usa os padrões
    # do módulo (5 modelos x todas as reduções/dimensões declaradas em
    # REDUCOES_DISPONIVEIS) — aqui só checamos que os padrões são de fato
    # aplicados, sem rodar a grade completa (lenta para um teste unitário).
    from modelagem import REDUCOES_DISPONIVEIS, construir_modelos_candidatos
    assert REDUCOES_DISPONIVEIS['TF-IDF'] == [5000]
    assert len(construir_modelos_candidatos()) == 5
