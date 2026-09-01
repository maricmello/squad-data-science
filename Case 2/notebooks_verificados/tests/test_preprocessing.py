"""Testes unitários para src/preprocessing.py.

Cobrem as funções de limpeza e alinhamento de dados. São testes simples,
com dados sintéticos pequenos, para rodar em menos de um segundo.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import (
    carregar_e_limpar,
    carregar_embeddings_alinhados,
    obter_ou_criar_split,
    medir_duplicatas_semanticas,
    sanitizar_reducao,
)


def _csv_temporario(linhas):
    """Cria um CSV temporário com colunas texto,categoria a partir de uma
    lista de tuplas (texto, categoria)."""
    df = pd.DataFrame(linhas, columns=['texto', 'categoria'])
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(f.name, index=False)
    return f.name


def test_carregar_e_limpar_remove_nulos():
    caminho = _csv_temporario([
        ('produto A', 'Books'),
        (None, 'Household'),
        ('produto B', None),
        ('produto C', 'Electronics'),
    ])
    df = carregar_e_limpar(caminho, remover_duplicatas=False)
    assert df.shape[0] == 2
    assert df['texto'].isna().sum() == 0
    assert df['categoria'].isna().sum() == 0


def test_carregar_e_limpar_remove_duplicatas_exatas():
    caminho = _csv_temporario([
        ('produto A', 'Books'),
        ('produto A', 'Books'),       # duplicata exata -> removida
        ('produto A', 'Household'),   # mesmo texto, categoria diferente -> mantida
        ('produto B', 'Electronics'),
    ])
    df = carregar_e_limpar(caminho, remover_duplicatas=True)
    assert df.shape[0] == 3
    assert (df['texto'] == 'produto A').sum() == 2


def test_carregar_e_limpar_padroniza_categoria():
    caminho = _csv_temporario([
        ('produto A', 'Clothing & Accessories'),
        ('produto B', '  Books  '),
    ])
    df = carregar_e_limpar(caminho, remover_duplicatas=False)
    assert set(df['categoria']) == {'Clothing_Accessories', 'Books'}


def test_carregar_e_limpar_guarda_n_linhas_sem_nulos():
    caminho = _csv_temporario([
        ('produto A', 'Books'),
        ('produto A', 'Books'),
        (None, 'Household'),
        ('produto B', 'Electronics'),
    ])
    df = carregar_e_limpar(caminho, remover_duplicatas=True)
    # 4 linhas no csv, 1 nula -> 3 linhas "sem nulos" antes da dedup
    assert df.attrs['n_linhas_sem_nulos'] == 3
    assert df.shape[0] == 2  # depois da dedup


def test_carregar_embeddings_alinhados_ok():
    caminho = _csv_temporario([
        ('produto A', 'Books'),
        ('produto A', 'Books'),      # duplicata, removida
        ('produto B', 'Electronics'),
        ('produto C', 'Household'),
    ])
    df = carregar_e_limpar(caminho, remover_duplicatas=True)
    # 4 linhas sem nulos originalmente (nenhuma nula aqui)
    embeddings_completos = np.arange(4 * 3).reshape(4, 3).astype(float)
    f_npy = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    np.save(f_npy.name, embeddings_completos)

    embeddings = carregar_embeddings_alinhados(f_npy.name, df)
    assert embeddings.shape[0] == len(df)
    # a linha 0 do df (produto A, mantido) deve corresponder à linha 0
    # do array de embeddings completo
    assert np.array_equal(embeddings[0], embeddings_completos[0])


def test_carregar_embeddings_alinhados_detecta_desalinhamento():
    caminho = _csv_temporario([
        ('produto A', 'Books'),
        ('produto B', 'Electronics'),
    ])
    df = carregar_e_limpar(caminho, remover_duplicatas=False)
    # embeddings com número de linhas ERRADO (deveria ser 2, é 5)
    embeddings_errados = np.zeros((5, 3))
    f_npy = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    np.save(f_npy.name, embeddings_errados)

    with pytest.raises(AssertionError):
        carregar_embeddings_alinhados(f_npy.name, df)


def test_obter_ou_criar_split_e_persistente():
    caminho = _csv_temporario([(f'produto {i}', 'Books' if i % 2 == 0 else 'Household')
                                for i in range(40)])
    df = carregar_e_limpar(caminho, remover_duplicatas=False)
    y = (df['categoria'] == 'Books').astype(int).values

    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_split = os.path.join(tmpdir, 'split.npz')
        idx_tr_1, idx_te_1 = obter_ou_criar_split(df, y, caminho_split, test_size=0.25, random_state=1)
        assert os.path.exists(caminho_split)
        idx_tr_2, idx_te_2 = obter_ou_criar_split(df, y, caminho_split, test_size=0.25, random_state=1)
        # segunda chamada deve carregar o mesmo split salvo, não recalcular
        assert np.array_equal(idx_tr_1, idx_tr_2)
        assert np.array_equal(idx_te_1, idx_te_2)
        # treino e teste não podem se sobrepor
        assert len(set(idx_tr_1) & set(idx_te_1)) == 0


def test_medir_duplicatas_semanticas_detecta_quase_iguais():
    linhas = [(f'produto totalmente diferente numero {i} xyz', 'Books') for i in range(20)]
    # um texto do "teste" é quase idêntico a um do "treino" (1 palavra a mais)
    linhas[0] = ('caneta azul bic ponta fina escrita suave', 'Books')
    linhas[10] = ('caneta azul bic ponta fina escrita suave nova', 'Books')
    caminho = _csv_temporario(linhas)
    df = carregar_e_limpar(caminho, remover_duplicatas=False)

    idx_train = np.array([0])
    idx_test = np.array([10])
    resultado = medir_duplicatas_semanticas(df, idx_train, idx_test, limiar=0.7)
    assert resultado['n_teste'] == 1
    assert resultado['n_com_quase_duplicata'] == 1


def test_sanitizar_reducao_imputa_pela_media_do_treino():
    X_tr = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    X_te = np.array([[np.nan, 1.0]])
    X_tr_limpo, X_te_limpo = sanitizar_reducao(X_tr, X_te, contexto="teste")
    assert not np.isnan(X_tr_limpo).any()
    assert not np.isnan(X_te_limpo).any()
    # a coluna 1 do treino tem média (2+6)/2 = 4.0 (ignorando o NaN)
    assert X_tr_limpo[1, 1] == pytest.approx(4.0)
