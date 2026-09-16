import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def sanitizar_reducao(X_tr, X_te, contexto=""):

    n_nan_tr = np.isnan(X_tr).sum()
    n_nan_te = np.isnan(X_te).sum()

    if n_nan_tr == 0 and n_nan_te == 0:
        return X_tr, X_te

    print(f"[AVISO]{' ' + contexto if contexto else ''} "
          f"encontrados {n_nan_tr} NaN no treino e {n_nan_te} NaN no teste, "
          f"imputando pela média do treino.")

    imputer = SimpleImputer(strategy='mean')
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)
    return X_tr, X_te


def carregar_e_limpar(caminho_csv, remover_duplicatas=True):

    df = pd.read_csv(caminho_csv)
    df = df[['texto', 'categoria']]
    df = df.dropna().reset_index(drop=True)
    n_linhas_sem_nulos = len(df)
    df['categoria'] = df['categoria'].replace('Clothing & Accessories', 'Clothing_Accessories')
    df['categoria'] = df['categoria'].astype(str).str.strip()

    if remover_duplicatas:
        df = df.drop_duplicates(subset=['texto', 'categoria'], keep='first')

    df.attrs['n_linhas_sem_nulos'] = n_linhas_sem_nulos
    return df


def carregar_embeddings_alinhados(caminho_npy, df):

    embeddings_completos = np.load(caminho_npy)

    n_esperado = df.attrs.get('n_linhas_sem_nulos')
    if n_esperado is None:
        raise ValueError(
            "df não tem o atributo 'n_linhas_sem_nulos'. Gere `df` com "
            "`carregar_e_limpar()` deste módulo, para que o número de "
            "linhas esperado nos embeddings seja conhecido antes de indexar."
        )

    if embeddings_completos.shape[0] != n_esperado:
        raise AssertionError(
            f"Desalinhamento entre '{caminho_npy}' e o dataset limpo: "
            f"o arquivo de embeddings tem {embeddings_completos.shape[0]} linhas, "
            f"mas eram esperadas {n_esperado} (linhas do CSV original sem valores "
            f"nulos, antes da deduplicação). Indexar por df.index.values nessas "
            f"condições produziria X e y desalinhados sem erro visível, por "
            f"isso a execução para aqui em vez de seguir adiante."
        )

    embeddings = embeddings_completos[df.index.values]

    if embeddings.shape[0] != len(df):
        # Segunda linha de defesa: nunca deveria disparar se o df.index
        # vem de carregar_e_limpar(), mas protege contra uso incorreto
        # da função (ex.: um df filtrado/reordenado manualmente depois).
        raise AssertionError(
            f"embeddings indexados ({embeddings.shape[0]} linhas) não batem "
            f"com df ({len(df)} linhas). Confira se df.index não foi "
            f"alterado (reset, filtro, sort) depois de carregar_e_limpar()."
        )

    return embeddings


def obter_ou_criar_split(df, y, caminho_split, test_size=0.2, random_state=42):

    if os.path.exists(caminho_split):
        dados = np.load(caminho_split)
        idx_train, idx_test = dados['idx_train'], dados['idx_test']
        # Confere que o split salvo é compatível com o df atual (mesmo
        # tamanho); se o dataset mudou, o split salvo está obsoleto.
        if idx_train.max() >= len(df) or idx_test.max() >= len(df):
            raise ValueError(
                f"O split salvo em '{caminho_split}' não é compatível com o "
                f"df atual (tamanho {len(df)}). Apague o arquivo para "
                f"recriar o split, ou verifique se os dados mudaram."
            )
        return idx_train, idx_test

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    os.makedirs(os.path.dirname(caminho_split), exist_ok=True)
    np.savez(caminho_split, idx_train=idx_train, idx_test=idx_test)
    return idx_train, idx_test


def medir_duplicatas_semanticas(df, idx_train, idx_test, limiar=0.9, max_features=20000):

    textos_train = df['texto'].values[idx_train]
    textos_test = df['texto'].values[idx_test]

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=1)
    X_train = vectorizer.fit_transform(textos_train)
    X_test = vectorizer.transform(textos_test)

    n_com_quase_duplicata = 0
    similaridades_max = np.zeros(X_test.shape[0])

    # Calcular em blocos para não estourar memória com uma matriz densa
    # gigante (n_test x n_train pode ser grande).
    bloco = 500
    for inicio in range(0, X_test.shape[0], bloco):
        fim = min(inicio + bloco, X_test.shape[0])
        sims = cosine_similarity(X_test[inicio:fim], X_train)
        max_por_linha = sims.max(axis=1)
        similaridades_max[inicio:fim] = max_por_linha

    n_com_quase_duplicata = int((similaridades_max >= limiar).sum())
    n_teste = X_test.shape[0]

    return {
        'limiar': limiar,
        'n_teste': n_teste,
        'n_com_quase_duplicata': n_com_quase_duplicata,
        'percentual': round(n_com_quase_duplicata / n_teste * 100, 2),
        'similaridade_media_maxima': round(float(similaridades_max.mean()), 4),
        'similaridades_max': similaridades_max,
    }
