"""Funções de pré-processamento compartilhadas pelos notebooks do case.

Centralizar a limpeza aqui garante que 01, 02, 03 e 04 apliquem exatamente
a mesma transformação ao dataset original, sem precisar salvar e manter
sincronizada uma cópia intermediária (o CSV original já tem ~37MB).
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def sanitizar_reducao(X_tr, X_te, contexto=""):
    """Remove eventuais NaN de uma redução de dimensionalidade (ex.: UMAP).

    O `.transform()` do UMAP em dados fora da amostra de treino pode, em
    alguns pontos, gerar NaN quando o UMAP não fixa `random_state` (como é
    o caso aqui, para permitir `n_jobs=2`). Para não deixar isso quebrar a
    modelagem mais adiante, imputamos qualquer NaN pela média de cada
    coluna, calculada apenas no treino (o teste nunca é usado para isso).
    """
    n_nan_tr = np.isnan(X_tr).sum()
    n_nan_te = np.isnan(X_te).sum()

    if n_nan_tr == 0 and n_nan_te == 0:
        return X_tr, X_te

    print(f"[AVISO]{' ' + contexto if contexto else ''} "
          f"encontrados {n_nan_tr} NaN no treino e {n_nan_te} NaN no teste "
          f"— imputando pela média do treino.")

    imputer = SimpleImputer(strategy='mean')
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)
    return X_tr, X_te


def carregar_e_limpar(caminho_csv, remover_duplicatas=True):
    """Carrega o dataset de e-commerce e aplica a limpeza padrão do projeto.

    Passos:
    - mantém apenas as colunas usadas (texto, categoria);
    - remove linhas com valores nulos;
    - padroniza o nome da categoria 'Clothing & Accessories' (o '&' pode
      causar problemas em alguns pipelines/arquivos) para 'Clothing_Accessories';
    - remove espaços em branco extras nos nomes das categorias;
    - remove duplicatas exatas de (texto, categoria), mantendo a primeira
      ocorrência (ver notebook 01 para a análise que motivou essa decisão:
      sem isso, boa parte do conjunto de teste acaba com uma cópia idêntica
      no treino, o que vaza informação e infla as métricas de avaliação).

    O índice do dataframe retornado preserva a posição da linha logo após
    a remoção de nulos (antes da deduplicação), e não é resetado. É esse
    índice que permite realinhar `embeddings_texto.npy` -- gerado uma única
    vez, para todas as linhas sem nulos -- com o dataframe já deduplicado,
    sem precisar recalcular os embeddings: basta indexar
    `embeddings[df.index.values]` depois de carregar o `.npy`.
    """
    df = pd.read_csv(caminho_csv)
    df = df[['texto', 'categoria']]
    df = df.dropna().reset_index(drop=True)
    df['categoria'] = df['categoria'].replace('Clothing & Accessories', 'Clothing_Accessories')
    df['categoria'] = df['categoria'].astype(str).str.strip()

    if remover_duplicatas:
        df = df.drop_duplicates(subset=['texto', 'categoria'], keep='first')

    return df
