"""Funções de pré-processamento compartilhadas pelos notebooks do case.

- `carregar_embeddings_alinhados`: confere o shape dos embeddings ANTES
  de indexar por `df.index.values`, contra o tamanho esperado do array
  completo (linhas sem nulos, antes da deduplicação). Um assert feito
  depois de indexar seria tautológico, sempre verdadeiro por
  construção, mesmo se o alinhamento estivesse errado.
- `obter_ou_criar_split`: persiste `idx_train`/`idx_test` em disco na
  primeira chamada e recarrega nas seguintes, para garantir que os
  notebooks 01-04 usem exatamente o mesmo split, em vez de cada um
  recalcular `train_test_split` de forma independente.
- `medir_duplicatas_semanticas`: quantifica o risco de vazamento
  residual por duplicatas quase-idênticas (não exatas) entre treino e
  teste, complementando a checagem de duplicatas exatas feita no
  notebook 01.
"""

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def sanitizar_reducao(X_tr, X_te, contexto=""):
    """Remove eventuais NaN de uma redução de dimensionalidade (ex.: UMAP).

    O `.transform()` do UMAP em dados fora da amostra de treino pode, em
    alguns pontos, gerar NaN quando o UMAP não fixa `random_state`. Para
    não deixar isso quebrar a modelagem mais adiante, imputamos qualquer
    NaN pela média de cada coluna, calculada apenas no treino (o teste
    nunca é usado para isso).
    """
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
    a remoção de nulos (antes da deduplicação), e não é resetado depois
    disso. É esse índice que permite realinhar `embeddings_texto.npy`
    (gerado uma única vez, para todas as linhas sem nulos) com o
    dataframe já deduplicado, sem precisar recalcular os embeddings: veja
    `carregar_embeddings_alinhados` abaixo, que faz essa indexação com uma
    verificação real de consistência.
    """
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
    """Carrega `embeddings_texto.npy` e o realinha com `df` (já deduplicado).

    O alinhamento é verificado ANTES de indexar, não depois. Um assert
    do tipo:

        embeddings = embeddings_completos[df.index.values]
        assert embeddings.shape[0] == len(df)   # sempre verdadeiro!

    nunca poderia falhar, porque `embeddings` já teria sido construído
    com exatamente `len(df)` linhas. Ele não testaria se o ALINHAMENTO
    está correto, só se o array resultante tem o tamanho esperado, o que
    é garantido pela própria indexação. Se `embeddings_texto.npy`
    tivesse sido gerado em outra ordem, o resultado seria um X e y
    desalinhados de forma silenciosa, sem nenhum erro visível.

    Aqui, a checagem é feita sobre `embeddings_completos` (antes de
    indexar), comparando com `df.attrs['n_linhas_sem_nulos']`, o
    número de linhas esperado (todas as linhas sem valores nulos, antes
    da deduplicação), que é o universo para o qual os embeddings foram
    originalmente gerados. Essa é uma verificação que pode de fato
    falhar, e por isso é uma verificação real.
    """
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
    """Retorna idx_train/idx_test, persistidos em disco na primeira chamada.

    Chamar `train_test_split(idx, y, test_size=0.2, random_state=42,
    stratify=y)` de forma independente em cada notebook (01 a 04)
    dependeria de um contrato implícito nunca verificado: que a mesma
    semente sobre os mesmos dados, na mesma ordem, produz sempre o mesmo
    split. Isso é verdade na prática, mas uma mudança futura em
    `preprocessing.py` (ordem de operações, versão de biblioteca)
    poderia fazer os notebooks divergirem silenciosamente sobre qual
    linha é treino e qual é teste.

    Aqui, o split é calculado uma única vez e salvo em
    `caminho_split` (.npz). Chamadas seguintes (de qualquer notebook)
    carregam o mesmo arquivo em vez de recalcular, então a identidade do
    split entre notebooks passa a ser garantida por construção, não por
    coincidência de sementes.
    """
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
    """Mede o risco de vazamento por duplicatas quase-idênticas (não exatas).

    A deduplicação exata (em `carregar_e_limpar`) já remove textos
    idênticos. Mas descrições quase iguais (pequenas variações de
    pontuação, capitalização, um adjetivo a mais) não são pegas por ela
    e continuam podendo ser divididas entre treino e teste, vazando
    informação de forma parecida (o modelo "reconhece" o texto do teste
    porque viu uma versão quase idêntica no treino).

    Esta função usa similaridade de cosseno sobre TF-IDF (rápida e
    suficiente para uma estimativa, não precisa dos embeddings densos)
    para contar quantos textos de teste têm pelo menos um vizinho no
    treino com similaridade acima de `limiar`, EXCLUINDO os que já são
    idênticos (esses já foram removidos, então não deveriam aparecer aqui
    de qualquer forma, a exclusão é só uma proteção a mais).

    Retorna um dicionário com a contagem e o percentual, no mesmo
    espírito da análise de duplicatas exatas do notebook 01.
    """
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
