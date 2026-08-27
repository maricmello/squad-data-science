"""
Funções de carga e pré-processamento dos dados de países.

Usadas pelos notebooks 01 a 03. Mantidas aqui para não repetir o mesmo
código em cada notebook e para facilitar testes/reuso fora do Jupyter.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Todas as colunas numéricas do dataset original (exclui 'pais').
TODAS_AS_FEATURES = [
    "mortalidade_infantil",
    "exportacoes",
    "saude",
    "importacoes",
    "renda",
    "inflacao",
    "expectativa_vida",
    "fertilidade_total",
    "pib_per_capita",
]


def checar_qualidade(df, colunas=None):
    """
    Roda checagens básicas de qualidade de dados: nulos, duplicados e
    um describe() das colunas numéricas.

    Retorna um dicionário com os resultados, para o notebook decidir o
    que imprimir ou não.
    """
    colunas = colunas or TODAS_AS_FEATURES
    return {
        "nulos_por_coluna": df.isna().sum(),
        "paises_duplicados": int(df["pais"].duplicated().sum()),
        "describe": df[colunas].describe().round(2).T,
    }


def padronizar(df, colunas):
    """
    Padroniza (média 0, desvio padrão 1) as colunas indicadas.

    Retorna (X_escalado, scaler) — o scaler é retornado para poder ser
    reaplicado depois (por exemplo, em dados novos) sem refazer o fit.
    """
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(df[colunas])
    return X_escalado, scaler


def dataframe_padronizado(df, colunas):
    """Versão em DataFrame do X_escalado, com o nome do país de volta."""
    X_escalado, scaler = padronizar(df, colunas)
    df_padronizado = pd.DataFrame(X_escalado, columns=colunas)
    df_padronizado.insert(0, "pais", df["pais"].values)
    return df_padronizado, scaler
