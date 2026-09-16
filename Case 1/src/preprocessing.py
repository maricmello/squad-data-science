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

    colunas = colunas or TODAS_AS_FEATURES
    return {
        "nulos_por_coluna": df.isna().sum(),
        "paises_duplicados": int(df["pais"].duplicated().sum()),
        "describe": df[colunas].describe().round(2).T,
    }


def padronizar(df, colunas):

    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(df[colunas])
    return X_escalado, scaler


def dataframe_padronizado(df, colunas):

    X_escalado, scaler = padronizar(df, colunas)
    df_padronizado = pd.DataFrame(X_escalado, columns=colunas)
    df_padronizado.insert(0, "pais", df["pais"].values)
    return df_padronizado, scaler
