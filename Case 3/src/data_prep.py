from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "gasto_mensal"
IDADE_MINIMA_VALIDA = 18
IDADE_MINIMA_INICIO_RELACIONAMENTO = 18

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "case_regression.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def idade_invalida_mask(df: pd.DataFrame) -> pd.Series:
    return df["idade"] < IDADE_MINIMA_VALIDA


def tempo_cliente_invalido_mask(df: pd.DataFrame) -> pd.Series:
    return df["tempo_cliente"] > (df["idade"] - IDADE_MINIMA_INICIO_RELACIONAMENTO)


def clean_data(
    df: pd.DataFrame,
    filtrar_idade_invalida: bool = True,
    filtrar_tempo_cliente_invalido: bool = True,
) -> pd.DataFrame:
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")]
    out = df.drop(columns=cols_to_drop)
    if filtrar_idade_invalida:
        out = out.loc[~idade_invalida_mask(out)].reset_index(drop=True)
    if filtrar_tempo_cliente_invalido:
        out = out.loc[~tempo_cliente_invalido_mask(out)].reset_index(drop=True)
    return out


def get_feature_target(df: pd.DataFrame, target: str = TARGET):
    features = [c for c in df.columns if c != target]
    return df[features], df[target]


def make_split(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_processed_split(processed_dir: Path = PROCESSED_DIR, target: str = TARGET):
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    X_train, y_train = get_feature_target(train, target)
    X_test, y_test = get_feature_target(test, target)
    return X_train, X_test, y_train, y_test
