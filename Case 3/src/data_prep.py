"""
Carregamento e preparação de dados, Case Final: Previsão de Gasto Mensal.

Mantém a lógica de limpeza e split em um único lugar para que todos os
notebooks (e o run_all.py) usem exatamente a mesma regra, evitando
divergência entre etapas do pipeline.

Tratamento de `idade` implausível:
- `idade` tem 39/2200 linhas (1,8%) com valores implausíveis (< 18 anos,
  mínimo de 3,18 anos) para clientes com histórico de compras. Testamos a
  remoção dessas linhas em 10 splits diferentes (seeds 0-9): o efeito médio
  no RMSE de teste é neutro a levemente negativo (dentro do ruído entre
  seeds) e o erro absoluto médio dessas 39 linhas no modelo treinado com
  tudo é, na verdade, menor que o da base (230 vs 285), ou seja, a remoção
  não é motivada por ganho de performance (não há), mas por validade dos
  dados: uma idade abaixo de 18 anos para um cliente com histórico de
  compras é um valor implausível e não deveria ser usado para treinar ou
  avaliar o modelo. Ver `notebooks/01_EDA.ipynb` (seção "Tratamento de
  idade implausível") para o detalhe da análise de sensibilidade.

Tratamento de `tempo_cliente` inconsistente com `idade`:
- Depois de remover a idade implausível, ainda restam 89/2161 linhas (4,1%)
  em que `tempo_cliente` é maior que `idade - 18`, ou seja, o cliente teria
  começado a comprar antes de completar 18 anos (assumindo 18 como idade
  mínima para iniciar o relacionamento, a mesma suposição já usada para o
  filtro de `idade`). Testamos a remoção dessas 89 linhas em 10 splits
  diferentes: o RMSE médio de teste cai de 370,1 para 362,1 (redução
  consistente em 8 dos 10 seeds), mas o erro absoluto médio dessas linhas no
  modelo treinado com tudo é menor que o da base (263 vs 285), isto é, elas
  não são casos difíceis para o modelo. A melhora de RMSE não vem de tirar
  ruído que o modelo não conseguia prever, é um efeito colateral de mudar a
  composição da base de teste. A decisão de remover segue o mesmo critério
  da idade: validade do dado, não ganho de performance. Ver
  `notebooks/01_EDA.ipynb` (seção "Tratamento de tempo_cliente inconsistente
  com idade") para o detalhe da análise de sensibilidade.
"""
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
    """Lê o CSV bruto do case."""
    return pd.read_csv(path)


def idade_invalida_mask(df: pd.DataFrame) -> pd.Series:
    """Máscara das linhas com `idade` abaixo do mínimo plausível para um
    cliente com histórico de compras. Isolado em função própria para que a
    análise de sensibilidade (com/sem essas linhas) possa ser reproduzida
    fora de `clean_data`."""
    return df["idade"] < IDADE_MINIMA_VALIDA


def tempo_cliente_invalido_mask(df: pd.DataFrame) -> pd.Series:
    """Máscara das linhas em que `tempo_cliente` é maior que
    `idade - IDADE_MINIMA_INICIO_RELACIONAMENTO`, ou seja, o cliente teria
    começado a comprar antes da idade mínima assumida. Deve ser calculada
    depois do filtro de `idade` implausível, para não misturar os dois
    problemas (uma idade já implausível, muito baixa, torna quase qualquer
    `tempo_cliente` "inconsistente" por conta própria)."""
    return df["tempo_cliente"] > (df["idade"] - IDADE_MINIMA_INICIO_RELACIONAMENTO)


def clean_data(
    df: pd.DataFrame,
    filtrar_idade_invalida: bool = True,
    filtrar_tempo_cliente_invalido: bool = True,
) -> pd.DataFrame:
    """Remove colunas de índice sem valor preditivo (ex.: 'Unnamed: 0') e,
    por padrão, remove linhas com `idade` implausível (< 18 anos) e linhas em
    que `tempo_cliente` é inconsistente com a `idade` (cliente teria começado
    a comprar antes dos 18 anos).

    Os dois filtros são aplicados nessa ordem: primeiro `idade`, depois
    `tempo_cliente`, já que o segundo depende de `idade` já estar plausível.

    `filtrar_idade_invalida=False` e `filtrar_tempo_cliente_invalido=False`
    servem só para reproduzir as análises de sensibilidade que compararam o
    pipeline com e sem essas linhas. O pipeline de produção (`run_all.py`,
    notebooks 02 a 06) sempre usa o default (True para os dois).
    """
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")]
    out = df.drop(columns=cols_to_drop)
    if filtrar_idade_invalida:
        out = out.loc[~idade_invalida_mask(out)].reset_index(drop=True)
    if filtrar_tempo_cliente_invalido:
        out = out.loc[~tempo_cliente_invalido_mask(out)].reset_index(drop=True)
    return out


def get_feature_target(df: pd.DataFrame, target: str = TARGET):
    """Separa features (X) e target (y)."""
    features = [c for c in df.columns if c != target]
    return df[features], df[target]


def make_split(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Split holdout treino/teste. Aleatório: não há estrutura temporal nem
    repetição da mesma entidade (cliente) entre linhas nesta base."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_processed_split(processed_dir: Path = PROCESSED_DIR, target: str = TARGET):
    """Carrega train.csv/test.csv já processados (gerados pelo notebook 02) e
    devolve X_train, X_test, y_train, y_test."""
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    X_train, y_train = get_feature_target(train, target)
    X_test, y_test = get_feature_target(test, target)
    return X_train, X_test, y_train, y_test
