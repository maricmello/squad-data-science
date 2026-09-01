"""
Definição dos modelos, baseline e grades de hiperparâmetros — Case Final.

Centralizar os modelos aqui garante que a comparação (notebook 03), a
cross-validation (notebook 04) e o tuning (notebook 05) usem exatamente as
mesmas definições de pipeline.
"""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42


def get_baseline() -> DummyRegressor:
    """Baseline: sempre prever a média do treino."""
    return DummyRegressor(strategy="mean")


def get_models() -> dict:
    """Modelos candidatos, todos em Pipeline (padronização aplicada só onde
    faz diferença — Linear/Ridge — para evitar vazamento entre folds)."""
    return {
        "Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "DecisionTree": Pipeline([
            ("model", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("model", XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        ]),
    }


PARAM_DISTRIBUTIONS = {
    "Ridge": {"model__alpha": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]},
    "Linear": {},
    "DecisionTree": {
        "model__max_depth": [3, 4, 5, 6, 8, 10, 12],
        "model__min_samples_leaf": [1, 2, 4, 8, 16, 32],
    },
    "RandomForest": {
        "model__n_estimators": [200, 300, 400, 600, 800],
        "model__max_depth": [None, 4, 6, 8, 10, 14],
        "model__min_samples_leaf": [1, 2, 4, 8, 16],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.8, 1.0],
    },
    "XGBoost": {
        "model__n_estimators": [200, 300, 400, 600, 800],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    },
}
