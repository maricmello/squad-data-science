"""
Métricas, cross-validation e diagnóstico de resíduos — Case Final.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate

RANDOM_STATE = 42


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R² e MAPE.

    MAE e RMSE sozinhos não dizem se o erro pesa mais, proporcionalmente,
    para clientes de gasto baixo ou alto — por isso o MAPE é reportado
    junto, já que a target varia de ~R$600 a ~R$4.400 (ver notebook 01)."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def evaluate_models(models: dict, X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Treina cada modelo no treino e avalia no teste. Retorna a tabela de
    métricas e um dict {nome: y_pred} para uso em gráficos posteriores."""
    rows, preds = [], {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        preds[name] = pred
        rows.append({"modelo": name, **regression_metrics(y_test, pred)})
    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results_df, preds


def cross_validate_models(models: dict, X, y, n_splits: int = 5, random_state: int = RANDOM_STATE):
    """K-Fold CV para todos os modelos. Retorna:
    - cv_df: média/desvio-padrão de MAE/RMSE/R² por modelo (estabilidade)
    - cv_raw: dict {nome: resultado bruto do cross_validate}, com os valores
      por fold, útil para gráficos de estabilidade (boxplot por fold)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

    cv_raw, rows = {}, []
    for name, pipe in models.items():
        cvres = cross_validate(pipe, X, y, cv=kf, scoring=scoring, n_jobs=-1)
        cv_raw[name] = cvres
        rows.append({
            "modelo": name,
            "MAE_cv_media": -cvres["test_neg_mean_absolute_error"].mean(),
            "MAE_cv_std": cvres["test_neg_mean_absolute_error"].std(),
            "RMSE_cv_media": -cvres["test_neg_root_mean_squared_error"].mean(),
            "RMSE_cv_std": cvres["test_neg_root_mean_squared_error"].std(),
            "R2_cv_media": cvres["test_r2"].mean(),
            "R2_cv_std": cvres["test_r2"].std(),
        })
    cv_df = pd.DataFrame(rows).sort_values("RMSE_cv_media").reset_index(drop=True)
    return cv_df, cv_raw


def breusch_pagan_test(residuals: np.ndarray, X_test: pd.DataFrame) -> dict:
    """Teste formal de heterocedasticidade usando as features do teste como
    exógenas. p-valor baixo = evidência de variância do erro não constante."""
    exog = sm.add_constant(X_test.values)
    lm_stat, lm_pvalue, f_stat, f_pvalue = sms.het_breuschpagan(residuals, exog)
    return {"lm_stat": lm_stat, "lm_pvalue": lm_pvalue, "f_stat": f_stat, "f_pvalue": f_pvalue}


def segment_metrics(y_true, y_pred, segment: pd.Series, n_bins: int = 4, label: str = "segmento") -> pd.DataFrame:
    """MAE/RMSE/MAPE por faixa (quartil) de uma variável de segmentação
    (ex.: renda, idade). Usado na análise de robustez para checar se o erro
    é uniforme entre grupos de negócio, e não só na média agregada."""
    y_true = pd.Series(np.asarray(y_true)).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
    seg = pd.Series(np.asarray(segment)).reset_index(drop=True)

    faixas = pd.qcut(seg, q=n_bins, duplicates="drop")
    rows = []
    for faixa, idx in y_true.groupby(faixas, observed=True).groups.items():
        yt, yp = y_true.loc[idx], y_pred.loc[idx]
        rows.append({
            label: str(faixa),
            "n": len(idx),
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": mean_squared_error(yt, yp) ** 0.5,
            "MAPE": mean_absolute_percentage_error(yt, yp),
        })
    return pd.DataFrame(rows)


def bootstrap_compare_rmse(y_true, pred_a, pred_b, n_boot: int = 5000, random_state: int = RANDOM_STATE) -> dict:
    """Compara o RMSE de dois modelos no MESMO conjunto de teste via
    bootstrap pareado (reamostra os pares (y, pred_a, pred_b) juntos, não
    cada vetor separadamente, preservando a estrutura pareada).

    Retorna a diferença observada (RMSE_a - RMSE_b), um IC 95% via
    percentil bootstrap para essa diferença, e um p-valor aproximado de
    dois lados. Usado para não declarar um modelo "vencedor" quando a
    diferença de RMSE no teste é da ordem do ruído de amostragem."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)

    rmse_a = mean_squared_error(y_true, pred_a) ** 0.5
    rmse_b = mean_squared_error(y_true, pred_b) ** 0.5
    diff_obs = rmse_a - rmse_b

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        r_a = mean_squared_error(y_true[idx], pred_a[idx]) ** 0.5
        r_b = mean_squared_error(y_true[idx], pred_b[idx]) ** 0.5
        diffs[i] = r_a - r_b

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = min(2 * min((diffs >= 0).mean(), (diffs < 0).mean()), 1.0)

    return {
        "rmse_a": rmse_a,
        "rmse_b": rmse_b,
        "diff_observado": diff_obs,
        "ic95_diff": (ci_low, ci_high),
        "p_valor_aprox": p_value,
        "diferenca_significativa_95": not (ci_low <= 0 <= ci_high),
    }
