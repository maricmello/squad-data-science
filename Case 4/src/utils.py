import os
import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3)
TARGET = "Close"
LAGS = (1, 2, 3, 4, 5, 10, 20)
WINDOWS = (5, 10, 20)

PALETTE = {
    "naive": "#8A8A8A",
    "seasonal_naive": "#C9CACB",
    "ses": "#D9782D",
    "holt": "#B23A48",
    "holt_winters": "#8E5FB5",
    "arima": "#5B8C5A",
    "sarima": "#2E9AA8",
    "ml_ridge": "#1D4E89",
    "ml_random_forest": "#118AB2",
    "ml_gradient_boosting": "#C9A227",
    "real": "#111111",
    "treino": "#2E5FA3",
}

RAW_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def load_raw(raw_dir="../data/raw"):
    train_raw = pd.read_csv(f"{raw_dir}/dataset_treinamento.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    test_raw = pd.read_csv(f"{raw_dir}/dataset_teste.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    return train_raw, test_raw


def clean_and_save(train_raw, test_raw, processed_dir="../data/processed"):
    os.makedirs(processed_dir, exist_ok=True)

    train = train_raw.copy()
    train[RAW_COLS] = train[RAW_COLS].interpolate(limit_direction="both")
    test = test_raw.copy()

    train.to_csv(f"{processed_dir}/train.csv", index=False)
    test.to_csv(f"{processed_dir}/test.csv", index=False)
    return train, test


def load_processed(processed_dir="../data/processed"):
    path_train = f"{processed_dir}/train.csv"
    path_test = f"{processed_dir}/test.csv"
    if not (os.path.exists(path_train) and os.path.exists(path_test)):
        raise FileNotFoundError(
            f"Dados processados não encontrados em {processed_dir}. "
            "Rode o notebook 01_analise_exploratoria.ipynb primeiro."
        )
    train = pd.read_csv(path_train, parse_dates=["Date"])
    test = pd.read_csv(path_test, parse_dates=["Date"])
    return train, test


def make_full_series(train, test):
    tr = train.copy()
    tr["split"] = "treino"
    te = test.copy()
    te["split"] = "teste"
    full = pd.concat([tr, te], ignore_index=True).sort_values("Date").reset_index(drop=True)
    return full


def build_features(df, lags=LAGS, windows=WINDOWS):
    d = df.copy().reset_index(drop=True)
    close_shifted = d["Close"].shift(1)

    for lag in lags:
        d[f"lag_{lag}"] = d["Close"].shift(lag)
    for w in windows:
        d[f"roll_mean_{w}"] = close_shifted.rolling(w).mean()
        d[f"roll_std_{w}"] = close_shifted.rolling(w).std()

    d["return_1"] = close_shifted.pct_change(1)
    d["return_5"] = close_shifted.pct_change(5)
    d["dow"] = d["Date"].dt.dayofweek
    d["day"] = d["Date"].dt.day
    d["month"] = d["Date"].dt.month
    d["volume_lag1"] = d["Volume"].shift(1)
    return d


def feature_columns(lags=LAGS, windows=WINDOWS):
    cols = [f"lag_{lag}" for lag in lags]
    for w in windows:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    cols += ["return_1", "return_5", "dow", "day", "month", "volume_lag1"]
    return cols


def build_direct_targets(feat_df, horizons=HORIZONS):
    d = feat_df.copy()
    for h in horizons:
        d[f"target_h{h}"] = d["Close"].shift(-h)
    return d


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def summarize_by_horizon(results_df, group_cols=("model", "horizon")):
    rows = []
    for keys, g in results_df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = regression_metrics(g["y_true"], g["y_pred"])
        rows.append(dict(zip(group_cols, keys), **m, n_obs=len(g)))
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def walkforward_origins(n_train, n_test, max_h=max(HORIZONS)):
    first_origin = n_train - 1
    last_origin = n_train + n_test - max_h - 1
    return list(range(first_origin, last_origin + 1))


def append_predictions(rows, origin_date, full_dates, full_values, origin_idx, h, model_name, y_pred):
    target_idx = origin_idx + h
    rows.append({
        "origin_date": origin_date,
        "target_date": full_dates[target_idx],
        "horizon": h,
        "model": model_name,
        "y_true": full_values[target_idx],
        "y_pred": y_pred,
    })
