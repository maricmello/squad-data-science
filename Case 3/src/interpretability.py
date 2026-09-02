"""
Interpretação de modelo — importância nativa, permutation importance e SHAP.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42


def native_importance(final_model, features: list) -> pd.Series:
    step = final_model.named_steps["model"]
    if hasattr(step, "coef_"):
        return pd.Series(step.coef_, index=features).sort_values(key=np.abs, ascending=False)
    return pd.Series(step.feature_importances_, index=features).sort_values(ascending=False)


def permutation_importance_df(final_model, X_test, y_test, features: list, n_repeats: int = 30) -> pd.DataFrame:
    perm = permutation_importance(
        final_model, X_test, y_test, n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": features,
        "importancia_media": perm.importances_mean,
        "importancia_std": perm.importances_std,
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)


def shap_explain(final_model, X_train, X_test, features: list):
    step = final_model.named_steps["model"]
    scaler = final_model.named_steps.get("scaler")

    if hasattr(step, "coef_"):
        X_train_bg = scaler.transform(X_train) if scaler is not None else X_train.values
        X_test_bg = scaler.transform(X_test) if scaler is not None else X_test.values
        explainer = shap.LinearExplainer(step, X_train_bg, feature_names=features)
        shap_values = explainer(X_test_bg)
    else:
        explainer = shap.TreeExplainer(step)
        shap_values = explainer(X_test.values)
        shap_values.feature_names = features

    return explainer, shap_values
