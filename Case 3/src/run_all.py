"""
Executa o pipeline completo do case (etapas 1-12) fora dos notebooks, do
carregamento dos dados brutos até o modelo final tunado e o relatório de
métricas. Útil para reprodutibilidade (ex.: rodar em CI) e para regenerar os
artefatos em data/processed/ e models/ sem precisar abrir o Jupyter.

Uso:
    python src/run_all.py

Decisões de design principais:
1. `idade` implausível (< 18 anos) e `tempo_cliente` inconsistente com a
   `idade` (cliente teria começado a comprar antes dos 18 anos) são removidos
   em data_prep.clean_data por padrão, nessa ordem. As duas são decisões de
   validade de dado, documentadas com teste de sensibilidade no notebook 01:
   o efeito médio na métrica é neutro/levemente negativo para a idade, e uma
   leve melhora de RMSE para o tempo_cliente, mas nos dois casos as linhas
   removidas não são "casos difíceis" para o modelo (erro absoluto médio
   menor que o da base), então a melhora não vem de tirar ruído do modelo.
2. A escolha do modelo final ENTRE os dois candidatos já tunados (melhor
   linear e melhor árvore/boosting) usa o RMSE médio de cross-validation
   (calculado no treino, pelo próprio RandomizedSearchCV), não o RMSE do
   conjunto de teste. O teste é usado só para reportar a métrica final do
   modelo já escolhido.
3. Um teste de significância (bootstrap pareado) compara os dois candidatos
   tunados no teste, para deixar explícito se a diferença observada é ou
   não maior que o ruído de amostragem.
4. MAPE e métricas segmentadas por faixa de renda são reportadas no resumo
   final, além de MAE/RMSE/R².
5. Um "model card" com as versões das bibliotecas usadas é salvo junto ao
   modelo, para reprodutibilidade futura do artefato serializado.
"""
import json
import platform
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import shap
from sklearn.model_selection import KFold, RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_prep, evaluation, interpretability, modeling

RANDOM_STATE = 42


def main():
    processed_dir = ROOT / "data" / "processed"
    models_dir = ROOT / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. carregar + limpar (remove colunas de índice, idade implausível e
    # tempo_cliente inconsistente com a idade, nessa ordem)
    df_raw = data_prep.load_raw_data()
    n_idade_invalida = int(data_prep.idade_invalida_mask(df_raw).sum())

    df_sem_idade_invalida = data_prep.clean_data(df_raw, filtrar_tempo_cliente_invalido=False)
    n_tempo_cliente_invalido = int(data_prep.tempo_cliente_invalido_mask(df_sem_idade_invalida).sum())

    df = data_prep.clean_data(df_raw)
    df.to_csv(processed_dir / "dataset_limpo.csv", index=False)
    X, y = data_prep.get_feature_target(df)
    features = list(X.columns)

    # 3. split
    X_train, X_test, y_train, y_test = data_prep.make_split(X, y)
    pd.concat([X_train, y_train], axis=1).to_csv(processed_dir / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(processed_dir / "test.csv", index=False)

    # 4. baseline
    baseline = modeling.get_baseline()
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluation.regression_metrics(y_test, baseline.predict(X_test))

    # 5-6. comparar modelos
    models = modeling.get_models()
    results_df, preds_test = evaluation.evaluate_models(models, X_train, y_train, X_test, y_test)
    results_df.to_csv(processed_dir / "model_comparison.csv", index=False)

    # 9. cross-validation (estabilidade) — usa só o treino, para o teste não
    # influenciar a escolha de qual modelo será otimizado no passo 10
    cv_df, _ = evaluation.cross_validate_models(models, X_train, y_train)
    cv_df.to_csv(processed_dir / "cv_results.csv", index=False)

    # 10. tuning do melhor modelo linear e do melhor modelo de árvore/boosting
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    linear_names = [n for n in cv_df["modelo"] if n in ("Linear", "Ridge")]
    tree_names = [n for n in cv_df["modelo"] if n not in ("Linear", "Ridge")]
    best_linear = cv_df[cv_df["modelo"].isin(linear_names)].sort_values("RMSE_cv_media").iloc[0]["modelo"]
    best_tree = cv_df[cv_df["modelo"].isin(tree_names)].sort_values("RMSE_cv_media").iloc[0]["modelo"]

    def tune(name):
        """Retorna (pipeline_tunado, params, rmse_medio_de_CV). O RMSE de CV
        (não o de teste) é o critério usado depois para decidir entre os
        dois candidatos tunados."""
        dist = modeling.PARAM_DISTRIBUTIONS[name]
        if not dist:
            pipe = models[name]
            pipe.fit(X_train, y_train)
            cv_rmse = float(cv_df.set_index("modelo").loc[name, "RMSE_cv_media"])
            return pipe, {}, cv_rmse
        search = RandomizedSearchCV(
            models[name], dist, n_iter=30, cv=kf, scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, float(-search.best_score_)

    linear_model, linear_params, linear_cv_rmse = tune(best_linear)
    tree_model, tree_params, tree_cv_rmse = tune(best_tree)

    candidates = {
        f"{best_linear} (tuned)": (linear_model, linear_cv_rmse),
        f"{best_tree} (tuned)": (tree_model, tree_cv_rmse),
    }

    # Seleção do modelo final pelo RMSE médio de CV (treino) — o teste é
    # tocado só depois, para reportar a métrica do modelo já escolhido.
    final_name = min(candidates, key=lambda n: candidates[n][1])
    final_model = candidates[final_name][0]
    joblib.dump(final_model, models_dir / "modelo_final.joblib")

    with open(processed_dir / "modelo_final_nome.txt", "w") as f:
        f.write(final_name)

    # Teste de significância (bootstrap pareado) entre os dois candidatos
    # tunados, avaliado no teste — só para quantificar se a diferença é
    # maior que o ruído amostral, não para decidir o modelo final.
    name_a, name_b = list(candidates.keys())
    pred_a = candidates[name_a][0].predict(X_test)
    pred_b = candidates[name_b][0].predict(X_test)
    boot = evaluation.bootstrap_compare_rmse(y_test, pred_a, pred_b)

    # 11. interpretabilidade
    perm_df = interpretability.permutation_importance_df(final_model, X_test, y_test, features)
    perm_df.to_csv(processed_dir / "permutation_importance.csv", index=False)

    # 12. resumo para negócio
    final_pred = final_model.predict(X_test)
    final_metrics = evaluation.regression_metrics(y_test, final_pred)
    resid = y_test.values - final_pred
    bp = evaluation.breusch_pagan_test(resid, X_test)

    seg_renda = evaluation.segment_metrics(y_test, final_pred, X_test["renda"], label="faixa_renda")
    seg_renda.to_csv(processed_dir / "metrics_por_faixa_renda.csv", index=False)

    summary = {
        "modelo_final": final_name,
        "criterio_escolha": "menor RMSE médio de cross-validation no treino (não o RMSE de teste)",
        "candidatos_tunados_rmse_cv": {name_a: candidates[name_a][1], name_b: candidates[name_b][1]},
        "metricas_teste": final_metrics,
        "metricas_baseline": baseline_metrics,
        "reducao_rmse_vs_baseline_pct": round(100 * (1 - final_metrics["RMSE"] / baseline_metrics["RMSE"]), 1),
        "mae_pct_do_gasto_medio": round(100 * final_metrics["MAE"] / y_test.mean(), 1),
        "breusch_pagan_pvalue": bp["lm_pvalue"],
        "top_5_features_permutation": perm_df.head(5)["feature"].tolist(),
        "linhas_idade_invalida_removidas": n_idade_invalida,
        "linhas_tempo_cliente_invalido_removidas": n_tempo_cliente_invalido,
        "teste_significancia_candidatos_tunados": {
            "comparacao": f"{name_a} vs {name_b}",
            "rmse_teste_a": boot["rmse_a"],
            "rmse_teste_b": boot["rmse_b"],
            "diferenca_observada": boot["diff_observado"],
            "ic95_diferenca": list(boot["ic95_diff"]),
            "p_valor_aprox": boot["p_valor_aprox"],
            "diferenca_significativa_95": bool(boot["diferenca_significativa_95"]),
        },
        "metricas_por_faixa_renda": seg_renda.to_dict(orient="records"),
    }
    with open(processed_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Model card: versões usadas para gerar o artefato final, para
    # reprodutibilidade futura do .joblib.
    model_card = {
        "modelo_final": final_name,
        "hiperparametros": linear_params if name_a == final_name else tree_params,
        "gerado_em": pd.Timestamp.now().isoformat(),
        "python": platform.python_version(),
        "bibliotecas": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "random_state": RANDOM_STATE,
        "n_linhas_treino": int(len(X_train)),
        "n_linhas_teste": int(len(X_test)),
        "features": features,
    }
    with open(models_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nModelo final salvo em: {models_dir / 'modelo_final.joblib'}")
    print(f"Artefatos processados salvos em: {processed_dir}")


if __name__ == "__main__":
    main()
