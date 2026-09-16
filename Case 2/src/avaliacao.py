import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def avaliar_modelo(nome_modelo, modelo, X_tr, y_tr, X_te, y_te):
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    metricas = {
        'modelo': nome_modelo,
        'accuracy': accuracy_score(y_te, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_te, y_pred),
        'precision_macro': precision_score(y_te, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_te, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_te, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_te, y_pred, average='weighted', zero_division=0),
    }
    return metricas, y_pred


def comparar_modelos_bootstrap(y_true, y_pred_a, y_pred_b, metrica='f1_macro',
                                n_boot=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)

    def calc(y_t, y_p):
        if metrica == 'f1_macro':
            return f1_score(y_t, y_p, average='macro', zero_division=0)
        elif metrica == 'accuracy':
            return accuracy_score(y_t, y_p)
        elif metrica == 'balanced_accuracy':
            return balanced_accuracy_score(y_t, y_p)
        raise ValueError(f"métrica não suportada: {metrica}")

    diff_observada = calc(y_true, y_pred_a) - calc(y_true, y_pred_b)

    diffs_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx_boot = rng.randint(0, n, size=n)
        m_a = calc(y_true[idx_boot], y_pred_a[idx_boot])
        m_b = calc(y_true[idx_boot], y_pred_b[idx_boot])
        diffs_boot[i] = m_a - m_b

    ic_baixo, ic_alto = np.percentile(diffs_boot, [2.5, 97.5])

    if diff_observada >= 0:
        p_valor = 2 * min((diffs_boot <= 0).mean(), 0.5)
    else:
        p_valor = 2 * min((diffs_boot >= 0).mean(), 0.5)
    p_valor = min(p_valor, 1.0)

    return {
        'metrica': metrica,
        'diferenca_observada': float(diff_observada),
        'ic_95_baixo': float(ic_baixo),
        'ic_95_alto': float(ic_alto),
        'p_valor': float(p_valor),
        'significativo_5pct': bool(p_valor < 0.05),
        'n_boot': n_boot,
    }


def top_features_por_classe(modelo_logistico, vectorizer, classes, top_n=15):

    vocabulario = np.array(vectorizer.get_feature_names_out())
    resultado = {}

    coef = modelo_logistico.coef_
    if coef.shape[0] == 1 and len(classes) == 2:
        # binário: sklearn guarda 1 linha só; a classe 0 é o espelho
        coef = np.vstack([-coef[0], coef[0]])

    for i, classe in enumerate(classes):
        pesos = coef[i]
        idx_top_pos = np.argsort(pesos)[::-1][:top_n]
        idx_top_neg = np.argsort(pesos)[:top_n]
        resultado[classe] = {
            'a_favor': list(zip(vocabulario[idx_top_pos], np.round(pesos[idx_top_pos], 3))),
            'contra': list(zip(vocabulario[idx_top_neg], np.round(pesos[idx_top_neg], 3))),
        }
    return resultado


def cobertura_por_confianca(y_true, y_pred, probas, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99)):

    probas_max = np.asarray(probas).max(axis=1)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_total = len(y_true)

    linhas = []
    for t in thresholds:
        mask_auto = probas_max >= t
        n_auto = int(mask_auto.sum())
        cobertura = n_auto / n_total
        if n_auto > 0:
            acc_auto = accuracy_score(y_true[mask_auto], y_pred[mask_auto])
        else:
            acc_auto = float('nan')
        n_manual = n_total - n_auto
        linhas.append({
            'threshold': t,
            'cobertura_autoclassificacao': round(cobertura * 100, 1),
            'n_autoclassificados': n_auto,
            'accuracy_autoclassificados': round(acc_auto * 100, 1) if n_auto > 0 else None,
            'n_revisao_manual': n_manual,
        })
    return pd.DataFrame(linhas)


def pr_auc_macro(y_true, y_proba, classes):

    y_true = np.asarray(y_true)
    resultado = {}
    for i, classe in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        resultado[classe] = average_precision_score(y_bin, y_proba[:, i])
    resultado['macro'] = float(np.mean(list(resultado.values())))
    return resultado
