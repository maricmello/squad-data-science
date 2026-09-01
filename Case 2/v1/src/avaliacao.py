"""Funções de avaliação de modelos compartilhadas pelos notebooks do case."""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def avaliar_modelo(nome_modelo, modelo, X_tr, y_tr, X_te, y_te):
    """Treina um modelo e retorna um dicionário com as principais métricas multiclasse.

    Sempre reporta accuracy junto de balanced accuracy e precision/recall/F1
    macro e weighted — nunca só accuracy, já que o problema é multiclasse
    e as categorias não são perfeitamente balanceadas.
    """
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
