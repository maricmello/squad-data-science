"""
Estilo e funções de plot reutilizadas nos notebooks — Case Final.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


def set_style():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 100


def plot_target_distribution(y, ax_hist, ax_box, label="gasto_mensal"):
    sns.histplot(y, kde=True, ax=ax_hist, color="#4C72B0")
    ax_hist.set_title(f"Distribuição de {label}")
    ax_hist.set_xlabel(f"{label} (R$)")

    sns.boxplot(x=y, ax=ax_box, color="#4C72B0")
    ax_box.set_title(f"Boxplot de {label}")
    ax_box.set_xlabel(f"{label} (R$)")


def plot_pred_vs_actual(ax, y_true, y_pred, title, lims):
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#4C72B0")
    ax.plot(lims, lims, "--", color="crimson", linewidth=1.5, label="Predição perfeita")
    ax.set_title(title)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def plot_residual_diagnostics(fitted, resid):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(fitted, resid, alpha=0.35, s=18, color="#4C72B0")
    axes[0].axhline(0, color="crimson", linestyle="--")
    axes[0].set_xlabel("Previsto (R$)")
    axes[0].set_ylabel("Resíduo (R$)")
    axes[0].set_title("Resíduo × previsto")

    sns.histplot(resid, kde=True, ax=axes[1], color="#4C72B0")
    axes[1].axvline(0, color="crimson", linestyle="--")
    axes[1].set_xlabel("Resíduo (R$)")
    axes[1].set_title("Distribuição dos resíduos")

    sm.qqplot(resid, line="s", ax=axes[2])
    axes[2].set_title("Q-Q plot dos resíduos")

    plt.tight_layout()
    return fig
