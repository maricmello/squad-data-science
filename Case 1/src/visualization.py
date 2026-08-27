"""
Funções de gráfico usadas pelos notebooks.

Centralizar aqui evita ficar repetindo configuração de matplotlib em
cada notebook e deixa o código dos notebooks mais focado na leitura dos
resultados do que na "plumbing" de plotar.
"""

import matplotlib.pyplot as plt

CORES_CLUSTER = {0: "#2563eb", 1: "#dc2626", 2: "#f59e0b", 3: "#16a34a", 4: "#7c3aed"}


def configurar_estilo():
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["font.size"] = 11


def plot_cotovelo_silhouette(metricas_df):
    """
    metricas_df: DataFrame com colunas 'k', 'inertia', 'silhouette'
    (o retorno de clustering.avaliar_faixa_de_k).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(metricas_df["k"], metricas_df["inertia"], marker="o", color="#2563eb")
    axes[0].set_title("Método do cotovelo")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(metricas_df["k"], metricas_df["silhouette"], marker="o", color="#dc2626")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silhouette")

    plt.tight_layout()
    return fig


def plot_umap_clusters(coords, labels, nomes_cluster):
    """
    Gráfico de dispersão dos países numa projeção UMAP, colorido por cluster.

    UMAP não tem um "% de variância explicada" como o PCA (ele não é uma
    projeção linear), por isso não recebe esse parâmetro, o gráfico serve
    só para ver a separação dos grupos com outro método, não para medir
    quanto da informação original foi preservada.

    coords: array (n, 2) do embedding UMAP.
    labels: cluster de cada país (o mesmo cluster do K-Means, calculado
    nas variáveis originais, o UMAP aqui é só visualização).
    nomes_cluster: dict {id_cluster: nome legível}.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=nomes_cluster.get(c, f"Cluster {c}"),
            color=CORES_CLUSTER.get(c, "#888888"),
            s=45, alpha=0.8, edgecolor="white", linewidth=0.5,
        )
    ax.set_title("Países agrupados por nível de desenvolvimento\n(UMAP)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ranking_barras(nomes, valores, titulo, xlabel):
    """Gráfico de barras horizontais para um top-N (ex.: ranking de países)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(nomes[::-1], valores[::-1], color="#dc2626")
    for bar, valor in zip(bars, valores[::-1]):
        ax.text(bar.get_width() + valores.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{valor:.0f}", va="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.set_xlim(0, valores.max() * 1.15)
    plt.tight_layout()
    return fig
