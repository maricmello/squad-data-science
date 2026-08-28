"""
Roda todos os notebooks do projeto em ordem, do 01 ao 06.

Cada notebook depende do(s) anterior(es) já terem rodado, porque lê os
arquivos que eles salvam em data/processed/. Este script executa cada um do
zero (equivalente a "Restart Kernel and Run All") e salva os resultados de
volta no próprio notebook.

Uso:
    python run_all.py
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"

NOTEBOOKS = [
    "01_EDA.ipynb",
    "02_Preprocessamento.ipynb",
    "03_Selecao_de_Features.ipynb",
    "04_Clusterizacao.ipynb",
    "05_Analise_dos_Clusters.ipynb",
    "06_Ranking_de_Atratividade.ipynb",
]

for nome in NOTEBOOKS:
    caminho = NOTEBOOKS_DIR / nome
    print(f"Rodando {nome}...")

    nb = nbformat.read(caminho, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )

    try:
        client.execute()
    except Exception as erro:
        nbformat.write(nb, caminho)
        print(f"Erro em {nome}: {erro}")
        sys.exit(1)

    nbformat.write(nb, caminho)
    print(f"{nome} ok")

print("Todos os notebooks rodaram com sucesso.")
