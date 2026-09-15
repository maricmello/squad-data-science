# Case: Previsão de Preço de Fechamento (1, 2 e 3 dias)

Previsão do `Close` de uma ação para 1, 2 e 3 dias úteis à frente, comparando o erro
de previsão entre horizontes e entre famílias de modelos (baseline, suavização
exponencial, ARIMA/SARIMA e Machine Learning).

## Estrutura do projeto

```
case_series_temporais/
├── data/
│   ├── raw/            dados originais, nunca editar manualmente
│   │   ├── dataset_treinamento.csv
│   │   └── dataset_teste.csv
│   └── processed/      gerado pelo notebook 01, não versionar à mão
│       ├── train.csv
│       └── test.csv
├── src/
│   └── utils.py        funções compartilhadas (dados, features, métricas, walk-forward)
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_baseline_modelos_classicos.ipynb
│   ├── 03_machine_learning.ipynb
│   └── 04_comparacao_final.ipynb
└── results/            métricas e previsões salvas pelos notebooks 02, 03 e 04
```

## Como reproduzir do zero

Os notebooks têm uma dependência de execução e devem ser executados nesta ordem:

1. `01_analise_exploratoria.ipynb`: lê `data/raw/`, faz a checagem de qualidade
   e é o único lugar que limpa o dado (interpola 3 linhas vazias) e grava o
   resultado em `data/processed/train.csv` e `data/processed/test.csv`.
2. `02_baseline_modelos_classicos.ipynb`: lê `data/processed/`, roda a validação
   walk-forward para Naïve, Seasonal Naïve, SES, Holt, Holt-Winters, ARIMA e SARIMA,
   e salva `results/predictions_classicos.csv` e `results/metrics_classicos.csv`.
3. `03_machine_learning.ipynb`: lê `data/processed/`, treina os modelos diretos
   de ML por horizonte, e salva `results/predictions_ml.csv` e `results/metrics_ml.csv`.
4. `04_comparacao_final.ipynb`: lê os dois arquivos de `results/` gerados acima
   e monta a comparação final.

Para quem só quer olhar os resultados, os 4 notebooks já vêm executados, com
gráficos e tabelas nos outputs, sem necessidade de rodar nada.

## Por que raw e processed

- `data/raw/` é a fonte da verdade: nunca é sobrescrito por código, então a
  origem dos números pode sempre ser conferida.
- `data/processed/` é totalmente derivado de `data/raw/` por uma única função
  (`utils.clean_and_save`). Apagar a pasta inteira e rodar o notebook 01 de novo
  reproduz exatamente o mesmo resultado.
- Os notebooks 02, 03 e 04 nunca leem `data/raw/` diretamente nem duplicam lógica
  de limpeza, o que evita que uma limpeza esquecida em um notebook gere resultados
  diferentes dos outros.

## Ambiente

```
pip install pandas numpy matplotlib statsmodels scikit-learn nbformat nbclient ipykernel
```
