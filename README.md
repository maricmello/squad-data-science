# Squad Data Science

Cases práticos do treinamento de Ciência de Dados, cobrindo o caminho completo de um projeto: análise exploratória, clusterização, classificação de texto, regressão e séries temporais. Cada case parte de um problema de negócio e entrega código reproduzível, resultados e limitações.

## Cases

| Case | Tema | Técnica | Resultado principal |
|---|---|---|---|
| [Case 0](Case%200) | Perfil e comportamento de clientes | EDA, testes estatísticos, PCA, KMeans | `segmento_latente` organiza a base, não o plano contratado (ARI de 0,871 recuperando os segmentos só com dados numéricos) |
| [Case 1](Case%201) | Onde ajudar países emergentes? | Clusterização (K-Means e hierárquico) | 3 grupos entre 167 países (31 desenvolvidos, 89 em desenvolvimento, 47 vulneráveis) e ranking dos 10 mais prioritários |
| [Case 2](Case%202) | Classificação de produtos de e-commerce por texto | TF-IDF, embeddings, PCA/UMAP, Logistic Regression, Optuna | F1-macro de 0,948 e ROC-AUC de 0,991 com TF-IDF + Logistic Regression |
| [Case 3](Case%203) | Previsão de gasto mensal por cliente | Regressão (Ridge, Gradient Boosting), SHAP | R² de 0,53 e MAE de R$ 288,3, contra R$ 418,9 do baseline |
| [Case 4](Case%204) | Previsão do preço de fechamento de uma ação (1, 2 e 3 dias) | Séries temporais: baselines, ARIMA/SARIMA, ML e Prophet | Holt teve o menor erro médio (MAPE de 4,88%); os modelos de ML não superaram os baselines |
| [portifólio](portif%C3%B3lio) | Portfólio da squad | HTML | Resumo navegável dos cases, com achados, decisões e limitações |

Cada case tem seu próprio `README.md` com o detalhamento do problema, dos dados, dos resultados e das limitações.

## Estrutura do repositório

```
squad-data-science/
├── Case 0/        # EDA: perfil e comportamento de clientes
├── Case 1/        # Clusterização: países emergentes
├── Case 2/        # Classificação de texto: produtos de e-commerce
├── Case 3/        # Regressão: previsão de gasto mensal
├── Case 4/        # Séries temporais: preço de fechamento de ação
└── portifólio/    # portfolio_data_science_final.html
```

Os cases mais completos (1, 2 e 3) seguem a mesma organização:

```
Case N/
├── data/
│   ├── raw/           # dados originais
│   └── processed/     # artefatos gerados pelo pipeline
├── notebooks/         # análise em etapas numeradas
├── src/               # código reutilizável importado pelos notebooks
├── requirements.txt
└── README.md
```

## Como rodar

Cada case é independente e tem suas próprias dependências.

```bash
git clone https://github.com/maricmello/squad-data-science.git
cd squad-data-science/"Case 1"          # troque pelo case desejado

python -m venv .venv
source .venv/bin/activate               # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab notebooks/
```

Pontos de atenção:

- **Ordem dos notebooks:** nos Cases 1, 2, 3 e 4 os notebooks são numerados e cada um lê o que o anterior salvou, então devem ser executados em sequência.
- **Pipeline completo sem Jupyter:** os Cases 1 e 3 têm um `run_all.py` em `src/`. No Case 3: `python src/run_all.py`.
- **Case 0:** tem um único notebook (`notebook/EDA.ipynb`). As bibliotecas usadas estão listadas no README do case.
- **Case 4:** não tem `requirements.txt`. As dependências estão no README do case:
  `pip install pandas numpy matplotlib statsmodels scikit-learn prophet nbformat nbclient ipykernel`.

## Principais bibliotecas

`pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `plotly`, `umap-learn`, `optuna`, `shap`, `prophet` e `jupyter`. A lista exata de cada case está no respectivo `requirements.txt` ou README.

## Boas práticas adotadas

- Baseline antes de qualquer modelo, para saber o que o modelo precisa superar.
- Split de treino e teste fixo e reutilizado entre os notebooks, com transformações ajustadas apenas no treino, para evitar vazamento de dados.
- Validação cruzada com `Pipeline` nos cases de classificação e regressão, e validação walk-forward nas séries temporais.
- Mais de uma métrica por problema, em vez de depender só de accuracy ou de R².
- Testes de significância (bootstrap, ANOVA, Kruskal-Wallis) antes de afirmar que um modelo ou grupo é melhor que outro.
- Limitações documentadas em cada case.

## Autora

[maricmello](https://github.com/maricmello)
