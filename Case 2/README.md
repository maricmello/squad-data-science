# Case: Classificação de Produtos de E-commerce por Texto

## Problema

A partir da descrição textual de um produto, prever automaticamente sua categoria (`Books`,
`Clothing_Accessories`, `Electronics` ou `Household`).

## Estrutura do projeto

```text
notebooks_verificados/
├── data/
│   ├── raw/                                     <- dados originais
│   │   ├── dataset_ecommerce.csv                   
│   │   └── embeddings_texto.npy                     
│   └── processed/
│       ├── split.npz                            <- índices de treino/teste
│       └── embeddings_reducoes/                 
│           ├── pca_10.npz, pca_20.npz, pca_30.npz
│           ├── umap_10.npz, umap_20.npz, umap_30.npz, umap_10_seed42.npz
│           └── tfidf_train.npz, tfidf_test.npz       
├── figs/                                        
├── scripts/                                     <- código-fonte dos notebooks em formato jupytext (.py)
├── src/
│   ├── preprocessing.py                         <- limpeza e alinhamento de dados 
│   ├── avaliacao.py                             <- métricas, teste de significância, interpretabilidade
│   └── modelagem.py                             <- modelos candidatos e grade comparativa 
├── tests/                                       <- testes unitários para src/
├── 01_eda_preprocessamento.ipynb
├── 02_embeddings_reducao_dimensional.ipynb
├── 03_modelagem_avaliacao.ipynb
├── 04_modelo_final_conclusao.ipynb
└── README.md
```


## O que cada notebook responde

| Notebook | Pergunta que responde |
|---|---|
| `01_eda_preprocessamento` | O que temos nos dados? |
| `02_embeddings_reducao_dimensional` | Como representar o texto (embeddings + PCA/UMAP, ou TF-IDF)? |
| `03_modelagem_avaliacao` | Qual abordagem (representação + modelo) funciona melhor? |
| `04_modelo_final_conclusao` | Qual é o resultado final e o que aprendemos? |

## Como rodar


1. Instale as dependências listadas em `requirements.txt`: `pip install -r requirements.txt`.
2. Rode os notebooks em ordem.


## Decisões e boas práticas seguidas

- Duplicatas exatas: texto e categoria são removidos antes do split.
- Quase-duplicatas: textos muito semelhantes são identificados e monitorados.
- Embeddings: o alinhamento entre os embeddings e o dataframe é verificado antes do uso.
- Split fixo: treino e teste são salvos e reutilizados em todos os notebooks.
- PCA/UMAP: ajustados somente com os dados de treino; o teste apenas é transformado.
- Validação cruzada: usa Pipeline, evitando vazamento entre treino e validação.
- Métricas: não depende apenas de accuracy; também considera F1, balanced accuracy, ROC-AUC e PR-AUC.
- Baseline: o DummyClassifier é usado como referência mínima.
- Optuna: os hiperparâmetros são otimizados apenas com dados de treino/validação.
- Comparação dos finalistas: os dois melhores modelos do ranking geral são otimizados com Optuna e a diferença entre eles é testada estatisticamente.
- Escolha do modelo: TF-IDF + Logistic Regression teve o melhor resultado (F1-macro 0,948).
- Interpretabilidade: os coeficientes da Logistic Regression permitem identificar as palavras mais importantes para cada categoria.


## Conclusão

**O modelo funciona?** Sim, com boa margem sobre o baseline. O F1-macro da Logistic Regression
otimizada (0,948) e o ROC-AUC (0,991) ficaram bem acima do baseline de classe majoritária (F1-macro
de apenas 0,138). A taxa de erro geral no teste ficou em 5,2% (288 de 5.561
produtos).

**Esse foi realmente o melhor modelo possível?** Sim.  Foi encontrado uma vantagem estatisticamente significativa da Logistic Regression (+0,0075 de F1-macro, IC 95% [+0,0017, +0,0133], p ≈ 0,0120) sobre o segundo colocado.

**Qual foi a melhor representação?** O TF-IDF, uma representação clássica de contagem de palavras
ponderada, teve o melhor resultado geral, superando qualquer combinação de embedding de frase
testada (PCA e UMAP sobre os embeddings pré-calculados). 

**Quais categorias são mais difíceis?** `Household` continua concentrando boa parte dos erros, tanto
sendo confundida com as outras quanto recebendo previsões que deveriam ser de outras categorias. Faz
sentido: é a categoria mais ampla e heterogênea do catálogo, o que aumenta a sobreposição de
vocabulário com as demais.

## Limitações do trabalho
- Quase-duplicatas: 21,9% do teste (1.218 de 5.561 linhas) possui textos muito semelhantes no treino, o que pode deixar a métrica um pouco otimista.
- UMAP: a diferença de F1-macro entre execuções foi de 0,0033, mostrando pequena variação por causa da aleatoriedade.
- Otimização: o Optuna foi usado apenas nos dois modelos finalistas; os demais ficaram com parâmetros padrão.
- Deduplicação: apenas duplicatas exatas foram removidas. As quase-duplicatas ainda permanecem.
