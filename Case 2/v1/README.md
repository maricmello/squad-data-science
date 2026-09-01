# Case — Classificação de Produtos de E-commerce por Texto

## Problema

A partir da descrição textual de um produto, prever automaticamente sua categoria (`Books`,
`Clothing_Accessories`, `Electronics` ou `Household`).

## Estrutura do projeto

```text
Case 2/
├── data/
│   ├── raw/                                     <- dados originais, intocados
│   │   ├── dataset_ecommerce.csv                    (texto + categoria)
│   │   └── embeddings_texto.npy                     (embeddings do texto, já calculados)
│   └── processed/
│       └── embeddings_reducoes/                 <- gerado pelo notebook 02 (representações já ajustadas)
│           ├── pca_10.npz, pca_20.npz, pca_30.npz
│           ├── umap_10.npz, umap_20.npz, umap_30.npz
│           └── tfidf_train.npz, tfidf_test.npz       (esparsos, salvos à parte)
├── notebooks/
│   ├── 01_eda_preprocessamento.ipynb
│   ├── 02_embeddings_reducao_dimensional.ipynb
│   ├── 03_modelagem_avaliacao.ipynb
│   └── 04_modelo_final_conclusao.ipynb
├── src/
│   ├── preprocessing.py                         <- limpeza dos dados (usada por todos os notebooks)
│   └── avaliacao.py                             <- função de avaliação de modelo (métricas multiclasse)
└── README.md
```

> **Sobre `data/raw/`:** o `dataset_ecommerce.csv` (~37MB) e o `embeddings_texto.npy` (~77MB) precisam
> estar fisicamente nessa pasta para os notebooks funcionarem — os arquivos individuais desta transferência
> ficam limitados a 20MB, por isso não consigo colocá-los lá automaticamente. Se ainda não estiverem em
> `data/raw/`, mova (ou copie) os dois arquivos que já existem na raiz de `Case 2/` para dentro dessa pasta;
> depois disso pode apagar as cópias antigas soltas na raiz e a pasta `embeddings_reducoes/` antiga (fora de
> `data/processed/`), que ficou obsoleta.

## O que cada notebook responde

| Notebook | Pergunta que responde |
|---|---|
| `01_eda_preprocessamento` | O que temos nos dados? |
| `02_embeddings_reducao_dimensional` | Como representar o texto (embeddings + PCA/UMAP, ou TF-IDF)? |
| `03_modelagem_avaliacao` | Qual abordagem (representação + modelo) funciona melhor? |
| `04_modelo_final_conclusao` | Qual é o resultado final e o que aprendemos? |

## Como rodar

1. Confirme que `dataset_ecommerce.csv` e `embeddings_texto.npy` estão dentro de `data/raw/` (ver nota
   acima).
2. Instale as dependências: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,
   `sentence-transformers`, `umap-learn`, `xgboost`, `optuna` (todas via `pip install`).
3. Rode os notebooks **em ordem**, de dentro da pasta `notebooks/` (eles usam caminhos relativos
   como `../data/raw/dataset_ecommerce.csv` e `../src`):
   - `01_eda_preprocessamento.ipynb` — não gera nenhum artefato usado pelos demais notebooks
     (a limpeza é centralizada em `src/preprocessing.py` e reaplicada em cada notebook a partir do
     CSV original), mas vale rodar primeiro para entender os dados.
   - `02_embeddings_reducao_dimensional.ipynb` — gera `../data/processed/embeddings_reducoes/*.npz`
     (PCA, UMAP e TF-IDF), usados pelo notebook 03. É a etapa mais demorada (ajustar UMAP em ~22 mil
     textos de treino leva alguns minutos).
   - `03_modelagem_avaliacao.ipynb` — depende dos arquivos gerados pelo 02.
   - `04_modelo_final_conclusao.ipynb` — usa a configuração vencedora decidida no notebook 03
     (representação, dimensão e hiperparâmetros), já registrada no topo do notebook.

## Decisões e boas práticas seguidas

- **Duplicatas exatas de texto e categoria são removidas antes do split** (em `preprocessing.py`,
  por padrão). Sem essa remoção, cerca de 64% do conjunto de teste acabava com uma cópia idêntica
  no treino, vazando informação e inflando as métricas de avaliação (a análise que motivou essa
  decisão está no notebook 01).
- **Nenhuma redução de dimensionalidade (UMAP/PCA) é ajustada usando dados de teste** — o `fit` é
  sempre feito só no treino, e o teste passa apenas por `transform`. A única exceção é a projeção
  UMAP 2D do notebook 02, que é puramente exploratória (não avalia nenhum modelo).
- **Nenhuma conclusão se apoia só em accuracy** — sempre reportamos balanced accuracy e F1 macro/weighted,
  já que as 4 categorias não têm o mesmo tamanho.
- **Baseline obrigatório** — o `DummyClassifier` (classe majoritária) é o piso de comparação para
  qualquer modelo.
- **Otimização de hiperparâmetros (Optuna) usa apenas uma fatia de validação do treino**, nunca o
  conjunto de teste, para não vazar informação para a escolha de hiperparâmetros.
- **Validação cruzada (5-fold)** avalia a estabilidade do resultado — reaproveita a representação já
  ajustada no treino completo (simplificação didática documentada no notebook 03, análoga à simplificação
  do SMOTE didático usada no material teórico do treinamento).
- **TF-IDF entra como representação de comparação, ao lado dos embeddings** — o notebook 03 mostra que
  TF-IDF + Logistic Regression, sem nenhum ajuste de hiperparâmetros, teve o melhor F1-macro entre
  todas as combinações testadas.
- **O Optuna otimiza os dois melhores candidatos, não só um modelo escolhido de antemão** — otimizar
  especificamente o XGBoost tinha sido uma decisão do projeto, não uma exigência do case. Como o
  TF-IDF + Logistic Regression já vinha na frente sem ajuste nenhum, o notebook 03 roda o Optuna
  também nele, e os dois finalistas otimizados são comparados no teste em pé de igualdade. A
  Logistic Regression com TF-IDF venceu (F1-macro 0,949 contra 0,938 do XGBoost), e é essa
  configuração que o notebook 04 usa como modelo final.
