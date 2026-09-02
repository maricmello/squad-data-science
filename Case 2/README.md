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

1. Confirme que `dataset_ecommerce.csv` e `embeddings_texto.npy` estão dentro de `data/raw/`.
2. Instale as dependências listadas em `requirements.txt`: `pip install -r requirements.txt`.
3. Rode os notebooks **em ordem**, de dentro desta pasta (eles usam caminhos relativos como
   `data/raw/dataset_ecommerce.csv` e `./src`):
   - `01_eda_preprocessamento.ipynb`: limpa os dados, mede duplicatas (exatas e semânticas) e gera
     o split de treino/teste, persistido em `data/processed/split.npz` para os demais notebooks
     carregarem.
   - `02_embeddings_reducao_dimensional.ipynb`: gera `data/processed/embeddings_reducoes/*.npz`
     (PCA, UMAP e TF-IDF), usados pelo notebook 03. É a etapa mais demorada (ajustar UMAP em ~22 mil
     textos de treino leva alguns minutos).
   - `03_modelagem_avaliacao.ipynb`: depende dos arquivos gerados pelo 02 e do split gerado pelo 01.
   - `04_modelo_final_conclusao.ipynb`: usa a configuração vencedora decidida no notebook 03
     (representação, dimensão e hiperparâmetros), já registrada no topo do notebook.
4. Para conferir as funções de `src/`, rode `pytest tests/` de dentro desta pasta (22 testes).


## Decisões e boas práticas seguidas

- **Duplicatas exatas de texto e categoria são removidas antes do split** (em `preprocessing.py`,
  por padrão). Sem essa remoção, uma fração grande do conjunto de teste acabaria com uma cópia
  idêntica no treino, vazando informação e inflando as métricas de avaliação (a análise que motivou
  essa decisão está no notebook 01).
- **Duplicatas semânticas (quase-idênticas, não exatas) são medidas**, via similaridade de cosseno
  sobre TF-IDF (notebook 01). Uma fração não desprezível do teste tem uma quase-duplicata no treino,
  o que é um risco residual de vazamento a se ter em conta ao interpretar a métrica de teste.
- **O alinhamento entre `embeddings_texto.npy` e o dataframe é verificado de forma real**, checando o
  shape do array de embeddings completo antes de indexar por posição, contra o número de linhas
  esperado (`carregar_embeddings_alinhados` em `src/preprocessing.py`).
- **O split de treino/teste é persistido em disco** (`obter_ou_criar_split`) e recarregado pelos
  quatro notebooks, garantindo que todos usem exatamente o mesmo split por construção, não só por
  coincidência de semente.
- **Nenhuma redução de dimensionalidade (UMAP/PCA) é ajustada usando dados de teste**: o `fit` é
  sempre feito só no treino, e o teste passa apenas por `transform`. A única exceção é a projeção
  UMAP 2D do notebook 02, que é puramente exploratória (não avalia nenhum modelo).
- **A validação cruzada (5-fold) usa `sklearn.pipeline.Pipeline`**, refazendo o `fit` do TF-IDF/UMAP a
  cada fold, evitando que a representação "veja", no ajuste, textos do próprio fold de validação.
- **Nenhuma conclusão se apoia só em accuracy**: sempre reportamos balanced accuracy, F1 macro/weighted,
  ROC-AUC e PR-AUC, já que as 4 categorias não têm o mesmo tamanho e o ROC-AUC satura perto de 1 neste
  problema.
- **Baseline obrigatório**: o `DummyClassifier` (classe majoritária) é o piso de comparação para
  qualquer modelo.
- **Otimização de hiperparâmetros (Optuna) usa apenas uma fatia de validação do treino**, nunca o
  conjunto de teste, para não vazar informação para a escolha de hiperparâmetros. Os dois melhores
  candidatos (não só um modelo escolhido a priori) são otimizados em pé de igualdade.
- **A diferença entre os dois finalistas é testada estatisticamente** (bootstrap pareado sobre o
  conjunto de teste, notebook 03), em vez de comparar só os números de F1-macro.
- **TF-IDF entra como representação de comparação, ao lado dos embeddings**: o notebook 03 mostra que
  TF-IDF + Logistic Regression teve o melhor F1-macro entre todas as combinações testadas, e a Logistic
  Regression com TF-IDF venceu o XGBoost otimizado (F1-macro 0,948 contra 0,938), diferença confirmada
  como estatisticamente significativa.
- **O modelo vencedor é interpretado**, não só avaliado. Os coeficientes da Logistic Regression sobre
  o vocabulário do TF-IDF mostram quais palavras mais pesam a favor de cada categoria (notebooks 03 e
  04).
- **A métrica de ML é conectada a uma decisão operacional**: o notebook 04 traduz o threshold de
  confiança das previsões em cobertura de autoclassificação vs. accuracy, uma régua concreta para
  decidir com a área de negócio onde vale a pena automatizar.
- **Um plano mínimo de monitoramento pós-deploy** é proposto no notebook 04, com indicadores para
  detectar deriva (drift) de vocabulário.

## Conclusão

**O modelo funciona?** Sim, com boa margem sobre o baseline. O F1-macro da Logistic Regression
otimizada (0,948) e o ROC-AUC (0,991) ficaram bem acima do baseline de classe majoritária (F1-macro
de apenas 0,138). A taxa de erro geral no teste ficou em 5,2% (288 de 5.561
produtos).

**Esse foi realmente o melhor modelo possível?** Dentro do que foi testado, sim, e agora com uma
confirmação estatística: o teste de significância do notebook 03 (bootstrap pareado, 3.000
reamostragens) mostrou que a vantagem da Logistic Regression sobre o XGBoost otimizado (+0,0075 de
F1-macro, IC 95% [+0,0017, +0,0133], p ≈ 0,0120) é estatisticamente significativa, não apenas um
número maior por sorte de amostragem.

**Qual foi a melhor representação?** O TF-IDF, uma representação clássica de contagem de palavras
ponderada, teve o melhor resultado geral, superando qualquer combinação de embedding de frase
testada (PCA e UMAP sobre os embeddings pré-calculados). A seção 5 do notebook 04 confirma com
dados que isso acontece porque as categorias têm vocabulário bastante característico: "book"/"author"
para `Books`, "laptop"/"camera" para `Electronics`, "vacuum"/"kitchen" para `Household`,
"women"/"cotton" para `Clothing_Accessories`.

**Quais categorias são mais difíceis?** `Household` continua concentrando boa parte dos erros, tanto
sendo confundida com as outras quanto recebendo previsões que deveriam ser de outras categorias. Faz
sentido: é a categoria mais ampla e heterogênea do catálogo, o que aumenta a sobreposição de
vocabulário com as demais.

**Quais são as limitações deste trabalho, medidas em vez de apenas citadas?**

- A validação cruzada do notebook 03 usa `Pipeline`, refazendo o fit da representação a cada fold.
  Os números de estabilidade resultantes devem ser lidos como confiáveis, sem o vazamento técnico
  que existiria se a representação fosse ajustada uma única vez no treino completo.
- A checagem de duplicatas semânticas do notebook 01 mediu o risco residual de vazamento por
  descrições quase idênticas (não exatas): **21,9% do conjunto de teste (1.218 de 5.561 linhas)
  tem uma quase-duplicata no treino** (similaridade de cosseno sobre TF-IDF ≥ 0,9). Isso é uma
  limitação real que deveria ser tratada antes de um deploy; a métrica de teste reportada aqui pode
  estar levemente otimista por causa disso.
- A checagem de sensibilidade do UMAP à semente (notebook 02) mostrou uma diferença de F1-macro de
  0,0033 entre rodar com e sem semente fixa nesta base. A decisão de não fixar a semente parece
  razoável, mas é uma confirmação empírica pontual (o UMAP é estocástico, então essa diferença varia
  um pouco a cada reexecução), não uma garantia geral.
- Uma representação de embedding monolíngue (por exemplo, via um modelo sentence-transformers
  rodando localmente sobre o texto em inglês) não foi implementada nesta versão. O notebook 02
  compara apenas os embeddings pré-calculados fornecidos (reduzidos por PCA/UMAP) contra o TF-IDF; a
  comparação com um embedding gerado especificamente para este vocabulário segue como trabalho
  futuro.
- O Optuna otimizou hiperparâmetros dos dois finalistas, mas não dos outros três modelos testados na
  comparação inicial (Random Forest, Extra Trees, HistGradientBoosting), que ficaram só com os
  hiperparâmetros padrão.
- A deduplicação continua sendo feita por igualdade exata na etapa de treino do modelo final (a
  checagem semântica do notebook 01 é uma medição de risco, não uma remoção). Remover também as
  quase-duplicatas identificadas seria o próximo passo natural antes de um deploy.
- **Reexecução do pipeline muda os números de forma pequena, mas real**: como o UMAP não fixa
  semente e o Optuna reotimiza a cada rodada, os valores exatos (F1-macro, IC do teste de
  significância, hiperparâmetros escolhidos) variam um pouco entre execuções. As conclusões
  qualitativas (TF-IDF + Logistic vence, a diferença é significativa, Household concentra os erros)
  se mantiveram estáveis nas reexecuções observadas, mas os números de 2 e 3 casas decimais não
  devem ser tratados como fixos.
- **Bug conhecido, não corrigido**: a célula de interpretação por palavra da seção 9 do notebook 03
  (`Interpretabilidade do modelo vencedor`) ajusta um `TfidfVectorizer` novo só para extrair o
  vocabulário, em vez de reaproveitar o vectorizer usado para treinar o modelo — o vocabulário desse
  vectorizer novo pode ficar em ordem diferente do que o modelo foi treinado, produzindo palavras
  sem relação com a categoria. A leitura confiável das palavras mais importantes por categoria é a do
  notebook 04 (que treina vectorizer e modelo juntos, de forma consistente), não a do notebook 03.
- **Hiperparâmetro do modelo final pode ficar desatualizado**: o notebook 04 usa um `CONFIG_FINAL`
  com os hiperparâmetros da Logistic Regression **hardcoded** (copiados manualmente de uma execução
  anterior do Optuna no notebook 03), em vez de carregar o resultado mais recente. Como o Optuna do
  notebook 03 pode retornar um `C` ligeiramente diferente a cada reexecução, o notebook 04 pode
  treinar o "modelo final" com um hiperparâmetro que não é mais o que o notebook 03 encontrou de
  melhor na última rodada. O ideal seria o notebook 03 salvar `melhores_params_log` em disco e o
  notebook 04 carregar esse arquivo, em vez de copiar o valor manualmente.
