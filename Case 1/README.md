# Case 1 - EDA: Perfil e Comportamento de Clientes

Análise exploratória de dados (EDA) sobre uma base de 2.000 clientes, cobrindo perfil demográfico, poder aquisitivo, padrões de consumo e segmentação. Este é o primeiro case do treinamento de Data Science.

## Estrutura da pasta

```
Case 1/
├── EDA.ipynb              # notebook principal da análise
├── data/
│   └── dataset_2k.csv     # base de dados (2.000 clientes, 13 colunas)
└── README.md              # este arquivo
```

## O dataset

`data/dataset_2k.csv` tem 2.000 linhas e 13 colunas, sem valores nulos, duplicados ou `cliente_id` repetido.

| Coluna | Tipo | Descrição |
|---|---|---|
| `cliente_id` | inteiro | identificador único do cliente |
| `idade` | inteiro | idade em anos (18 a 73) |
| `renda_mensal` | decimal | renda mensal em R$ (1.684,33 a 37.051,72) |
| `gasto_mensal` | decimal | gasto mensal em R$ (184,23 a 7.954,77) |
| `frequencia_compra` | decimal | compras por mês, em média (1,0 a 19,6) |
| `ticket_medio` | decimal | valor médio por compra em R$ (44,22 a 3.285,62) |
| `tempo_cliente_anos` | decimal | tempo de relacionamento em anos (0,1 a 18,0) |
| `score_engajamento` | decimal | score de engajamento (-13,4 a 80,2; ver *Pontos de atenção*) |
| `canal` | categórica | canal de aquisição: Aplicativo, Site, Loja física, Marketplace |
| `regiao` | categórica | região do cliente: Sudeste, Nordeste, Sul, Centro-Oeste, Norte |
| `plano` | categórica | plano contratado: Básico, Premium, VIP |
| `data_cadastro` | data | data de cadastro do cliente (jan/2022 a jul/2026) |
| `segmento_latente` | categórica | segmento pré-atribuído: Jovem Digital, Tradicional, Alto Valor |

## Pontos de atenção

- **`score_engajamento` negativo**: 4 clientes (0,2% da base) têm score negativo, mínimo de -13,4. Todos pertencem ao segmento Tradicional. O volume é irrelevante para as conclusões da análise e os registros foram mantidos sem tratamento, mas vale confirmar com quem gera essa métrica se valores negativos são esperados antes de usá-la em modelos ou relatórios.
- A base é desbalanceada entre planos (53% Básico, 35,6% Premium, 11,4% VIP) e entre regiões (Norte e Centro-Oeste têm as menores amostras) — leituras segmentadas por esses grupos menores merecem cautela.
- Os dados de 2026 cobrem só até julho (parcial) — não comparar diretamente com os anos fechados sem anualizar.

## Como rodar

Bibliotecas usadas: `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`, `adjusted_rand_score`).

```bash
pip install pandas matplotlib seaborn scipy scikit-learn jupyter
jupyter notebook EDA.ipynb
```

## Estrutura do notebook

O `EDA.ipynb` segue seis perguntas de negócio, complementadas por explorações adicionais:

1. **Qual é o perfil de idade dos clientes?** distribuição, boxplots por plano e por segmento.
2. **Existem rendas extremamente altas?** distribuição, outliers pelo critério do IQR, ranking dos maiores valores.
3. **Renda e gasto parecem relacionados?** dispersão, reta de regressão, correlações de Pearson e Spearman.
4. **Os planos possuem comportamentos diferentes?** comparação de métricas entre planos, testes ANOVA e Kruskal-Wallis.
5. **Quais variáveis parecem relacionadas?** matriz de correlação (heatmap) das variáveis numéricas.
6. **Existe alguma estrutura visual escondida?** PCA das variáveis numéricas, com dispersão em 2D colorida por `segmento_latente`.

Depois das seis perguntas, a seção **Explorações adicionais** cobre mais seis tópicos:

1. Distribuição de clientes por canal, região e segmento latente.
2. Evolução dos cadastros ao longo do tempo.
3. Frequência de compra x Ticket médio e Engajamento.
4. Engajamento por canal.
5. Tempo de cliente x Engajamento, por plano.
6. Clusterização (KMeans) e validação do `segmento_latente` via Adjusted Rand Index.

## Principais achados

- **`segmento_latente` é a variável que organiza a base**, não `plano`. Idade, renda e estrutura de comportamento variam fortemente por segmento e quase nada por plano contratado.
- **O segmento é recuperável só com os dados numéricos**: PCA + KMeans, sem usar o rótulo `segmento_latente`, reconstrói os três grupos com Adjusted Rand Index de 0,871.
- **Nenhuma das 6 métricas testadas** (renda, gasto, ticket médio, frequência, engajamento, tempo de cliente) mostrou diferença estatisticamente significativa entre os planos Básico, Premium e VIP (ANOVA e Kruskal-Wallis, p > 0,3 em todas).
- **Alto Valor concentra as rendas extremas**: 143 outliers de renda (7,2% da base) são 100% desse segmento, mas representam só 31% dele, ou seja, a maior parte do segmento tem renda alta sem ser estatisticamente atípica.
- **Renda e gasto são fortemente correlacionados** (Pearson 0,85), mas a correlação de Spearman bem mais baixa (0,39) indica que essa relação é puxada pelos clientes de renda mais alta.

## Recomendações de negócio

- Repensar a lógica dos planos: nenhuma métrica muda entre Básico, Premium e VIP.
- Segmentar por `segmento_latente`, não por plano contratado.
- Criar um subcorte dentro de Alto Valor para os clientes com renda outlier (31% do segmento).
- Diferenciar estratégia de engajamento por idade: recorrência para clientes mais velhos, upsell por transação para os mais jovens.
- Investigar a estagnação na aquisição de novos clientes (~25-51 cadastros/mês, sem tendência de crescimento).
