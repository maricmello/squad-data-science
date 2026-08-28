# Projeto: Onde ajudar países emergentes?

**Case:** um milionário quer usar seu dinheiro para ajudar países emergentes. Quais países
escolher, e por quê?

A resposta usa clusterização (K-Means) para agrupar 167 países por indicadores de saúde,
economia e demografia, encontrar o grupo em situação mais crítica e, dentro dele, ranquear os
países que precisam de atenção mais urgente. 

## Estrutura do projeto

```
projeto-paises-emergentes/
│
├── notebooks/
│   ├── 01_EDA.ipynb                     → exploração dos dados brutos
│   ├── 02_Preprocessamento.ipynb        → padronização (z-score) das variáveis
│   ├── 03_Selecao_de_Features.ipynb     → análise de multicolinearidade e escolha das features
│   ├── 04_Clusterizacao.ipynb           → escolha de k, K-Means final, validação hierárquica
│   ├── 05_Analise_dos_Clusters.ipynb    → perfil e nome de cada grupo, visualização com PCA
│   └── 06_Ranking_de_Atratividade.ipynb → score de prioridade dentro do grupo mais vulnerável
│
├── data/
│   ├── raw/                             → dataset original
│   │   ├── dataset_paises.csv
│   │   └── dicionario_dataset_paises.csv
│   └── processed/                       → gerado ao rodar os notebooks
│
├── src/
│   ├── preprocessing.py                 → carga de dados e padronização
│   ├── clustering.py                    → K-Means, hierárquico, score de prioridade
│   └── visualization.py                 → gráficos usados nos notebooks
│
├── requirements.txt
├── run_all.py               → roda todos os notebooks em ordem, de uma vez
└── README.md
```

Cada notebook lê o que o anterior salvou em `data/processed/`, por isso a ordem importa. 

## Como rodar

1. Instalar as dependências: `pip install -r requirements.txt`.
2. Rodar os notebooks. Duas opções:
   - Tudo de uma vez: `python run_all.py`, na raiz do projeto. Roda os seis notebooks em
     ordem e salva os resultados de volta em cada um.
   - Um por um: abrir em ordem, a partir de `notebooks/01_EDA.ipynb`, e rodar todas as
     células (`Restart Kernel and Run All`). Cada um assume que o(s) anterior(es) já
     rodaram, porque depende dos arquivos que eles salvam em `data/processed/`.

### Dependências

Listadas em `requirements.txt`: pandas, numpy, scikit-learn, matplotlib, scipy, joblib e
plotly, nbformat e nbclient

Testado com: pandas 3.0, numpy 2.4, scikit-learn 1.8, matplotlib 3.10, scipy 1.17, joblib 1.5,
plotly 7.0.

## O dado

`data/raw/dataset_paises.csv` traz, para 167 países, os indicadores abaixo (descrição
completa em `data/raw/dicionario_dataset_paises.csv`):

| Coluna | O que é |
|---|---|
| `mortalidade_infantil` | Mortes de crianças menores de 5 anos por 1.000 nascidos vivos |
| `exportacoes` | Exportações de bens e serviços, % do PIB |
| `saude` | Gasto total com saúde, % do PIB |
| `importacoes` | Importações de bens e serviços, % do PIB |
| `renda` | Renda líquida por pessoa |
| `inflacao` | Taxa anual de crescimento do PIB total |
| `expectativa_vida` | Expectativa de vida ao nascer |
| `fertilidade_total` | Número médio de filhos por mulher |
| `pib_per_capita` | PIB total dividido pela população |

## Conclusão

A análise dos 167 países revelou três grupos: 31 desenvolvidos, 89 em desenvolvimento e 47
vulneráveis. O último grupo concentra os maiores problemas de saúde e renda, sendo o mais
indicado para receber ajuda.

Os 10 países prioritários são: Haiti, República Centro-Africana, Chade, Níger, Mali, Angola,
Moçambique, Nigéria, Guiné e República Democrática do Congo. Haiti se destaca como o caso mais
crítico, seguido por República Centro-Africana e Chade.

A recomendação seria concentrar os recursos nesses países, principalmente em saúde básica,
pré-natal, vacinação e combate à desnutrição infantil. Em Angola e Nigéria, porém, seria mais
adequado investir diretamente em serviços de saúde e infraestrutura, já que possuem mais
recursos financeiros, mas ainda apresentam indicadores de saúde muito ruins.

## Limitações

- Os dados são um retrato estático, não capturam conflitos, instabilidade política ou
  eventos recentes.
- Clusterização agrupa países parecidos, mas não explica causa: mostra onde a situação é mais
  grave, não por que cada país chegou lá.
- O score de prioridade é uma escolha razoável entre várias possíveis, outros pesos ou outras
  variáveis podem mudar a ordem dentro do top 10, especialmente para países fronteiriços entre
  clusters.
- Por isso, a decisão final deveria combinar esses resultados com dados atualizados e
  conhecimento local sobre cada país.
