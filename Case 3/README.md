# Case Final: Previsão de Gasto Mensal

Pipeline completo de regressão para prever `gasto_mensal` de cada cliente a
partir do seu perfil e comportamento recente, seguindo os 12 passos do case
final do treinamento de Ciência de Dados (regressão).

## Objetivo de negócio

Estimar o gasto mensal esperado por cliente para apoiar priorização de ações
de CRM, dimensionamento de metas de receita por carteira e identificação de
clientes gastando abaixo do esperado para o seu perfil.


## Estrutura do projeto

```
CASE 3/
├── data/
│   ├── raw/                  # dado bruto
│   └── processed/            # artefatos gerados pelo pipeline (split, métricas, modelo)
├── notebooks/
│   ├── 01_EDA.ipynb                                  # entender a target, EDA univariada de todas as features, 
                                                      #  tratamento de idade implausível 
                                                      #  e tempo_cliente inconsistente (com teste de sensibilidade)
│   ├── 02_Preprocessamento.ipynb                      # split treino/teste e baseline (métricas do baseline salvas em data/processed/baseline.json)
│   ├── 03_Modelagem_e_Comparacao.ipynb                # comparar modelos, métricas (inclui MAPE), previsto x observado, experimento log1p(renda)
│   ├── 04_Residuos_e_Cross_Validation.ipynb           # resíduos, estabilidade via CV, robustez por faixa de renda
│   ├── 05_Tuning_e_Modelo_Final.ipynb                 # tuning e escolha do modelo final por cross-validation, com teste de significância entre candidatos
│   └── 06_Interpretabilidade_e_Validacao_Negocio.ipynb # feature importance/SHAP (casos de erro e de acerto) e validação de negócio
├── pdf/
│   └── regression.pdf        # material teórico do treinamento
├── models/
│   ├── modelo_final.joblib   # modelo final tunado, salvo pelo notebook 05 / run_all.py
│   └── model_card.json       # versões de bibliotecas, hiperparâmetros e metadados do modelo final
├── src/
│   ├── data_prep.py          # carregamento, limpeza (idade implausível e tempo_cliente inconsistente), split
│   ├── modeling.py           # definição dos modelos e grades de tuning
│   ├── evaluation.py         # métricas (MAE/RMSE/R²/MAPE), cross-validation, heterocedasticidade, métricas por segmento, teste de significância (bootstrap)
│   ├── interpretability.py   # importância nativa, permutation importance, SHAP
│   ├── visualization.py      # estilo e funções de plot compartilhadas
│   └── run_all.py            # roda o pipeline inteiro fora do Jupyter (reprodutibilidade)
├── DATA_DICTIONARY.md        # definição assumida de cada variável e o que precisa ser validado com a fonte
├── README.md
└── requirements.txt
```


## Como rodar

```bash
pip install -r requirements.txt

# opção A: rodar os notebooks em ordem (Jupyter)
jupyter lab notebooks/

# opção B: rodar o pipeline inteiro de uma vez, sem abrir o Jupyter
python src/run_all.py
```


## Resultado do modelo final

| Métrica | Baseline (média) | Modelo final |
|---|---|---|
| MAE | R$ 418,9 | **R$ 288,3** |
| RMSE | R$ 520,5 | **R$ 357,4** |
| R² | ~0,00 | **0,53** |
| MAPE | 19,8% | **13,1%** |

- **Modelo final: Ridge** (`alpha=10`, tunado via `RandomizedSearchCV`, 5-fold CV).
  Escolhido entre os dois melhores candidatos (Ridge tunado e GradientBoosting
  tunado) pelo **RMSE médio de cross-validation no treino** (363,7 vs. 370,7).
 
- Um teste de significância (bootstrap pareado, 5.000 reamostragens) mostra
  que a diferença de RMSE entre Ridge tunado e GradientBoosting tunado no
  teste (357,4 vs. 346,6) **não é estatisticamente significativa** (IC 95%
  da diferença inclui zero, entre -3,1 e 24,4; p ≈ 0,13). Na prática, os dois
  modelos são equivalentes em desempenho preditivo, e Ridge foi escolhido por
  ser igualmente bom e mais simples/interpretável, com base na evidência mais
  confiável disponível (CV no treino), não numa diferença de teste dentro do
  ruído.
- Principais direcionadores (convergem entre importância nativa, permutation
  e SHAP): `renda`, `desconto`, `tempo_cliente`, `engajamento`.



### Leitura para o negócio

- **Erro em reais:** o modelo erra, em média, cerca de **R$ 288,3 por cliente por mês**, o equivalente a **12,2%** do gasto médio mensal (MAPE de 13,1%). Essa precisão é compatível com uso para *ranking* e priorização de clientes, mas não para decisões que exijam exatidão financeira linha a linha.
- **Ganho sobre o baseline:** o RMSE cai **31,3%** frente a simplesmente prever a média, e o MAPE cai de 19,8% para 13,1%, mostrando que o perfil do cliente possui informação real e utilizável.
- **Estabilidade:** o desvio-padrão do RMSE entre os folds de CV é de cerca de R$ 16 (~4,4% da média), indicando que o resultado não depende de uma divisão de dados favorável por sorte.
- **Resíduos:** não há heterocedasticidade forte nem viés sistemático por faixa de previsão (Breusch-Pagan p ≈ 0,77).
- **Erro por segmento:** o erro relativo (MAPE) é menor para clientes de renda mais alta (~10,6% no quartil de renda mais alta) e um pouco maior nas faixas de renda mais baixa (~14,2%). Decisões de priorização devem considerar essa diferença, já que a métrica agregada sozinha não mostra isso (ver notebook 04 e `data/processed/metrics_por_faixa_renda.csv`).

### Experimentos testados e não adotados


- **Remoção de `idade` implausível**: o efeito médio no RMSE, testado em 10 splits, é neutro a levemente negativo, a remoção é mantida por validade do dado (idade abaixo de 18 anos é implausível para um cliente com histórico de compras), não por ganho de métrica. Ver notebook 01.
- **Remoção de `tempo_cliente` inconsistente com a idade**: 89 linhas em que o cliente teria começado a comprar antes dos 18 anos (`tempo_cliente > idade - 18`). Diferente da idade, aqui o RMSE médio de teste melhora de forma mais consistente ao remover (370,1 para 362,1 em 10 splits), mas o erro absoluto médio dessas linhas é menor que o do restante da base, ou seja, não são casos difíceis para o modelo. A melhora de RMSE é um efeito colateral de mudar a composição da base, não o motivo da remoção: a decisão segue o mesmo critério de validade do dado usado para a idade. Ver notebook 01.

### Limitações

1. **Dados sem timestamp e sem dicionário oficial:** a suposição sobre o "momento da previsão" (e a definição exata de cada feature) precisa ser validada com o pipeline de dados real da empresa antes de qualquer uso em produção. Ver `DATA_DICTIONARY.md`.
2. **`idade` implausível e `tempo_cliente` inconsistente:** 39 registros com idade < 18 anos e mais 89 registros em que `tempo_cliente` seria incompatível com a idade são removidos do treino/teste (ver notebook 01), mas a causa raiz (por que a fonte gera esses valores) não foi investigada, só o sintoma foi corrigido no dataset.
3. **R² moderado (0,53):** o modelo explica boa parte da variação do gasto, mas ainda existem fatores não presentes nos dados, como sazonalidade, campanhas e categoria de produto.
4. **Generalização:** previsões para clientes com perfil muito diferente dos dados observados (ex.: renda muito acima do observado) devem ser tratadas com cautela.
5. **Desempenho por segmento:** erro relativo um pouco maior para clientes de renda mais baixa (ver acima), relevante se o modelo for usado para decisões que afetem esse segmento de forma desproporcional.

