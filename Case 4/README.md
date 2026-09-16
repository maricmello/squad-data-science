# Case: Previsão de Preço de Fechamento (1, 2 e 3 dias)

Previsão do `Close` de uma ação para 1, 2 e 3 dias úteis à frente, nas mesmas 17
origens de validação walk-forward. Os modelos ficam em três grupos: baselines
simples (Naïve, Seasonal Naïve, SES, Holt e Holt-Winters), modelos clássicos
(ARIMA e SARIMA) e Machine Learning (Ridge, Random Forest, Gradient Boosting e
XGBoost), com o Prophet testado à parte.

## Requirements

```
pip install pandas numpy matplotlib statsmodels scikit-learn xgboost prophet nbformat nbclient ipykernel
```

## Principais resultados

A tabela final tem nove modelos: os cinco baselines, os dois clássicos, o Prophet
e o Ridge. Dos quatro candidatos de Machine Learning, só o Ridge passou na
validação interna nos três horizontes, por isso é o único da família que aparece
abaixo. MAPE e RMSE médios entre os três horizontes:

| Modelo | MAPE médio | RMSE médio (US$) |
|---|---|---|
| Holt | 4,88% | 1,12 |
| ARIMA | 5,22% | 1,18 |
| SARIMA | 5,22% | 1,18 |
| SES | 5,27% | 1,16 |
| Naïve | 5,30% | 1,17 |
| Holt-Winters | 5,31% | 1,18 |
| ml_ridge | 6,58% | 1,32 |
| Seasonal Naïve | 11,13% | 2,51 |
| Prophet | 37,33% | 5,96 |

Holt tem o menor erro médio e é o único que bate o Naïve nos três horizontes,
por uma margem pequena em h=1 e h=2. Seasonal Naïve e Prophet ficam bem atrás
dos outros porque os dois usam um valor mais distante no tempo, uma semana atrás
ou a tendência de todo o histórico, em vez do último preço conhecido, e por isso
demoram mais para reagir à queda de 13,7% que fecha o treino, um dia antes do
teste começar.

O ml_ridge usa 19 features (lags, médias e desvios móveis, retorno, calendário)
e mesmo assim erra mais que o Naïve na maioria dos horizontes. A análise de
correlação do notebook 03 explica isso: lag_1 até lag_20 e as médias móveis
chegam a 0,99 de correlação entre si, porque a série não é estacionária, então
a maior parte dessas variáveis repete a mesma informação do último preço, sem
sinal novo.
