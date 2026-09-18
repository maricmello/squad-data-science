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

A tabela final tem doze modelos: os cinco baselines, os dois clássicos, os quatro
candidatos de Machine Learning e o Prophet. Só o Ridge passou na validação interna
do notebook 03, mas os outros três candidatos (Random Forest, Gradient Boosting e
XGBoost) também foram retreinados com todo o histórico e avaliados nas mesmas 17
origens walk-forward, para efeito de comparação. MAPE e RMSE médios entre os três
horizontes:

<table>
<thead>
<tr><th>Grupo</th><th>Modelo</th><th>MAPE médio</th><th>RMSE médio (US$)</th></tr>
</thead>
<tbody>
<tr><td rowspan="5">Baselines simples</td><td>Holt</td><td>4,88%</td><td>1,12</td></tr>
<tr><td>SES</td><td>5,27%</td><td>1,16</td></tr>
<tr><td>Naïve</td><td>5,30%</td><td>1,17</td></tr>
<tr><td>Holt-Winters</td><td>5,31%</td><td>1,18</td></tr>
<tr><td>Seasonal Naïve</td><td>11,13%</td><td>2,51</td></tr>
<tr><td rowspan="2">Modelos clássicos</td><td>ARIMA</td><td>5,22%</td><td>1,18</td></tr>
<tr><td>SARIMA</td><td>5,22%</td><td>1,18</td></tr>
<tr><td>Modelos lineares</td><td>Ridge</td><td>6,58%</td><td>1,32</td></tr>
<tr><td rowspan="3">Modelos de árvores</td><td>Random Forest</td><td>7,24%</td><td>1,61</td></tr>
<tr><td>XGBoost</td><td>9,40%</td><td>1,83</td></tr>
<tr><td>Gradient Boosting</td><td>10,14%</td><td>2,00</td></tr>
<tr><td>Modelo aditivo</td><td>Prophet</td><td>37,33%</td><td>5,96</td></tr>
</tbody>
</table>

Holt tem o menor erro médio e é o único que bate o Naïve nos três horizontes,
por uma margem pequena em h=1 e h=2. Seasonal Naïve e Prophet ficam bem atrás
dos outros porque os dois usam um valor mais distante no tempo, uma semana atrás
ou a tendência de todo o histórico, em vez do último preço conhecido, e por isso
demoram mais para reagir à queda de 13,7% que fecha o treino, um dia antes do
teste começar.

O Ridge usa 19 features (lags, médias e desvios móveis, retorno, calendário)
e mesmo assim erra mais que o Naïve na maioria dos horizontes. A análise de
correlação do notebook 03 explica isso: lag_1 até lag_20 e as médias móveis
chegam a 0,99 de correlação entre si, porque a série não é estacionária, então
a maior parte dessas variáveis repete a mesma informação do último preço, sem
sinal novo.

Random Forest, Gradient Boosting e XGBoost ficam todos atrás do Ridge nos três horizontes, 
com o Gradient Boosting no fim da fila (10,14% de MAPE médio). Nenhum dos quatro chega perto 
do Holt.
