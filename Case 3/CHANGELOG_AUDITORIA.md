# Changelog — correções pós-auditoria técnica

Este documento mapeia cada problema encontrado na auditoria técnica externa
para a correção efetivamente implementada neste repositório. Nenhum ponto da
auditoria foi ignorado; onde uma correção não mudou o resultado (ex.:
`log1p(renda)`), isso também está documentado, com o resultado do teste.

## Problemas críticos → correção

**1. Seleção do modelo final usava o conjunto de teste.**
Antes: `run_all.py` e o notebook 05 comparavam Ridge tunado e GradientBoosting
tunado pelo RMSE **no teste** e escolhiam o menor — uma diferença de 0,44%,
menor que o desvio-padrão entre folds de CV (10–19).
Depois: a escolha usa o **RMSE médio de cross-validation no treino**
(`search.best_score_` do `RandomizedSearchCV`), calculado antes de tocar o
teste. O teste só é usado depois, para (a) reportar a métrica final do
modelo já escolhido e (b) rodar um teste de significância (bootstrap
pareado) entre os dois candidatos. Resultado: o modelo final mudou de
**GradientBoosting** para **Ridge**, e o bootstrap confirma que a diferença
entre os dois nunca foi estatisticamente significativa (IC 95% da diferença
de RMSE inclui zero). Ver `src/run_all.py::tune()` e `src/evaluation.py::bootstrap_compare_rmse()`.

**2. Notebooks entregues não batiam com os artefatos finais salvos.**
Antes: notebooks 02 e 05 imprimiam caminhos de um ambiente Linux
(`/tmp/case3/project/...`) diferente do projeto real, e os números de CV do
notebook 04 não batiam com `cv_results.csv`.
Depois: todos os 6 notebooks foram reconstruídos e executados de ponta a
ponta, em ordem (`Restart & Run All`), a partir de `data/processed/` e
`models/` zerados. Os resultados foram conferidos contra uma execução
independente de `python src/run_all.py`: os dois caminhos chegam ao mesmo
modelo final (Ridge tunado) e às mesmas métricas de teste na precisão
reportada (MAE, RMSE, R², MAPE idênticos nas 2 casas decimais usadas no
README). Uma diferença residual na 10ª-15ª casa decimal em alguns valores
brutos de `RandomForest` (não usado no modelo final) foi observada entre
execuções — é ruído de ponto flutuante do backend paralelo do
`RandomForestRegressor` (`n_jobs=-1`), não um problema de execução
inconsistente, e não afeta nenhuma conclusão do projeto.

**3. Ausência de dicionário de dados.**
Criado `DATA_DICTIONARY.md`: definição assumida (não confirmada com a fonte)
de cada variável, o range observado, e exatamente o que precisa ser validado
antes de confiar na suposição de ausência de vazamento temporal. Inclui os
testes indiretos de vazamento (correlação `ticket × frequência` vs. target =
0,17; VIF ≈ 1,0 entre todas as features) que foram feitos para reduzir
(não eliminar) esse risco.

**4. `idade` implausível identificada mas não tratada.**
Antes: 39 linhas com `idade` < 18 anos ficavam no dataset sem tratamento.
Depois: `data_prep.clean_data()` remove essas linhas por padrão
(`filtrar_idade_invalida=True`). A decisão foi baseada em teste de
sensibilidade (10 seeds): o efeito médio no RMSE é neutro/levemente negativo
(365,4 → 370,1 numa comparação; e o erro absoluto médio dessas 39 linhas é
*menor*, não maior, que o do resto da base) — ou seja, **a remoção não é
motivada por ganho de performance**, e sim por validade do dado. Ver
notebook 01, seção "Tratamento de idade implausível".

## Melhorias de média prioridade → implementadas

**5. EDA univariada ausente para as features.**
Adicionada seção completa no notebook 01: histogramas de todas as 9
features, tabela de outliers via IQR por feature, e VIF formal (antes só
havia inspeção visual do heatmap de correlação). Achado novo: `renda` tem
80/2200 linhas (3,6%) acima do limite superior de IQR, com máximo ~4,8x o
percentil 75 — mantido sem tratamento por serem valores plausíveis (clientes
de alta renda), ao contrário do caso de `idade`.

**6. Sem métricas segmentadas por grupo de negócio.**
Adicionada `evaluation.segment_metrics()` e usada nos notebooks 04 e 06 para
reportar MAE/RMSE/MAPE por quartil de `renda`. Achado: MAPE varia de ~14%
(quartis mais baixos) a ~9,7% (quartil mais alto) — informação que a métrica
agregada escondia. Ver `data/processed/metrics_por_faixa_renda.csv`.

**7. MAPE ausente.**
Adicionado a `evaluation.regression_metrics()` — agora todo lugar que
reporta MAE/RMSE/R² também reporta MAPE (baseline: 20,4%; modelo final:
12,9%).

**8. Sem teste estatístico formal comparando os candidatos finais.**
Adicionado `evaluation.bootstrap_compare_rmse()` — bootstrap pareado (5.000
reamostragens) comparando Ridge tunado e GradientBoosting tunado no teste.
Resultado: diferença não significativa a 95% (ver item 1 acima). Usado tanto
em `run_all.py` quanto no notebook 05, com a leitura explícita de que os
modelos são estatisticamente equivalentes.

## Melhorias de baixa prioridade → implementadas

**9. `log1p(renda)` nunca testado.**
Testado no notebook 03 em 10 splits diferentes: piora o RMSE de forma
consistente (médias de 370,1 vs. 377,1). **Não adotado** — resultado
negativo documentado, não uma omissão.

**10. SHAP local só mostrava o pior caso.**
Adicionados no notebook 06 dois exemplos de "acerto típico" (erro absoluto
mais próximo da mediana), ao lado do caso de maior erro já existente, para
dar contraste entre onde o modelo funciona bem e onde falha.

**11. Sem registro de versões de bibliotecas usadas no modelo salvo.**
Criado `models/model_card.json`, gerado por `run_all.py`: versões de
scikit-learn, xgboost, shap, pandas, numpy, hiperparâmetros do modelo final,
`random_state`, tamanho de treino/teste e lista de features — para
reprodutibilidade futura do artefato `.joblib`.

**12. Sem plano de monitoramento pós-deploy.**
Adicionada a seção "Plano mínimo de monitoramento pós-deploy" no `README.md`
com indicadores concretos (qualidade realizada, drift de `renda`, taxa de
extrapolação, taxa de `idade` implausível na entrada, revalidação da
suposição temporal, cadência de retreino). É um plano, não uma
funcionalidade implementada neste repositório.

## O que NÃO foi resolvido (limitação que segue em aberto)

A suposição central do projeto — que as features não têm sobreposição
temporal com `gasto_mensal` — **continua sem confirmação com a fonte de
dados**, porque essa fonte não está disponível para este trabalho. O
`DATA_DICTIONARY.md` deixa exatamente isso explícito, em vez de tratar a
suposição como resolvida. Qualquer uso em produção deve começar por essa
validação.
