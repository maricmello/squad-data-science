# notebooks_verificados — o que mudou em relação à versão original

Esta pasta é uma versão revisada dos notebooks 01–04, implementando os ajustes
apontados na auditoria técnica do projeto (`auditoria_case2.md`, na raiz de
`Case 2/`). Os quatro notebooks foram **executados de ponta a ponta com os
dados reais** — os números abaixo são resultados reais desta execução, não
estimativas.

Como rodar: esta pasta é autocontida — tem sua própria `data/` (`data/raw/` e
`data/processed/`), em vez de depender da pasta `data/` que fica um nível
acima (a usada pelos notebooks originais). Dá para rodar só esta pasta sem
tocar em `notebooks/` nem em `data/` na raiz de `Case 2/`.

**Antes de rodar os notebooks 01–04 do zero**, copie os dois arquivos brutos
para dentro de `notebooks_verificados/data/raw/`:
- `dataset_ecommerce.csv`
- `embeddings_texto.npy`

Eles já existem em `Case 2/data/raw/` (a pasta original) — é só copiar, não
precisa gerar de novo. Não foram colocados aqui automaticamente porque são
grandes (~37MB e ~77MB) e o canal usado para gravar arquivos nesta pasta a
partir da conversa tem um limite de 20MB por arquivo.

`data/processed/` (o split treino/teste e as representações PCA/UMAP/TF-IDF)
já vem preenchido nesta entrega — gerado pela execução real dos notebooks —
então não é obrigatório rodar tudo de novo só para inspecionar os resultados.
As figuras de todos os notebooks ficam juntas em `figs/`, com o número do
notebook como prefixo do nome do arquivo.

`src/` e `tests/` têm as versões corrigidas de `preprocessing.py` e
`avaliacao.py`. Rode `pytest tests/` de dentro desta pasta para conferir (15
testes, cobrindo as funções novas e as antigas). `scripts/` tem o código-fonte
de cada notebook em formato `jupytext` (`.py`), útil para revisar as mudanças
por diff de texto em vez de JSON de notebook.

## Ajustes de alta prioridade

**1. Vazamento na validação cruzada (notebook 03).** A CV original ajustava o
TF-IDF/UMAP uma única vez no treino completo e reaproveitava isso em todos os
folds do `cross_validate` — um vazamento técnico real, mesmo documentado como
simplificação. Aqui, os dois finalistas viram `sklearn.pipeline.Pipeline`
(representação + modelo), e o `cross_validate` recebe dados crus (texto para
o TF-IDF, embeddings densos para o UMAP), refazendo o fit da representação a
cada fold.
*Resultado real*: a CV corrigida do XGBoost+UMAP caiu de F1-macro
0,947 (±0,002, com vazamento) para 0,941 (±0,004, sem vazamento) — confirmando
empiricamente que a CV original estava otimista. A CV da Logistic+TF-IDF
mudou pouco (0,946 nos dois casos), porque o TF-IDF é menos sensível a esse
tipo de vazamento quando o vocabulário já satura em 5000 palavras.

**2. Interpretabilidade do modelo vencedor (notebooks 03 e 04).** Extraímos os
coeficientes da Logistic Regression sobre o vocabulário do TF-IDF.
*Resultado real*: as palavras de maior peso por categoria fazem sentido de
negócio imediato — `Books`: book, author, guide; `Clothing_Accessories`:
women, men, cotton; `Electronics`: laptop, camera, lens; `Household`: vacuum,
door, kitchen. Isso confirma com dados a hipótese que a versão original só
levantava como plausível.

**3. Assert tautológico (notebook 02).** O assert original
(`assert embeddings.shape[0] == len(df)`) era checado *depois* de indexar por
`df.index.values`, e por isso sempre passava, mesmo que o alinhamento
estivesse errado. A nova função `carregar_embeddings_alinhados` (em
`src/preprocessing.py`) checa o shape do array *antes* de indexar, contra o
número de linhas esperado (guardado em `df.attrs['n_linhas_sem_nulos']` por
`carregar_e_limpar`). Coberto por teste unitário que injeta um desalinhamento
proposital e confirma que a função levanta erro.

## Ajustes de média prioridade

**4. Split recalculado em cada notebook.** `obter_ou_criar_split` persiste
`idx_train`/`idx_test` em `data/processed/split.npz` na primeira chamada (notebook
01) e todos os demais notebooks carregam o mesmo arquivo, em vez de confiar
apenas na semente para produzirem o mesmo resultado.

**5. Duplicatas semânticas nunca medidas.** Nova seção no notebook 01, usando
similaridade de cosseno sobre TF-IDF. *Resultado real*: **21,85% do conjunto
de teste tem uma quase-duplicata (similaridade ≥ 0,9, não idêntica) no
treino** — um risco de vazamento residual real e não desprezível, que a
versão original só citava como hipótese para trabalho futuro. Isso é
provavelmente a descoberta mais importante desta revisão: sugere que a
métrica de teste reportada (F1-macro ≈ 0,95) pode estar levemente otimista, e
que uma deduplicação por similaridade (não só exata) deveria ser o próximo
passo antes de qualquer deploy.

**6. Sem teste de significância entre os finalistas.** Nova função
`comparar_modelos_bootstrap` (bootstrap pareado, 3000 reamostragens) no
notebook 03. *Resultado real*: diferença de F1-macro de +0,0123 a favor da
Logistic Regression, IC 95% [+0,0066, +0,0181], p ≈ 0,0000 — a vantagem é
estatisticamente significativa, não ruído de amostragem.

**7. Sem métrica de negócio.** Nova função `cobertura_por_confianca` e seção 7
do notebook 04. *Resultado real*: com threshold de 90% de confiança, 78,4% do
catálogo poderia ser autoclassificado com 98,6% de accuracy nesse
subconjunto, deixando 1201 de 5561 produtos do teste para revisão manual —
uma régua concreta para decidir, com a área de negócio, onde vale a pena
automatizar.

**8. UMAP sem semente fixa nunca testado quanto à sensibilidade.** Nova seção
6 do notebook 02: refit do UMAP 10D com `random_state=42` fixo, comparado
contra a versão sem semente com um modelo simples. *Resultado real*: diferença
de F1-macro de 0,0011 entre as duas versões — pequena o suficiente para não
mudar a conclusão de qual representação é melhor, o que dá algum respaldo
empírico (pontual, não uma garantia geral) à decisão original de priorizar
velocidade.

## Ajustes de baixa prioridade

**9. Embedding monolíngue nunca testado.** Nova seção 8 do notebook 02,
implementada e pronta para rodar — mas **não pôde ser executada neste
ambiente**: o download do modelo `sentence-transformers/all-MiniLM-L6-v2` via
Hugging Face Hub é bloqueado pela política de rede deste workspace de nuvem
(`ProxyError: 403 Forbidden`). O notebook trata esse erro sem quebrar a
execução e documenta o bloqueio. Quem rodar este notebook num ambiente com
acesso liberado ao Hugging Face Hub (por exemplo, localmente) vai conseguir
completar esse teste sem alterar nenhum código.

**10. ROC-AUC redundante.** PR-AUC macro (`pr_auc_macro`) agora é reportado ao
lado do ROC-AUC nos notebooks 03 e 04, com uma nota explicando por que o
ROC-AUC comunica pouco neste problema (satura perto de 1).

**11. Sem testes unitários.** `tests/test_preprocessing.py` e
`tests/test_avaliacao.py` — 15 testes cobrindo limpeza de dados, alinhamento
de embeddings, persistência de split, medição de duplicatas semânticas,
teste de significância estatística, extração de interpretabilidade e
cobertura por confiança.

**12. Sem plano de monitoramento.** Nova seção 8 do notebook 04: tabela de
indicadores propostos (distribuição de confiança, cobertura de
autoclassificação, distribuição de classes previstas, taxa de correção
manual) com a lógica de por que cada um sinalizaria um problema.

## O que não mudou

A metodologia que a auditoria já considerava correta foi mantida sem
alteração: deduplicação exata antes do split, `fit` das representações
(PCA/UMAP/TF-IDF) só no treino, Optuna usando apenas uma fatia de validação
recortada do treino (nunca o teste), baseline obrigatório, e otimização dos
dois melhores candidatos em pé de igualdade (não só o modelo escolhido a
priori). Os números finais de teste (F1-macro ≈ 0,949 para a Logistic
Regression) permanecem consistentes com a versão original — a correção do
vazamento na CV não muda a métrica final de teste, que já era calculada
corretamente; ela muda a confiabilidade da estimativa de estabilidade entre
folds.
