# Guia comentado dos notebooks — Case 2 (classificação de produtos por texto)

Este documento explica, célula a célula, o que cada notebook faz e por que foi feito daquele jeito.
A ideia é que alguém que nunca abriu o projeto consiga acompanhar o raciocínio sem precisar rodar
nada — e que quem já conhece o projeto tenha uma referência rápida do "por quê" por trás de cada
decisão técnica. Os quatro notebooks rodam em sequência (01 → 02 → 03 → 04); cada um consome o que
o anterior salvou em `data/processed/`.

---

## 01_eda_preprocessamento — o que temos nos dados?

Esse é o notebook de reconhecimento de terreno. Antes de treinar qualquer coisa, ele responde três
perguntas básicas: os dados estão limpos, as categorias estão balanceadas, e existe algum jeito de
vazar informação do teste pro treino sem perceber?

**Célula [2] (imports):** carrega pandas, sklearn e as funções auxiliares de `src/preprocessing.py`
(`carregar_e_limpar`, `obter_ou_criar_split`, `medir_duplicatas_semanticas`). Nada de especial aqui,
é só a bagagem que o resto do notebook usa.

**Célula [4] (carregamento):** lê o CSV cru (`dataset_ecommerce.csv`) direto, sem nenhuma limpeza,
só pra ver a cara real dos dados antes de mexer neles. É um hábito saudável: nunca aplicar limpeza
"de cabeça" sem antes olhar o que está entrando.

**Células [6]-[7] (qualidade dos dados):** `df.info()` mostra tipos e contagem de não-nulos; a
célula seguinte soma nulos por coluna, conta linhas 100% duplicadas, textos duplicados e textos
vazios, e lista as categorias únicas antes de qualquer padronização. É o checklist de higiene básica
que todo dataset novo merece antes de virar feature.

**Célula [9] (limpeza):** roda `carregar_e_limpar` duas vezes — uma vez **sem** remover duplicatas
(`df_sem_dedup`) e outra **com** a remoção padrão (`df`). Isso é proposital: guardar a versão
intermediária permite medir, na célula seguinte, o efeito de cada etapa da limpeza separadamente
(quantas linhas caíram por nulo, quantas por duplicata) em vez de só reportar o resultado final sem
explicar de onde veio a perda.

**Célula [11] (por que remover duplicatas exatas):** aqui está o argumento quantitativo, não só a
afirmação. Ele pega o dataframe *sem* deduplicação, faz um split treino/teste nele, e mede quantas
linhas do teste têm uma cópia idêntica (mesmo texto + mesma categoria) no treino. O resultado dessa
conta é a justificativa real para remover duplicatas exatas antes do split de verdade: se uma boa
fração do teste já apareceu no treino, a métrica de avaliação fica inflada — o modelo estaria sendo
"testado" com algo que já decorou.

**Célula [13] (distribuição das categorias):** conta quantos produtos existem por categoria, calcula
o percentual e a razão entre a maior e a menor classe, e plota um barplot. Isso importa porque define
decisões mais à frente: se as classes fossem muito desbalanceadas, faria sentido usar
`class_weight='balanced'` ou métricas menos sensíveis a desbalanceamento (o F1-macro, usado no
projeto inteiro, já é uma escolha nessa direção).

**Células [15] e [17] (tamanho do texto):** medem tamanho em caracteres e em número de palavras, no
geral e por categoria (boxplot). A pergunta de fundo é se o comprimento da descrição já entrega uma
pista sobre a categoria — o que ajudaria a explicar por que um modelo simples baseado em palavras
(TF-IDF) funciona tão bem mais à frente.

**Célula [19] (exemplos de texto por categoria):** imprime um exemplo real de cada categoria. Parece
trivial, mas é o tipo de checagem manual que pega problema que estatística nenhuma pega sozinha —
por exemplo, uma categoria mal rotulada ou um texto claramente fora do padrão.

**Célula [21] (encoding do target):** transforma a coluna `categoria` (texto) em números com
`LabelEncoder`. A justificativa está no markdown acima: como é o alvo (não uma feature de entrada),
não faz sentido usar One-Hot aqui — o modelo de classificação já lida com o inteiro codificado
internamente.

**Célula [23] (split treino/teste):** chama `obter_ou_criar_split`, que grava os índices em
`data/processed/split.npz` na primeira vez que roda. Esse é um ponto de design importante do projeto
inteiro: em vez de cada notebook fazer seu próprio `train_test_split` com a mesma seed (o que
funciona só *enquanto* ninguém muda a ordem das linhas ou a versão do sklearn), o split é persistido
em disco uma única vez e recarregado pelos quatro notebooks. Garante que todos usem exatamente o
mesmo treino/teste por construção, não por coincidência.

**Célula [25] (duplicatas semânticas):** essa é a análise mais reveladora do notebook. A
deduplicação exata (célula 9) só pega textos idênticos — mas não pega descrições quase idênticas,
com uma palavra trocada, uma pontuação diferente, uma cor diferente. Para medir esse risco residual,
`medir_duplicatas_semanticas` calcula a similaridade de cosseno (sobre TF-IDF) entre cada texto de
teste e todos os textos de treino, e conta quantos têm um "vizinho" no treino com similaridade
≥ 0,9. O resultado (uma fração nada desprezível do teste com quase-duplicata no treino) é uma
limitação real do projeto, documentada com número em vez de deixada como hipótese — e é isso que
qualifica a métrica final de teste como "possivelmente um pouco otimista".

---

## 02_embeddings_reducao_dimensional — como representar o texto?

Esse notebook não treina nenhum classificador de verdade — o trabalho dele é preparar as diferentes
formas de transformar texto em números que serão comparadas no notebook 03. Duas rotas são
exploradas em paralelo: embeddings densos (já calculados, vindos de `embeddings_texto.npy`)
comprimidos por PCA ou UMAP, e TF-IDF, que representa o texto diretamente pelas palavras.

**Célula [1] (imports) e [2] (carregar dados + split):** repete a mesma limpeza e o mesmo split do
notebook 01 — mas como o split já foi persistido, `obter_ou_criar_split` só recarrega o arquivo em
vez de gerar um novo, garantindo consistência entre notebooks.

**Célula [4] (carregar embeddings):** usa `carregar_embeddings_alinhados` para ler o `.npy` de
embeddings e alinhá-lo com o dataframe. Vale mencionar por que essa função existe: numa versão
anterior do projeto, o alinhamento entre o array de embeddings e as linhas do dataframe era checado
*depois* de já ter indexado por posição, então um desalinhamento nunca seria pego — o assert sempre
passava. Aqui o shape é checado *antes* de indexar, contra o número de linhas esperado, então um
desalinhamento real de fato quebra a execução em vez de passar silenciosamente.

**Célula [6] (PCA 10/20/30D):** ajusta o PCA só no treino (`fit_transform`) e aplica no teste só com
`transform` — nunca o contrário. É a regra de ouro de qualquer redução de dimensionalidade: se o
teste participasse do ajuste, a representação já "veria" a estrutura do teste antes da avaliação, o
que é uma forma sutil de vazamento. O resultado de cada dimensão é salvo em `.npz` para o notebook 03
consumir sem precisar recalcular.

**Célula [8] (UMAP 10/20/30D):** mesma lógica do PCA (fit só no treino), mas aqui o `random_state` é
deliberadamente deixado solto, para poder usar `n_jobs=2` e rodar mais rápido — o UMAP paralelizado
não é determinístico bit a bit mesmo com seed fixa, então fixar a seed aqui não traria garantia total
de qualquer forma, só desacelerar. Essa escolha é testada de verdade na célula 13. `sanitizar_reducao`
entra aqui porque o UMAP ocasionalmente produz `NaN` ao fazer `transform` no teste; a função troca
esses valores pela média do treino e avisa quando isso acontece, em vez de deixar o `NaN` se propagar
silenciosamente para o modelo seguinte.

**Célula [10] (TF-IDF):** monta a representação TF-IDF com no máximo 5000 palavras e ignorando
termos que aparecem em menos de 2 documentos (`min_df=2`, um filtro simples contra ruído de
vocabulário raro). Assim como o PCA e o UMAP, o `fit` acontece só no treino. O resultado é dois
arrays esparsos, salvos separadamente porque `np.savez` não lida bem com matrizes esparsas.

**Células [12]-[13] (UMAP com semente fixa):** essa é a checagem de robustez da decisão tomada na
célula 8. Refaz o UMAP 10D com `random_state=42` fixo e `n_jobs=1` (célula 12), depois treina um
Random Forest simples em cima da versão com semente e da versão sem semente, comparando o F1-macro
das duas (célula 13). A pergunta é direta: será que essa aleatoriedade que aceitamos em troca de
velocidade muda o resultado o suficiente para importar? O código compara a diferença contra um limiar
de 0,01 e imprime uma conclusão automática — é uma forma de transformar uma decisão de design
("não vou fixar seed por causa de performance") em algo verificado, não só assumido.

**Célula [15] (visualização UMAP 2D):** projeta todos os embeddings em 2D só para inspeção visual,
colorido por categoria. Essa é a única exceção às regras acima: aqui o UMAP roda sobre o dataset
inteiro (treino + teste juntos), porque essa projeção não alimenta nenhum modelo — é puramente
exploratória, para enxergar visualmente se as categorias já se separam no espaço de embeddings antes
mesmo de qualquer classificador entrar em cena.

---

## 03_modelagem_avaliacao — qual abordagem funciona melhor?

Esse é o notebook mais denso do projeto: compara representação × modelo, otimiza os dois melhores
candidatos, valida sem vazamento e testa se a diferença entre eles é estatisticamente real.

**Célula [1] (imports):** além das bibliotecas de sempre, traz `optuna` (otimização de
hiperparâmetros), `xgboost`, e as funções do projeto (`avaliar_modelo`, `comparar_modelos_bootstrap`,
`top_features_por_classe`, `pr_auc_macro`, `construir_modelos_candidatos`, `rodar_grade_comparativa`).

**Célula [2] (carregar dados):** recarrega o dataframe limpo, o split e monta `texto_train`/
`texto_test`. Vale registrar aqui uma correção feita depois da primeira execução: originalmente essas
duas linhas eram `df['texto'].values[idx_train]`. Em versões recentes do pandas, quando a coluna
`texto` fica com dtype de string do PyArrow, `.values` devolve um `ArrowExtensionArray` em vez de um
array numpy comum — e esse tipo não aceita indexação por um array de inteiros (fancy indexing), só
índice único ou slice. Isso passava despercebido até a validação cruzada da célula 15, onde o
`cross_validate` do sklearn indexa os folds internamente com arrays de inteiros e quebrava com
`TypeError: only integer scalar arrays can be converted to a scalar index`. A correção
(`.to_numpy(dtype=object)` antes de indexar) converte para um array numpy comum logo na origem, então
qualquer uso posterior de `texto_train`/`texto_test` — inclusive dentro do `Pipeline` da célula 15 —
fica imune ao problema.

**Célula [4] (baseline):** treina um `DummyClassifier` que sempre prevê a classe majoritária, usando
a representação PCA-10D só porque qualquer uma serviria (o Dummy ignora o X, decide só olhando o y).
Esse número (F1-macro de 0,138) é o piso: qualquer modelo real só vale a pena se superar isso por uma
margem clara — é o contrato que o projeto assume desde o início e cobra de si mesmo lá na conclusão.

**Célula [6] (grade comparativa):** roda `rodar_grade_comparativa`, que testa 5 modelos (Logistic
Regression, Random Forest, Extra Trees, HistGradientBoosting, XGBoost) contra todas as representações
salvas no notebook 02 (PCA e UMAP em 10/20/30D, e TF-IDF). É uma busca ampla e barata antes de
qualquer otimização fina — a ideia é não escolher um modelo "a priori" e só depois descobrir que
outra combinação era melhor. HistGradientBoosting é pulado para TF-IDF porque essa implementação do
sklearn não aceita entrada esparsa.

**Células [8]-[9] (tabela e heatmap):** organiza os resultados da grade num dataframe ordenado por
F1-macro e desenha um heatmap (representação × modelo). Isso é o que permite enxergar de uma vez só
que TF-IDF + Logistic Regression vence a tabela inteira, e por quanto — não só qual é o "número
campeão" isolado, mas o panorama comparativo completo.

**Célula [11] (escolher configurações para otimizar):** em vez de otimizar hiperparâmetros dos 5
modelos × todas as dimensões (caro e sem necessidade), o notebook pega apenas a melhor configuração
de cada um dos dois modelos mais promissores — XGBoost e Logistic Regression — e leva só essas duas
adiante. É uma forma de recortar o espaço de busca sem descartar candidatos de naturezas diferentes
(um linear, um de árvore) antes da hora.

**Célula [13] (otimização com Optuna):** aqui mora um cuidado importante contra vazamento: a busca de
hiperparâmetros usa uma fatia de validação recortada de dentro do próprio treino
(`train_test_split` com `stratify=y_train`), nunca o conjunto de teste. Os dois modelos usam os
mesmos índices de treino/validação, então a comparação entre eles fica em pé de igualdade. Cada
`objective` treina o modelo, mede F1-macro na validação e o Optuna (`TPESampler` com seed fixa, 25
tentativas) busca o conjunto de hiperparâmetros que maximiza essa métrica.

**Célula [15] (validação cruzada sem vazamento):** essa célula existe para corrigir um problema
metodológico real de uma versão anterior. Se a representação (TF-IDF ou UMAP) fosse ajustada uma
única vez no treino completo e depois só reaproveitada em cada fold do `cross_validate`, cada fold de
validação estaria "contaminado" — a representação já teria visto, no ajuste, dados que deveriam estar
de fora naquele fold. A correção é envolver representação + modelo num `sklearn.pipeline.Pipeline` e
passar dados **crus** (texto para o TF-IDF, embeddings densos para o UMAP) para o `cross_validate`.
Assim, a cada fold, a representação é reajustada do zero só com os dados de treino daquele fold
específico, e a estimativa de estabilidade entre folds fica confiável de verdade, não só
metodologicamente "documentada como simplificação".

**Célula [17] (avaliação final no teste):** treina os dois finalistas otimizados no treino completo e
avalia os dois no teste (que nenhum dos dois viu antes desse momento). Reporta F1-macro, F1
ponderado, ROC-AUC e PR-AUC macro para os dois, com uma nota explicando por que o ROC-AUC é pouco
informativo aqui (ele satura perto de 1 nesse problema, porque as classes já são bem separáveis — o
PR-AUC é mais sensível ao desbalanceamento residual entre elas). O modelo com maior F1-macro vira o
vencedor oficial e segue para as análises seguintes e para o notebook 04.

**Célula [19] (teste de significância):** a diferença de F1-macro entre os dois finalistas é pequena
— pequena o bastante para perguntar se não é só ruído de amostragem do conjunto de teste específico
que caiu no split. `comparar_modelos_bootstrap` reamostra os índices de teste 3.000 vezes (com
reposição), recalcula a diferença de F1-macro em cada reamostragem, e usa a distribuição resultante
para montar um intervalo de confiança de 95% e um p-valor. Isso transforma "a Logistic Regression deu
um número maior" em "a vantagem da Logistic Regression é estatisticamente significativa", uma
afirmação bem mais forte e defensável.

**Célula [21] (interpretabilidade):** se o vencedor for a Logistic Regression (que foi o caso), extrai
os coeficientes do modelo por classe, usando o mesmo vocabulário TF-IDF que treinou o modelo. Cada
coeficiente positivo indica o quanto a presença daquela palavra empurra a previsão a favor daquela
categoria — não é uma relação causal, é literalmente o peso que o modelo linear aprendeu. Essa célula
existe para não deixar a hipótese "o TF-IDF vence porque as categorias têm vocabulário característico"
como uma suposição — ela vira um fato verificável, palavra por palavra.

**Células [23]-[24] (análise de erros):** monta uma tabela só com as previsões erradas, mostra uma
amostra e depois agrupa por par (categoria real, categoria prevista) para ver quais confusões são
mais frequentes. Essa é a análise que aponta `Household` como a categoria mais problemática — não por
achismo, mas porque ela aparece repetidamente nos pares de maior contagem de erro, tanto como origem
quanto como destino da confusão.

---

## 04_modelo_final_conclusao — qual é o resultado final e o que aprendemos?

Esse notebook não testa mais nada novo em termos de modelagem — ele fixa a configuração vencedora
decidida no notebook 03, retreina e reavalia essa configuração isoladamente, e depois traduz os
resultados técnicos em conclusões e decisões operacionais.

**Célula [1] (imports e dados):** repete a preparação de dados dos notebooks anteriores (carregar,
limpar, recarregar o split). Como esse notebook só treina *um* modelo (o vencedor), ele não precisa
de toda a maquinaria de comparação do notebook 03.

**Célula [3] (configuração final):** registra explicitamente, num dicionário, a representação
(TF-IDF, 5000 dimensões), o modelo (Logistic Regression) e os hiperparâmetros vencedores decididos no
notebook 03. Deixar isso hard-coded e visível no topo do notebook, em vez de recalcular ou reotimizar
aqui, é proposital: esse notebook é sobre consolidar uma decisão já tomada, não sobre tomar uma nova.

**Célula [5] (treinar o modelo final):** ajusta o `TfidfVectorizer` só no treino, transforma o teste,
e treina a Logistic Regression com os hiperparâmetros da célula 3. É essencialmente um retreino limpo
e isolado da configuração vencedora, sem nenhum código de comparação misturado junto.

**Célula [7] (avaliação no teste):** calcula accuracy, F1-macro, F1 ponderado, ROC-AUC e PR-AUC macro
para o modelo final, e imprime o `classification_report` completo por categoria (precisão, recall e
F1 individuais). Esse é o número "oficial" do case, o que vai para a conclusão.

**Célula [9] (matriz de confusão final):** a versão visual da mesma avaliação — mostra não só quantos
acertos e erros no total, mas exatamente que par de categorias está sendo confundido, com a
intensidade de cor proporcional à quantidade.

**Célula [11] (interpretabilidade):** repete a extração de coeficientes feita no notebook 03, mas
agora sobre o modelo final treinado aqui (não é redundante: garante que a interpretação bate com o
modelo que realmente vai para a conclusão, não com uma versão anterior). O comentário no final do
código já assume a conclusão como fato verificado: as palavras fazem sentido de negócio nas 4
categorias, então a hipótese do vocabulário característico se sustenta.

**Célula [13] (principais erros):** mede a taxa de erro geral no teste e lista os pares de categoria
mais confundidos — mesma lógica das células 23-24 do notebook 03, mas aplicada ao modelo final
isoladamente.

**Célula [15] (métrica de negócio — cobertura por confiança):** essa é a célula que conecta ciência
de dados a decisão de produto. A probabilidade máxima que o modelo atribui a cada previsão
(`probas_max`) vira a base de uma regra simples: "acima de um threshold de confiança, aceita a
previsão automática; abaixo disso, manda para revisão manual". `cobertura_por_confianca` calcula, para
vários thresholds, que fração do catálogo seria autoclassificada e qual a accuracy só nesse
subconjunto. O ponto central é que aumentar o threshold sempre melhora a accuracy dos
autoclassificados (porque você só aceita os casos que o modelo está mais confiante) às custas de
autoclassificar menos coisa — e essa troca é uma decisão de negócio, não uma decisão estatística.

**Célula [17] (plano de monitoramento):** não roda modelo nenhum — organiza, a partir do que já foi
calculado na célula 15, uma tabela de indicadores para acompanhar se este modelo fosse de fato para
produção (distribuição de confiança, cobertura de autoclassificação, distribuição de classes
previstas, taxa de correção manual). A lógica por trás é que um modelo de TF-IDF tem vocabulário fixo,
aprendido no treino — produtos novos, marcas novas ou gírias que não existiam ali simplesmente não têm
representação, e isso é um risco real de deriva (drift) que vale a pena vigiar depois do deploy.

**Célula [18], markdown — Conclusão:** fecha o notebook amarrando tudo: o modelo funciona (compara
contra o baseline), a escolha do vencedor é estatisticamente defensável (não só o maior número), a
melhor representação foi a mais simples (TF-IDF, não os embeddings), `Household` é a categoria mais
difícil, e lista as limitações reais do trabalho (todas medidas ao longo dos notebooks, não citadas de
graça) junto com os próximos passos. Esse conteúdo foi movido para o `README.md` do projeto, com os
números conferidos contra a execução mais recente dos notebooks — por isso a célula pode ser removida
daqui sem perder a informação.
