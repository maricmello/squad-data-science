# Auditoria Técnica — Case 2: Classificação de Produtos de E-commerce por Texto

**Revisor:** Senior Data Scientist (revisão técnica externa)
**Escopo revisado:** `README.md`, `src/preprocessing.py`, `src/avaliacao.py`, notebooks `01` a `04`, material de apoio (`pdf/classification.pdf`)
**Contexto identificado:** este é um exercício de um treinamento em Ciência de Dados (pasta "Treinamento DS"), usando um dataset público de classificação de texto de e-commerce (Books / Clothing_Accessories / Electronics / Household). Não há stakeholder real nem briefing de negócio formal anexado — o `classification.pdf` é material teórico genérico do curso (leakage, métricas, threshold, calibração), não um brief específico deste case. Essa constatação importa para calibrar a seção de "problema de negócio": o projeto é avaliado como um exercício técnico, mas o rigor de conexão com negócio ainda é um critério válido, porque é isso que separa um projeto técnico de um projeto de dados que gera valor.

---

## 1. Problema de negócio

O problema está definido de forma clara e correta no nível técnico: prever `categoria` (4 classes) a partir de `texto`. O momento da previsão é implícito, mas razoável de inferir — no cadastro/ingestão de um produto no catálogo, antes de qualquer outra informação estruturada existir. Não há dimensão temporal no dataset (não existe timestamp), então a pergunta "existe vazamento temporal" não se aplica aqui — ponto correto que o projeto acerta por omissão (não precisa tratar o que não existe).

O que falta é a ponte entre a métrica de ML e uma métrica de negócio. O texto nunca diz quem usaria essa previsão, qual decisão operacional ela dispara (auto-tagging de catálogo? correção de cadastro incorreto? insumo para busca/recomendação?) nem qual erro custa mais caro. Isso importa porque as 4 categorias não são equivalentes em termos de impacto: confundir `Electronics` com `Household` pode ter consequência de busca/logística diferente de confundir `Books` com `Household`. O projeto já calcula `probas_max` (probabilidade da classe prevista) no notebook 04, mas não converte isso em uma proposta de negócio óbvia — por exemplo, "X% do catálogo pode ser autoclassificado com confiança acima de 95%, o resto vai para revisão manual". Essa é a melhoria de maior alavancagem que falta no projeto inteiro: é barata de fazer (os dados já estão calculados) e é exatamente o tipo de ponte que separa uma entrega técnica de uma entrega de negócio.

Não há premissas de negócio inventadas ou escondidas — nesse sentido o projeto é honesto, só é raso na conexão de valor.

**Nota: 6,0/10.**

---

## 2. Dados

Este é um dos pontos mais fortes do projeto. A origem dos dados é clara (CSV com `texto` e `categoria`, embeddings pré-computados). O tratamento de qualidade é bom: 1 nulo removido, categorias padronizadas (`Clothing & Accessories` → `Clothing_Accessories`), espaços em branco tratados.

O achado de maior qualidade do projeto inteiro está aqui: a decisão de remover duplicatas exatas de (texto, categoria) **antes do split** não foi tomada por "boa prática genérica" — foi *medida*. O notebook 01 refaz o split sem deduplicar e mostra que 63,6% do conjunto de teste teria uma cópia idêntica no treino, o que inflaria artificialmente qualquer métrica de avaliação. Isso é exatamente o tipo de verificação empírica de leakage que a maioria dos projetos pula, e aqui foi feita corretamente, com número na mão.

Dois pontos, porém, pesam contra a nota máxima:

1. **A verificação de alinhamento entre `embeddings_texto.npy` e o dataframe deduplicado é fraca.** O `assert` no notebook 02 (`assert embeddings.shape[0] == len(df)`) é feito *depois* de indexar `embeddings_completos[df.index.values]` — ou seja, ele sempre vai passar por construção, mesmo que o alinhamento estivesse errado (por exemplo, se `embeddings_texto.npy` tivesse sido gerado em outra ordem que não a assumida). Não existe nenhuma checagem independente disso (nem comparar `embeddings_completos.shape[0]` contra o tamanho esperado antes de indexar, nem re-gerar o embedding de uma amostra de textos e comparar similaridade com o valor carregado). Se essa suposição de ordem estiver errada, X e y ficam desalinhados silenciosamente — um erro grave e, pior, indetectável pelas métricas (o modelo aprenderia algo, só que errado).
2. Não há comparação explícita da distribuição de categorias antes/depois da deduplicação — não é grave (o "depois" está bem reportado), mas seria natural verificar se a deduplicação afeta desproporcionalmente alguma classe (ex.: se `Household` tinha mais produtos "clonados" no catálogo original).

Outliers de texto (textos de 4 caracteres a >50 mil caracteres) foram identificados na EDA, mas não houve decisão explícita sobre eles (nem remoção, nem tratamento, nem justificativa de mantê-los como estão) — fica implícito que ficam como estão, o que é uma escolha aceitável, mas deveria ser dita, não presumida.

**Nota: 8,0/10.**

---

## 3. EDA

A EDA cobre o essencial e é usada para decisão, não é decorativa: a análise de duplicatas leva direto à decisão de deduplicar (correto: EDA → decisão, não EDA → gráfico). A distribuição de classes é usada corretamente para justificar métricas além de accuracy. A análise de tamanho de texto justifica a escolha de uma representação vetorial.

O que falta para uma EDA de nível sênior em problema de texto:

- Nenhuma análise de vocabulário por categoria (palavras/n-gramas mais frequentes por classe, mesmo que superficial) — seria a EDA mais natural possível dado que a conclusão final do projeto ("as categorias têm vocabulário característico") é uma *hipótese pós-hoc* levantada só no notebook 04, quando poderia ter sido *demonstrada com dados* já na EDA (ex.: top TF-IDF terms por classe).
- Não há análise de relação entre tamanho do texto e categoria (é plausível que `Books` tenha sinopses mais longas que `Household`, por exemplo), que ajudaria a entender de onde vem o sinal preditivo.
- A conclusão da EDA ("desbalanceamento moderado, 2x") está correta e é usada de forma consistente depois — sem erro de interpretação aqui.

**Nota: 7,0/10.**

---

## 4. Preparação e Feature Engineering

O pipeline de features é simples (o texto vira embedding, PCA/UMAP ou TF-IDF) e isso é uma vantagem, não um defeito — não há engenharia de features tabular arbitrária sujeita a leakage sutil. O `fit` das reduções de dimensionalidade (PCA, UMAP, TF-IDF) é feito exclusivamente no treino, e o teste passa só por `transform`, em todos os notebooks. Isso está correto e é reforçado por comentários explícitos no código, não é um acidente.

A geração dos embeddings de frase acontece **antes** do split, sobre a base inteira. Isso, isoladamente, **não é vazamento**: o modelo de embeddings é pré-treinado (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), cada vetor é função só do próprio texto, não usa nenhuma informação de outras amostras nem do rótulo. É diferente de, por exemplo, calcular uma estatística agregada (média, frequência) usando toda a base — isso sim vazaria. O projeto acerta essa distinção mesmo sem discuti-la explicitamente; vale registrar que seria bom o README dizer isso de forma explícita, porque a pergunta "gerar embeddings antes do split vaza dado?" é exatamente o tipo de pergunta que um entrevistador júnior faria e a resposta certa (não vaza, porque é determinístico por linha) não está escrita em lugar nenhum do projeto.

Ponto de atenção real: a escolha de um modelo de embedding **multilíngue** para descrições que parecem ser majoritariamente em inglês não é justificada em nenhum momento. Não compromete o resultado final (TF-IDF venceu de qualquer forma), mas é uma decisão de projeto sem base — um modelo monolíngue em inglês tende a representar melhor esse tipo de texto.

**Nota: 7,5/10.**

---

## 5. Split e Validação

O split treino/teste (80/20, estratificado, `random_state=42`) é adequado ao problema — não há dimensão temporal nem entidades repetidas relevantes (produto = linha, já deduplicado). O conjunto de teste é preservado corretamente: não é tocado durante a seleção de features, nem durante a otimização de hiperparâmetros (o Optuna usa uma fatia de validação recortada *dentro* do treino, nunca o teste). Isso é o comportamento correto e está bem documentado no próprio notebook.

Dois problemas reais aqui, um deles mais sério que o outro:

**(a) A validação cruzada do notebook 03 tem vazamento técnico real, apesar de documentado.** O `cross_validate` é chamado sobre `X_tr_xgb`/`X_tr_log`, que já são o TF-IDF/UMAP ajustados uma única vez no treino completo — não há um `Pipeline` que refaça o `fit` da representação a cada fold. Isso significa que, em cada fold do 5-fold CV, o vocabulário/IDF do TF-IDF (e a projeção UMAP) já "viu" os textos que estão no fold de validação daquela rodada, porque eles fizeram parte do ajuste original no treino completo. O README chama isso de "simplificação didática", o que é uma forma honesta de descrever o problema, mas não deixa de ser um vazamento técnico dentro do CV. A consequência prática: os desvios-padrão reportados (±0,002–0,003) são **artificialmente otimistas**, e a conclusão de que "os dois modelos são igualmente estáveis" não está totalmente sustentada pela metodologia usada para chegar nela. Importante: isso **não invalida a métrica final de teste** (F1-macro = 0,949), que foi calculada com o TF-IDF ajustado só no treino e testado em dados nunca vistos — essa parte está limpa. O problema é isolado à interpretação de estabilidade via CV.
 *Correção concreta:* encapsular vetorizador + modelo em um `sklearn.pipeline.Pipeline` e passar isso para `cross_validate`, de forma que o `fit` do TF-IDF/UMAP aconteça a cada fold, só com os dados de treino daquele fold.

**(b) O split é recalculado do zero em cada notebook, em vez de salvo uma única vez.** Os notebooks 01, 02, 03 e 04 chamam `train_test_split(idx, y, test_size=0.2, random_state=42, stratify=y)` de forma independente, contando com o fato de que mesma semente + mesma ordem de dados produz o mesmo resultado. Na prática isso funciona (é determinístico), mas é um contrato implícito, não verificado em lugar nenhum: se `preprocessing.py` mudar sutilmente (ordem de operações, versão do pandas/sklearn), os quatro notebooks podem silenciosamente passar a usar splits diferentes entre si, sem que nada acuse o erro — e aí o notebook 03 estaria avaliando um modelo treinado com um split e o 04 com outro, sem ninguém perceber. *Correção concreta:* salvar `idx_train`/`idx_test` (ou um hash deles) uma vez no notebook 01 e carregar nos demais, em vez de recalcular.

O `random_state` é usado de forma consistente na maior parte do pipeline (split, modelos, Optuna, StratifiedKFold), com a exceção documentada do UMAP (sem seed fixa, para permitir paralelismo) — isso quebra a reprodutibilidade exata das combinações UMAP entre execuções, mas como o vencedor final é TF-IDF, o impacto no resultado reportado é nulo. Ainda assim, é uma troca consciente de reprodutibilidade por velocidade que idealmente seria validada rodando 2–3 sementes e reportando a variância, em vez de só assumida como "não deveria importar".

**Nota: 6,5/10.**

---

## 6. Baselines

O baseline (`DummyClassifier` com estratégia `most_frequent`) é apropriado e foi usado corretamente: mesmas features, mesmo conjunto de teste, mesmas métricas dos modelos reais. O ganho reportado sobre o baseline (F1-macro de 0,138 para 0,949) é real e enorme — o modelo claramente supera uma solução trivial.

Falta apenas variar o tipo de baseline (por exemplo, um `DummyClassifier(strategy='stratified')`, que amostra aleatoriamente respeitando a distribuição de classes) para ilustrar melhor a diferença entre "não aprender nada" e "não ter nenhum viés de classe majoritária" — mas isso é um refinamento opcional, não uma lacuna grave, porque a baseline de classe majoritária já é a mais relevante para esse problema desbalanceado.

**Nota: 8,5/10.**

---

## 7. Modelagem

A comparação inicial usa 5 modelos (Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, XGBoost) contra 3 representações (PCA, UMAP, TF-IDF) em 3 dimensões cada — uma varredura ampla e razoável, cobrindo modelos lineares e não lineares, bagging e boosting. A exclusão de HistGradientBoosting para TF-IDF (por não aceitar entrada esparsa nessa versão do scikit-learn) é tecnicamente correta e foi comunicada, não escondida.

A decisão de otimizar com Optuna **os dois melhores candidatos** (o que já vinha na frente sem tuning — TF-IDF + Logistic — e o que tinha sido escolhido a priori pelo projeto — XGBoost), em vez de só o modelo "escolhido antes de rodar a comparação", é a decisão mais madura do notebook 03. Evita o viés clássico de "escolhi um modelo antes de ver os dados, e agora só otimizo ele para justificar a escolha". Isso é comparação justa: mesmos dados de treino/validação internos, mesmo orçamento de 25 trials, mesmo conjunto de teste final.

O ponto fraco real: apenas 2 dos 5 modelos testados foram tunados. Random Forest, Extra Trees e HistGradientBoosting ficaram só com hiperparâmetros default em toda a comparação. Isso é declarado como limitação no notebook 04, o que é honesto, mas enfraquece tecnicamente a afirmação "TF-IDF + Logistic é a melhor combinação testada": é a melhor **entre as configurações testadas com esse nível de esforço**, não necessariamente a melhor possível — Random Forest ou Extra Trees tunados poderiam, em tese, mudar o ranking. Não chamo isso de erro grave porque foi transparente e a decisão de focar nos dois finalistas foi justificada por tempo/escopo, mas a nota reflete que a comparação final não é 100% simétrica.

Não há sinais de overfitting severo: o desempenho no teste é consistente com a validação cruzada e com o F1 antes do tuning, o que é um bom sinal de que os modelos não estão apenas memorizando o treino.

**Nota: 7,5/10.**

---

## 8. Métricas

A escolha de F1-macro como métrica de decisão está bem justificada (dá peso igual às 4 classes, evitando que `Household`, a mais frequente, domine a escolha) e é coerente com o desbalanceamento moderado identificado na EDA. Accuracy nunca é usada isoladamente para decidir nada — sempre aparece ao lado de balanced accuracy, precision/recall/F1 macro e weighted. Isso é exatamente o comportamento esperado dado o desbalanceamento de 2x.

Dois pontos de crítica:

- **ROC-AUC (multi-classe, "ovr macro") é reportado, mas acrescenta pouca informação.** Em um problema com boa separação entre classes como este (0,991), o ROC-AUC tende a saturar perto de 1 e comunica menos do que a matriz de confusão e o F1 por classe, que já estão sendo reportados. Não é um erro de interpretação — o número não é usado de forma equivocada — mas é uma métrica redundante que poderia ter sido substituída por algo mais informativo, como PR-AUC por classe (mais sensível a desbalanceamento residual entre categorias) ou análise de calibração das probabilidades (tema que o próprio material teórico do treinamento, no PDF, trata como importante quando a probabilidade em si será usada para decisão — o que seria o caso se o projeto tivesse proposto revisão manual por threshold de confiança).
- **Não há teste de significância estatística entre os dois finalistas.** A diferença de F1-macro entre Logistic Regression (0,949) e XGBoost (0,938) é de 0,011 — pequena. O desvio-padrão do CV (±0,002–0,003, mesmo com a ressalva de leakage da seção 5) sugere que a diferença provavelmente é real, mas isso é uma inferência implícita, nunca testada formalmente (por exemplo, com bootstrap sobre o conjunto de teste ou teste de McNemar entre as duas matrizes de predição). Para uma decisão de "qual modelo vai para produção", vale a pena formalizar isso.

**Nota: 7,5/10.**

---

## 9. Interpretabilidade

Esta é a lacuna mais clara do projeto. O rubric usado pelo próprio material de treinamento (seção de "importância de features e interpretabilidade") pede isso explicitamente, e não aparece em nenhum dos quatro notebooks.

O modelo vencedor é uma Logistic Regression sobre TF-IDF — o cenário mais fácil possível para interpretabilidade em todo o Machine Learning: bastaria extrair `modelo_final.coef_` e mapear os pesos mais altos/mais baixos de volta para o vocabulário do `TfidfVectorizer` (`tfidf.get_feature_names_out()`) para mostrar, por categoria, quais palavras mais pesam a favor e contra cada previsão. Isso não foi feito. É uma lacuna barata de corrigir (poucas linhas de código) e de alto valor: validaria (ou refutaria) a hipótese levantada no fechamento do notebook 04 ("as categorias têm vocabulário característico, por exemplo 'capa' e 'página' para Books") — hoje essa frase é uma suposição plausível, não uma conclusão demonstrada com dados.

A análise de erros feita (matriz de confusão, pares mais confundidos, exemplos de acertos/erros com probabilidade) é boa e ajuda a entender **onde** o modelo erra, mas não **por quê** — não é interpretabilidade de modelo, é análise de resíduos. Os dois são complementares, mas só o segundo foi feito.

Não há nenhuma interpretação causal indevida no texto — os autores são cuidadosos em usar linguagem como "hipótese razoável" e "sugere", não "prova" ou "causa". Isso é positivo e evita um erro comum.

**Nota: 4,0/10.**

---

## 10. Testes de Robustez

Existe validação cruzada (mitigada pela ressalva de leakage da seção 5) e existe deduplicação exata testada empiricamente — isso já é mais robustez do que a maioria dos projetos comparáveis costuma ter.

O que falta:

- **Nenhuma avaliação por segmento além das 4 classes-alvo.** Não há quebra de desempenho por tamanho de texto (o próprio projeto identificou textos de 4 a mais de 50 mil caracteres na EDA — como o modelo se sai nos extremos, por exemplo em textos com poucas palavras como "Yes!"?), nem por outros cortes possíveis.
- **A deduplicação apenas exata deixa um risco residual não medido.** O notebook 04 reconhece que descrições quase idênticas (não idênticas) podem continuar espalhadas entre treino e teste, mas, ao contrário da deduplicação exata (que foi quantificada com o número de 63,6%), esse risco residual nunca foi medido — fica só como hipótese para trabalho futuro. Seria possível, com o próprio TF-IDF já calculado, medir quantos pares treino/teste têm similaridade de cosseno muito alta (>0,9, por exemplo) sem serem idênticos, para saber se esse risco é desprezível ou relevante.
- **Não há teste de sensibilidade a variação de seed** além da CV com leakage (que já mistura duas coisas: variação por fold e o próprio problema de vazamento do fit).

**Nota: 5,0/10.**

---

## 11. Conclusão

Este é o segundo ponto mais forte do projeto, depois da seção de dados. As conclusões do notebook 04 são cuidadosas: dizem exatamente o que foi testado, o que venceu e por quê, sem prometer mais do que os dados sustentam. A frase "dentro do que foi testado, sim" ao responder se este foi o melhor modelo possível é exatamente o tipo de hedge correto — reconhece que o espaço de busca foi limitado, sem fingir que a busca foi exaustiva.

A seção de limitações do notebook 04 é, na prática, uma lista de autoavaliação madura: cita o vazamento técnico do CV, cita os 3 modelos não tunados, cita a ausência de deduplicação semântica, cita o modelo de embedding único testado. Isso é raro de ver — a maioria dos projetos omite essas ressalvas ou as trata superficialmente. Aqui elas coincidem quase integralmente com os problemas reais que esta auditoria identificou de forma independente, o que é um sinal forte de maturidade técnica de quem escreveu.

O que falta é a tradução dessas conclusões em uma recomendação de negócio objetiva (ver seção 1) e um "próximo passo" mais concreto — a lista de "o que eu faria em uma próxima versão" é boa, mas é uma lista de melhorias técnicas, não uma recomendação de próximo passo priorizada (ex.: "antes de qualquer deploy, corrigir o CV com Pipeline e adicionar interpretabilidade; isso deveria levar N dias").

**Nota: 8,5/10.**

---

## 12. Qualidade do notebook / projeto

O código está bem organizado: lógica de limpeza e avaliação centralizada em `src/preprocessing.py` e `src/avaliacao.py`, reaproveitada de forma consistente pelos quatro notebooks, em vez de copiada e colada. Os nomes de variáveis são claros e em português consistente (`idx_train`, `melhor_modelo`, `resultados`). Há separação nítida entre EDA (01), representação (02), modelagem (03) e conclusão (04) — a estrutura de pastas e o README deixam essa divisão explícita.

A documentação é acima da média: cada decisão de projeto (por que deduplicar, por que não fixar seed no UMAP, por que otimizar os dois finalistas e não só o XGBoost) tem uma frase de justificativa em markdown, no código ou no README. Isso facilita muito a auditoria — inclusive esta.

Pontos a melhorar:
- O `assert` tautológico da seção 2 é o único ponto de código que eu classificaria como "parece uma verificação, mas não verifica nada" — vale corrigir porque é enganoso, não só incompleto.
- Não há testes automatizados (unitários) para `preprocessing.py` ou `avaliacao.py`, mesmo sendo funções pequenas e fáceis de testar (ex.: testar que `carregar_e_limpar` remove duplicatas corretamente com um CSV de exemplo). Para um projeto de treinamento isso é aceitável; para produção, seria exigível.
- Reprodutibilidade: rodar os notebooks do zero reproduz o resultado quase integralmente, com a exceção conhecida do UMAP.

**Nota: 8,5/10.**

---

## 13. Governança e Produção

Este é o segundo ponto mais fraco do projeto, e é esperado dado o escopo (exercício de treinamento, sem stakeholder real) — mas a pergunta foi feita explicitamente, então respondo sem suavizar.

Riscos não endereçados:
- **Deriva de vocabulário.** O modelo final depende de um `TfidfVectorizer` com vocabulário fixo de 5000 palavras aprendido no treino. Produtos novos, marcas novas ou gírias que não existiam no treino simplesmente não têm representação — o vocabulário do TF-IDF não se atualiza sozinho. Não há discussão sobre retraining periódico nem sobre como detectar essa deriva.
- **Categorias novas.** Se o catálogo introduzir uma 5ª categoria no futuro, o modelo não tem mecanismo para lidar com isso (vai forçar a previsão em uma das 4 classes existentes).
- **Monitoramento pós-deploy.** Não há proposta de quais indicadores acompanhar em produção. Os candidatos óbvios, dado o que já foi calculado no projeto, seriam: distribuição da probabilidade máxima prevista ao longo do tempo (queda sistemática sugere deriva de vocabulário), taxa de previsões de baixa confiança (abaixo de um threshold) precisando de revisão manual, e comparação periódica da distribuição de classes previstas contra a distribuição histórica.
- **Sem plano de fallback.** Não há proposta de "o que fazer quando o modelo erra" (ex.: fila de revisão humana para previsões de baixa confiança) — apesar dos dados para isso (`probas_max`) já estarem disponíveis no notebook 04.

Condições mínimas antes de um deploy real: corrigir o CV com `Pipeline` (seção 5), adicionar interpretabilidade do modelo vencedor (seção 9), definir e testar uma estratégia de threshold/revisão manual, e desenhar um plano de monitoramento de deriva de vocabulário.

**Nota: 4,0/10.**

---

# A) Nota Geral do Projeto

## 7,8 / 10

Este é um projeto tecnicamente sólido para o escopo de um exercício de treinamento, com um nível de rigor metodológico em prevenção de leakage (deduplicação medida empiricamente, fit/transform sempre separados, hiperparâmetros otimizados sem tocar o teste, comparação justa entre os dois finalistas) que está acima do que a maioria dos projetos comparáveis entrega. As conclusões são honestas e bem fundamentadas, e a seção de limitações do próprio autor coincide, em grande parte, com os problemas que esta auditoria encontrou de forma independente — isso é um sinal de maturidade real, não de sorte.

A nota não é mais alta por três motivos concretos: (1) a validação cruzada do notebook 03 tem um vazamento técnico real, mesmo que documentado como simplificação — isso compromete a interpretação de estabilidade, embora não a métrica final de teste; (2) não há nenhuma análise de interpretabilidade do modelo vencedor, uma lacuna barata de corrigir e que reduz a confiança na hipótese central da conclusão; (3) a conexão com valor de negócio e a prontidão para produção (monitoramento, deriva, fallback) são rasas, mesmo levando em conta que é um exercício de treinamento sem stakeholder real.

---

# B) Notas por categoria

| Categoria | Nota |
|---|---|
| Problema de negócio | 6,0 |
| Dados | 8,0 |
| EDA | 7,0 |
| Feature engineering | 7,5 |
| Validação | 6,5 |
| Modelagem | 7,5 |
| Métricas | 7,5 |
| Interpretabilidade | 4,0 |
| Robustez | 5,0 |
| Conclusão | 8,5 |
| Código/reprodutibilidade | 8,5 |
| Prontidão para produção | 4,0 |

---

# C) Problemas críticos

Nenhum problema deste projeto é crítico no sentido de "invalida o resultado principal reportado" — o F1-macro de 0,949 no teste foi calculado com um pipeline limpo (fit só no treino, teste nunca tocado antes da avaliação final). Mas há um problema que classifico como **crítico para a confiabilidade metodológica do processo de decisão**, não do número final:

1. **Vazamento técnico na validação cruzada (notebook 03, seção 6).** O TF-IDF/UMAP usado no `cross_validate` foi ajustado uma única vez no treino completo, não refeito por fold. Isso torna a conclusão "os dois modelos são igualmente estáveis entre folds" não totalmente confiável, mesmo com a ressalva documentada. Correção: `Pipeline` do sklearn dentro do `cross_validate`.

Os demais problemas relevantes são moderados, não críticos, e estão detalhados nas seções 1–13 acima. Resumindo os mais importantes:

2. Verificação de alinhamento embeddings↔dataframe é tautológica (não prova nada, mesmo que provavelmente esteja correta).
3. Ausência de interpretabilidade do modelo vencedor.
4. Split recalculado do zero em cada notebook, sem persistência/verificação de identidade entre eles.
5. Deduplicação apenas exata, com risco residual de duplicatas semânticas nunca medido.

---

# D) Pontos fortes

- Deduplicação de texto+categoria antes do split, com o risco de vazamento **medido empiricamente** (63,6% do teste teria cópia no treino sem essa etapa) — a melhor decisão metodológica do projeto.
- Separação `fit`/`transform` correta e consistente em todas as reduções de dimensionalidade e no TF-IDF.
- Baseline obrigatório (classe majoritária) usado como piso de comparação em todas as etapas.
- Hiperparâmetros otimizados via Optuna sem tocar o conjunto de teste, usando uma fatia de validação recortada do treino.
- Os dois melhores candidatos (não só o modelo escolhido a priori) foram otimizados e comparados em pé de igualdade — evita o viés de "escolher o modelo antes de ver os dados".
- Métricas múltiplas e apropriadas ao desbalanceamento (nunca só accuracy).
- Conclusões bem fundamentadas, sem overclaim, com hedges corretos ("dentro do que foi testado").
- Autoavaliação de limitações no notebook 04 madura e, em boa parte, coincidente com os achados desta auditoria independente.
- Código modular, organizado, com decisões documentadas em texto — facilita revisão e manutenção.

---

# E) Melhorias prioritárias

**Alta prioridade**
1. Corrigir o vazamento técnico da validação cruzada (notebook 03) com um `Pipeline` que refaça o `fit` da representação a cada fold.
2. Adicionar interpretabilidade do modelo vencedor: extrair `coef_` da Logistic Regression e mapear para o vocabulário do TF-IDF, por categoria — confirma ou refuta a hipótese central da conclusão ("vocabulário característico por categoria").
3. Substituir o `assert` tautológico do notebook 02 por uma verificação real do alinhamento entre `embeddings_texto.npy` e o dataframe deduplicado (checar o shape antes de indexar, ou comparar uma amostra recomputada).

**Média prioridade**
4. Persistir `idx_train`/`idx_test` uma única vez (notebook 01) e carregar nos demais notebooks, em vez de recalcular o split em cada um.
5. Medir o risco de duplicatas semânticas (não exatas) entre treino e teste, usando similaridade de cosseno sobre o próprio TF-IDF já calculado.
6. Formalizar a comparação entre os dois finalistas com um teste estatístico (bootstrap ou McNemar), já que a diferença de F1-macro é pequena (0,011).
7. Conectar a previsão a uma métrica de negócio simulada (ex.: % do catálogo autoclassificável com confiança acima de um threshold, liberando revisão manual só para o restante).
8. Rodar o UMAP com seed fixa (mesmo que mais lento) pelo menos uma vez, para confirmar que a comparação entre representações no notebook 03 não muda com a semente.

**Baixa prioridade**
9. Justificar a escolha do modelo de embedding multilíngue para um corpus majoritariamente em inglês, ou testar um modelo monolíngue.
10. Remover ou complementar o ROC-AUC com uma métrica mais informativa (PR-AUC por classe ou calibração), já que ele satura perto de 1 e comunica pouco a mais que o F1 já reportado.
11. Adicionar testes unitários simples para `preprocessing.py` e `avaliacao.py`.
12. Esboçar um plano mínimo de monitoramento pós-deploy (deriva de vocabulário, taxa de baixa confiança, distribuição de classes previstas ao longo do tempo).

---

# F) Avaliação como processo seletivo

**O projeto demonstra domínio de Data Science?** Sim, de forma consistente. O autor entende por que leakage acontece (não só que deve evitá-lo), sabe medir o impacto de uma decisão metodológica em vez de só declará-la (o teste de 63,6% de duplicatas é o melhor exemplo disso), e escreve conclusões hedged corretamente. Isso é mais raro do que dominar sklearn.

**O que impressionaria um entrevistador:**
- A quantificação do impacto da deduplicação antes de decidir remover — a maioria dos candidatos apenas afirma "removi duplicatas para evitar leakage" sem provar que o leakage existiria.
- Otimizar os dois melhores candidatos, não só o modelo escolhido a priori — mostra desapego à decisão inicial do projeto quando os dados discordam dela.
- A seção de limitações do notebook 04, que é honesta e específica, não genérica ("o modelo poderia melhorar com mais dados").

**O que levantaria questionamentos:**
- A ausência de interpretabilidade em um modelo linear, onde ela é trivial de obter — um entrevistador vai notar que essa foi a parte "fácil" que ficou de fora.
- O `assert` tautológico — um revisor de código sênior vai pegar isso rápido e perguntar "esse assert realmente testa alguma coisa?".
- A CV com vazamento técnico, mesmo documentada — vai gerar a pergunta "então por que vocês confiam no desvio-padrão reportado?".

**Perguntas técnicas prováveis:**
- "Por que gerar os embeddings antes do split não é vazamento de dados, já que normalmente dizemos para nunca tocar no teste antes de separar?"
- "Como você garantiria, de forma verificável, que os embeddings estão alinhados com as linhas certas do dataframe depois da deduplicação?"
- "Se o TF-IDF venceu, por que vocês continuaram testando embeddings e UMAP? O que isso te ensinou sobre o problema?"
- "Você disse que a validação cruzada tem uma simplificação didática. Quanto isso pode estar inflando (ou não) a estabilidade reportada? Como você mediria isso?"
- "Quais palavras a Logistic Regression está usando para decidir cada categoria? Faz sentido de negócio?"

**Nível do projeto:** pleno forte, com trechos de nível sênior (a quantificação do risco de leakage antes de agir, a comparação justa entre os dois finalistas, a autoavaliação de limitações) e trechos que ainda pedem amadurecimento típico de sênior (interpretabilidade não é opcional quando é trivial de obter; um assert que não testa nada é um hábito a corrigir cedo; a ponte com negócio não pode ficar totalmente implícita). Não classificaria como júnior — o nível de ceticismo metodológico demonstrado (testar antes de assumir) está bem acima do que se vê tipicamente em projetos júnior.

---

# G) Veredito final

**SIM, MAS COM RESSALVAS.**

Eu aprovaria este projeto como um projeto de Data Science tecnicamente confiável **no sentido de que o número final reportado (F1-macro = 0,949 no teste) é real e foi obtido com um pipeline sem vazamento entre treino e teste na etapa de avaliação final**. A metodologia de prevenção de leakage no que importa mais — o split final e a avaliação do modelo escolhido — está correta e é, em vários pontos, mais rigorosa do que a média.

As ressalvas que impedem uma aprovação sem qualificações:
1. A validação cruzada usada para argumentar estabilidade tem um vazamento técnico documentado, mas real — a conclusão de "estabilidade equivalente entre os dois finalistas" não deveria ser tomada como definitiva sem refazer esse teste com um `Pipeline` correto.
2. Não existe nenhuma interpretabilidade do modelo vencedor — a hipótese central da conclusão do projeto (vocabulário característico por categoria) nunca foi demonstrada com dados, só proposta como explicação plausível.
3. A prontidão para produção é insuficiente (esperado dado o escopo de treinamento, mas real): não há plano de monitoramento, deriva de vocabulário ou fallback para baixa confiança.

Nenhuma dessas três ressalvas é motivo para reprovar o projeto — são exatamente o tipo de lacuna que se espera encontrar e corrigir antes de um deploy real, e o próprio autor já havia sinalizado a maior parte delas honestamente. É por isso que a resposta é "sim, com ressalvas", e não "não": o projeto erra em pontos específicos e nomeáveis, não na postura metodológica geral.
