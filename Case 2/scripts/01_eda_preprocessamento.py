# %% [markdown]
# # 01. EDA e Pré-processamento
#
# ## Objetivo
#
# Antes de aplicar qualquer técnica de Machine Learning, é preciso entender bem o problema e conhecer os dados.
#
# ## Contexto do problema
#
# O objetivo aqui é prever a categoria de um produto a partir da sua descrição em texto. A variável que queremos prever, o target, é a coluna `categoria`, que tem mais de duas classes possíveis. Para fazer essa previsão contamos apenas com o texto da descrição do produto, na coluna `texto`.

# %%
import sys
sys.path.append('./src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from preprocessing import carregar_e_limpar, obter_ou_criar_split, medir_duplicatas_semanticas

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 2. Carregamento

# %%
df_bruto = pd.read_csv('data/raw/dataset_ecommerce.csv')
print(f"Shape original: {df_bruto.shape}")
df_bruto.head()

# %% [markdown]
# ## 3. Qualidade dos dados

# %%
df_bruto.info()

# %%
print("Valores nulos por coluna:")
print(df_bruto.isna().sum())
print(f"\nLinhas totalmente duplicadas: {df_bruto.duplicated().sum()}")
print(f"Descrições de texto duplicadas: {df_bruto['texto'].duplicated().sum()}")
print(f"Textos vazios (string vazia após strip): {(df_bruto['texto'].astype(str).str.strip() == '').sum()}")
print(f"\nCategorias únicas antes da limpeza: {sorted(df_bruto['categoria'].dropna().unique())}")

# %% [markdown]
# ## 4. Limpeza
#
# A limpeza remove os registros com valores nulos, padroniza o nome da categoria `Clothing & Accessories` e tira espaços em branco extras. Ela também remove duplicatas exatas de texto e categoria, pelo motivo explicado logo abaixo.

# %%
df_sem_dedup = carregar_e_limpar('data/raw/dataset_ecommerce.csv', remover_duplicatas=False)
df = carregar_e_limpar('data/raw/dataset_ecommerce.csv')

n_nulos_removidos = df_bruto.shape[0] - df_sem_dedup.shape[0]
n_duplicatas_removidas = df_sem_dedup.shape[0] - df.shape[0]

print(f"Shape original: {df_bruto.shape}")
print(f"Após remover nulos: {df_sem_dedup.shape} (-{n_nulos_removidos} linha(s) com valores nulos)")
print(f"Shape após limpeza completa: {df.shape} (-{n_duplicatas_removidas} linha(s) duplicada(s) de texto+categoria)")
print(f"Categorias após padronização: {sorted(df['categoria'].unique())}")
df.head()

# %% [markdown]
# ### Removendo duplicatas
#
# As duplicatas encontradas na seção anterior são linhas com texto e categoria idênticos, provavelmente o mesmo produto reaparecendo no catálogo (revenda, variações de cor ou tamanho com a mesma descrição, etc.). O risco é o `train_test_split` da seção 8 colocar cópias do mesmo texto dos dois lados, uma no treino e outra no teste. Nesse caso o modelo não estaria sendo avaliado em texto novo, estaria só reconhecendo algo que já viu, o que infla as métricas sem representar a capacidade real de generalização.
#
# Para medir o tamanho do problema, refazemos a mesma divisão treino/teste da seção 8, mas sobre os dados sem deduplicar, e contamos quantas linhas do teste têm uma cópia idêntica (mesmo texto e mesma categoria) no treino.

# %%
from sklearn.model_selection import train_test_split

idx_sem_dedup = np.arange(len(df_sem_dedup))
y_sem_dedup = LabelEncoder().fit_transform(df_sem_dedup['categoria'])
idx_tr_sd, idx_te_sd = train_test_split(
    idx_sem_dedup, test_size=0.2, random_state=42, stratify=y_sem_dedup
)

chave = df_sem_dedup['texto'] + '||' + df_sem_dedup['categoria']
em_treino = set(chave.iloc[idx_tr_sd])
teste_com_gemeo_no_treino = chave.iloc[idx_te_sd].isin(em_treino).sum()

print(f"Sem remover duplicatas, {teste_com_gemeo_no_treino} das {len(idx_te_sd)} linhas de teste "
      f"({teste_com_gemeo_no_treino / len(idx_te_sd) * 100:.1f}%) teriam uma cópia idêntica no treino.")
print("Por isso a limpeza deste projeto remove as duplicatas antes de qualquer split.")

# %% [markdown]
# ## 5. Distribuição das categorias

# %%
contagem = df['categoria'].value_counts()
percentual = (contagem / len(df) * 100).round(1)

resumo_categorias = pd.DataFrame({'quantidade': contagem, 'percentual (%)': percentual})
display(resumo_categorias)

plt.figure(figsize=(8, 5))
sns.barplot(x=contagem.index, y=contagem.values, order=contagem.index)
plt.title('Distribuição das categorias')
plt.ylabel('Quantidade')
plt.xlabel('Categoria')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('figs/01_distribuicao_categorias.png', dpi=100)
plt.show()

razao_desbalanceamento = contagem.max() / contagem.min()
print(f"Razão entre a maior e a menor classe: {razao_desbalanceamento:.1f}x")

# %% [markdown]
# ## 6. Análise do texto

# %%
tamanho_texto = df['texto'].str.len()
n_palavras = df['texto'].str.split().str.len()

display(pd.DataFrame({'tamanho_texto': tamanho_texto, 'n_palavras': n_palavras}).describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(tamanho_texto, bins=50, ax=axes[0])
axes[0].set_title('Distribuição do tamanho do texto (caracteres)')
sns.histplot(n_palavras, bins=50, ax=axes[1])
axes[1].set_title('Distribuição do número de palavras')
plt.tight_layout()
plt.savefig('figs/01_distribuicao_tamanho_texto.png', dpi=100)
plt.show()

print("Texto mais curto:", df.loc[tamanho_texto.idxmin(), 'texto'])
print("\nTexto mais longo (300 primeiros caracteres):", df.loc[tamanho_texto.idxmax(), 'texto'][:300])

# %% [markdown]
# ### 6.1 Tamanho do texto por categoria
#
# Vale checar se alguma categoria tem descrições sistematicamente mais longas ou mais curtas. Isso ajuda a entender de onde pode vir parte do sinal preditivo, além do vocabulário em si.

# %%
df_tmp = df.copy()
df_tmp['n_palavras'] = n_palavras

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_tmp, x='categoria', y='n_palavras', order=sorted(df['categoria'].unique()))
plt.title('Número de palavras por categoria')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('figs/01_palavras_por_categoria.png', dpi=100)
plt.show()

display(df_tmp.groupby('categoria')['n_palavras'].describe()[['mean', '50%', 'std']].round(1))

# %% [markdown]
# ## 7. Exemplos de texto por categoria

# %%
print("Exemplo de texto por categoria:")
for cat in sorted(df['categoria'].unique()):
    exemplo = df[df['categoria'] == cat]['texto'].iloc[0]
    print(f"\n[{cat}] {exemplo[:200]}...")

# %% [markdown]
# ## 8. Encoding do target
#
# Como o modelo não entende texto diretamente, é preciso transformar a categoria em números. O `LabelEncoder` faz isso da forma mais simples: atribui um único número inteiro para cada categoria (0, 1, 2, 3...).
#
# Existe também o one-hot encoding, feito com `pd.get_dummies`, que cria uma coluna binária para cada categoria. Essa abordagem é mais indicada quando a variável categórica é usada como feature de entrada, por exemplo em um modelo linear, porque evita sugerir uma ordem ou escala entre categorias que na verdade não existe.
#
# Neste caso, `categoria` é o alvo, então faz sentido usar o `LabelEncoder`. Se ela fosse usada como feature, o que não acontece aqui já que as features vêm dos embeddings/TF-IDF do texto, aí sim o `pd.get_dummies` seria a escolha certa.

# %%
le = LabelEncoder()
y = le.fit_transform(df['categoria'])

print("Mapeamento LabelEncoder:")
print(dict(zip(le.classes_, le.transform(le.classes_))))
print(f"\nPrimeiros 5 valores de y: {y[:5]}")

# %% [markdown]
# ## 9. Divisão treino/teste (persistida)
#
# O split é calculado aqui e salvo em `data/processed/split.npz`. Os notebooks seguintes (02, 03, 04) carregam esse mesmo arquivo em vez de recalcular `train_test_split` de forma independente. Isso garante, por construção, que todos usam exatamente o mesmo treino/teste.

# %%
idx_train, idx_test = obter_ou_criar_split(df, y, caminho_split='data/processed/split.npz', test_size=0.2, random_state=42)
y_train, y_test = y[idx_train], y[idx_test]

print(f"Treino: {len(idx_train)} registros | Teste: {len(idx_test)} registros")
print("Distribuição treino (%):", np.round(np.bincount(y_train) / len(y_train) * 100, 1))
print("Distribuição teste (%): ", np.round(np.bincount(y_test) / len(y_test) * 100, 1))

# %% [markdown]
# ## 10. Risco de duplicatas semânticas (quase-idênticas, não exatas)
#
# A deduplicação da seção 4 remove textos idênticos, mas descrições quase iguais (uma palavra a mais, pontuação diferente, uma variação de tamanho/cor no fim do texto) não são pegas por ela e podem continuar espalhadas entre treino e teste. Isso vazaria informação de forma parecida com a duplicata exata, só que de forma parcial.
#
# Medimos isso com similaridade de cosseno sobre TF-IDF: para cada texto do teste, olhamos a maior similaridade com qualquer texto do treino. Um limiar de 0,9 é um corte conservador (bem mais exigente que "parecido"; é praticamente "quase a mesma frase").

# %%
resultado_dup_semantica = medir_duplicatas_semanticas(df, idx_train, idx_test, limiar=0.9)
print(f"Com limiar de similaridade >= {resultado_dup_semantica['limiar']}:")
print(f"  {resultado_dup_semantica['n_com_quase_duplicata']} de {resultado_dup_semantica['n_teste']} "
      f"linhas de teste ({resultado_dup_semantica['percentual']}%) têm uma quase-duplicata no treino.")
print(f"  Similaridade média (do vizinho mais próximo no treino, por linha de teste): "
      f"{resultado_dup_semantica['similaridade_media_maxima']}")

plt.figure(figsize=(8, 4))
sns.histplot(resultado_dup_semantica['similaridades_max'], bins=50)
plt.axvline(0.9, color='red', linestyle='--', label='limiar = 0.9')
plt.title('Maior similaridade de cosseno de cada texto de teste com o treino (TF-IDF)')
plt.xlabel('similaridade de cosseno máxima')
plt.legend()
plt.tight_layout()
plt.savefig('figs/01_duplicatas_semanticas.png', dpi=100)
plt.show()

# %% [markdown]
# ## Resumo

# %%
print("=" * 60)
print("RESUMO: EDA e Pré-processamento")
print("=" * 60)
print(f"Registros após limpeza: {df.shape[0]}")
print(f"Categorias: {df['categoria'].nunique()} -> {sorted(df['categoria'].unique())}")
print(f"Desbalanceamento (maior/menor classe): {razao_desbalanceamento:.1f}x")
print(f"Tamanho médio do texto: {tamanho_texto.mean():.0f} caracteres / {n_palavras.mean():.0f} palavras")
print(f"Split: {len(idx_train)} treino / {len(idx_test)} teste (stratify=y, persistido em data/processed/split.npz)")
print(f"Duplicatas exatas evitadas pela dedup: {teste_com_gemeo_no_treino} de {len(idx_te_sd)} "
      f"({teste_com_gemeo_no_treino / len(idx_te_sd) * 100:.1f}%) do teste, se não tivéssemos deduplicado")
print(f"Quase-duplicatas residuais (limiar 0.9) no split atual: "
      f"{resultado_dup_semantica['n_com_quase_duplicata']} de {resultado_dup_semantica['n_teste']} "
      f"({resultado_dup_semantica['percentual']}%)")

# %% [markdown]
# ## Conclusão da etapa
#
# Os dados têm 4 categorias com um desbalanceamento moderado, como mostra a razão calculada acima. Não é um caso extremo, mas já é suficiente para justificar o uso de métricas além da accuracy nas próximas etapas, como balanced accuracy e F1 macro. As descrições de texto também variam bastante de tamanho, o que reforça a necessidade de uma representação vetorial, os embeddings ou TF-IDF, capaz de capturar o significado do texto independentemente do seu tamanho.
#
# A checagem de duplicatas semânticas (seção 10) mostra que, além das duplicatas exatas já removidas, ainda resta um risco residual de vazamento parcial por descrições quase idênticas. Vale considerar uma deduplicação por similaridade, não só exata, em uma próxima iteração.
