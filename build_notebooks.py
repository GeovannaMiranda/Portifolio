import nbformat as nbf

def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return n

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# =====================================================================================
# NOTEBOOK 1 - EDA
# =====================================================================================
cells1 = [
md("""# Inadimplência no Pix Parcelado — 1. Análise Exploratória

Esse notebook é um estudo que fiz pra praticar análise de dados aplicada a crédito,
usando como tema o **Pix parcelado** (um produto que virou bem comum nos bancos
digitais). Simulei uma base de compras parceladas e fui atrás de responder algumas
perguntas que fariam sentido pra um time de crédito/cobrança:

- Quem são os clientes que mais atrasam?
- O número de parcelas influencia na inadimplência?
- Cliente recorrente atrasa menos?

Os dados são **fictícios**, gerados por mim em `data/gerar_dados.py` (não é dado real
de ninguém).

Esse é o notebook 1 de 3:
1. **Análise exploratória** (esse aqui)
2. Estatística + teste A/B
3. Modelo simples de IA pra prever inadimplência
"""),

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("../data/pix_parcelado.csv")
df.head()
"""),

md("""## Dando uma primeira olhada nos dados"""),

code("""df.info()
"""),

code("""df.describe()
"""),

code("""# taxa de inadimplência geral, só pra ter um número de partida
taxa_geral = df["inadimplente"].mean()
print(f"Taxa de inadimplência geral: {taxa_geral:.1%}")
"""),

md("""## A quantidade de parcelas influencia na inadimplência?

Minha hipótese aqui é que quanto mais parcelado, maior a chance da pessoa atrasar
(faz sentido, o compromisso fica mais longo). Vamos ver se os dados confirmam isso.
"""),

code("""inad_por_parcela = df.groupby("qtd_parcelas")["inadimplente"].mean() * 100

fig, ax = plt.subplots(figsize=(7,4))
inad_por_parcela.plot(kind="bar", color="#FF7A00", ax=ax)
ax.set_ylabel("Taxa de inadimplência (%)")
ax.set_xlabel("Quantidade de parcelas")
ax.set_title("Inadimplência por número de parcelas")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../images/inadimplencia_parcelas.png", dpi=120)
plt.show()
"""),

md("""Dá pra ver que conforme aumenta o número de parcelas, a inadimplência também
tende a subir. Faz sentido: compras em mais vezes representam um comprometimento
maior no orçamento da pessoa ao longo do tempo."""),

md("""## Cliente que já usou o Pix parcelado antes atrasa menos?

Outra hipótese: quem já é cliente recorrente do produto tende a ser mais
organizado com os pagamentos (ou já foi "filtrado" naturalmente, já que quem
atrasa muito tende a parar de usar).
"""),

code("""tabela = df.groupby("ja_usou_pix_parcelado_antes")["inadimplente"].mean() * 100
tabela.index = ["Primeira vez", "Já usou antes"]

fig, ax = plt.subplots(figsize=(5,4))
tabela.plot(kind="bar", color=["#999999", "#0057FF"], ax=ax)
ax.set_ylabel("Taxa de inadimplência (%)")
ax.set_title("Inadimplência: primeira compra vs cliente recorrente")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../images/inadimplencia_recorrencia.png", dpi=120)
plt.show()

print(tabela)
"""),

md("""## E o valor da compra em relação à renda?

Faz sentido pensar que quanto maior o "peso" da compra no orçamento da pessoa,
maior a chance dela não conseguir pagar. Criei uma coluna simples de
comprometimento de renda pra checar isso.
"""),

code("""df["comprometimento_renda"] = df["valor_compra"] / df["renda_estimada"]

df["faixa_comprometimento"] = pd.cut(
    df["comprometimento_renda"],
    bins=[0, 0.05, 0.15, 10],
    labels=["baixo (<5%)", "médio (5-15%)", "alto (>15%)"]
)

tabela_comprometimento = df.groupby("faixa_comprometimento")["inadimplente"].mean() * 100

fig, ax = plt.subplots(figsize=(6,4))
tabela_comprometimento.plot(kind="bar", color="#B00020", ax=ax)
ax.set_ylabel("Taxa de inadimplência (%)")
ax.set_title("Inadimplência por comprometimento de renda na compra")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../images/inadimplencia_comprometimento.png", dpi=120)
plt.show()
"""),

md("""## O que deu pra ver nessa primeira análise

- Quanto mais parcelas, maior a inadimplência
- Cliente recorrente parece atrasar menos que cliente de primeira compra
- Quanto maior o peso da compra na renda da pessoa, maior o risco de atraso

No próximo notebook eu confirmo se essas diferenças são estatisticamente
significativas ou se podem ser só coincidência da amostra, e testo duas
estratégias diferentes de cobrança (teste A/B).
"""),
]

# =====================================================================================
# NOTEBOOK 2 - ESTATISTICA + TESTE A/B
# =====================================================================================
cells2 = [
md("""# Inadimplência no Pix Parcelado — 2. Estatística e Teste A/B

Continuando o estudo do notebook 1. Aqui eu quero confirmar com testes
estatísticos se as diferenças que vi nos gráficos são "de verdade" ou podem ser
só coincidência, e também simular um teste A/B: o banco (fictício) testou duas
formas de lembrar o cliente inadimplente de pagar, e quero ver qual funcionou
melhor pra recuperar esse cliente.
"""),

code("""import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/pix_parcelado.csv")
df.head()
"""),

md("""## A diferença entre "já usou antes" e "primeira vez" é real ou coincidência?

Pra não ficar só no achismo olhando gráfico, rodei um teste qui-quadrado, que
serve justamente pra ver se duas variáveis categóricas (nesse caso, "já usou
antes" e "ficou inadimplente") têm relação real ou se a diferença que vi pode
ser só acaso da amostra.
"""),

code("""tabela_contingencia = pd.crosstab(df["ja_usou_pix_parcelado_antes"], df["inadimplente"])
print(tabela_contingencia)

chi2, p, dof, esperado = stats.chi2_contingency(tabela_contingencia)
print(f"\\nqui-quadrado: {chi2:.2f}")
print(f"p-valor: {p:.4f}")

if p < 0.05:
    print("\\nComo o p-valor é menor que 0.05, dá pra dizer que existe sim uma relação"
          " entre já ter usado o produto antes e a chance de ficar inadimplente"
          " (não é só coincidência da amostra).")
else:
    print("\\nComo o p-valor é maior que 0.05, não dá pra afirmar com confiança que"
          " existe relação entre as duas coisas.")
"""),

md("""## Score de crédito x inadimplência

Queria ver também se o score de crédito (que é uma variável clássica em análise
de crédito) separa bem quem paga de quem atrasa, e se essa diferença é
estatisticamente significativa (teste t).
"""),

code("""fig, ax = plt.subplots(figsize=(7,4))
sns.kdeplot(data=df, x="score_credito", hue="inadimplente", fill=True, alpha=0.4,
            palette={0: "#0057FF", 1: "#B00020"}, ax=ax)
ax.set_title("Distribuição do score: quem pagou vs quem atrasou")
plt.tight_layout()
plt.savefig("../images/score_inadimplencia.png", dpi=120)
plt.show()
"""),

code("""score_pagou = df.loc[df["inadimplente"] == 0, "score_credito"]
score_atrasou = df.loc[df["inadimplente"] == 1, "score_credito"]

t, p = stats.ttest_ind(score_pagou, score_atrasou, equal_var=False)
print(f"média score (pagou): {score_pagou.mean():.0f}")
print(f"média score (atrasou): {score_atrasou.mean():.0f}")
print(f"p-valor: {p:.2e}")
"""),

md("""O gráfico já mostra visualmente que quem atrasa tende a ter um score mais baixo,
e o teste t confirma que essa diferença não é coincidência (p-valor bem menor que
0.05).
"""),

md("""## Teste A/B: qual lembrete de cobrança funciona melhor?

Entre os clientes que ficaram inadimplentes, metade recebeu um **lembrete
simples** (notificação padrão) e a outra metade recebeu um **lembrete com opção
de renegociar** a parcela direto no app. A métrica que importa aqui é: quantos
desses clientes voltaram a ficar em dia (recuperou)?

Pra comparar as duas proporções de forma estatística (e não só olhar o número
cru), usei um teste Z de proporções, que é basicamente o teste padrão pra
comparar taxa de conversão entre dois grupos — a mesma lógica usada em teste A/B
de produto/marketing.
"""),

code("""from statsmodels.stats.proportion import proportions_ztest, proportion_confint

inad = df[df["inadimplente"] == 1].copy()

grupo_a = inad[inad["estrategia_lembrete"] == "Lembrete simples"]
grupo_b = inad[inad["estrategia_lembrete"] == "Lembrete com opção de renegociar"]

recuperados = np.array([grupo_a["recuperou"].sum(), grupo_b["recuperou"].sum()])
totais = np.array([len(grupo_a), len(grupo_b)])
taxas = recuperados / totais

print(f"Grupo A (lembrete simples):        {recuperados[0]}/{totais[0]} = {taxas[0]:.1%}")
print(f"Grupo B (lembrete com renegociar):  {recuperados[1]}/{totais[1]} = {taxas[1]:.1%}")

z_stat, p_value = proportions_ztest(recuperados, totais)
print(f"\\nestatística Z: {z_stat:.2f}")
print(f"p-valor: {p_value:.4f}")

ci_a = proportion_confint(recuperados[0], totais[0], alpha=0.05, method="wilson")
ci_b = proportion_confint(recuperados[1], totais[1], alpha=0.05, method="wilson")
print(f"\\nintervalo de confiança 95% grupo A: [{ci_a[0]:.1%}, {ci_a[1]:.1%}]")
print(f"intervalo de confiança 95% grupo B: [{ci_b[0]:.1%}, {ci_b[1]:.1%}]")

if p_value < 0.05:
    print("\\n=> a diferença é estatisticamente significativa, então dá pra dizer que"
          " o lembrete com opção de renegociar realmente recupera mais gente, não é"
          " só sorte da amostra.")
else:
    print("\\n=> não deu pra confirmar estatisticamente que um lembrete é melhor que"
          " o outro com esse tamanho de amostra.")
"""),

code("""fig, ax = plt.subplots(figsize=(6,4))
labels = ["Lembrete simples", "Lembrete com\\nrenegociar"]
ax.bar(labels, taxas * 100, color=["#999999", "#FF7A00"])
ax.set_ylabel("Taxa de recuperação (%)")
ax.set_title("Teste A/B — qual lembrete recupera mais cliente")
for i, v in enumerate(taxas * 100):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("../images/teste_ab_recuperacao.png", dpi=120)
plt.show()
"""),

md("""## Conclusão desse notebook

- A diferença entre cliente recorrente e cliente novo é real (qui-quadrado deu
  significativo)
- Score de crédito realmente separa bem quem paga de quem atrasa (teste t
  significativo)
- No teste A/B, o lembrete com opção de renegociar parece recuperar mais cliente
  do que o lembrete simples, e a diferença é estatisticamente significativa

Se fosse um cenário real, o próximo passo seria rodar esse teste com mais tempo/
amostra antes de migrar 100% da base pro lembrete novo, e acompanhar se o
resultado se mantém em diferentes perfis de cliente.
"""),
]

# =====================================================================================
# NOTEBOOK 3 - MODELO IA
# =====================================================================================
cells3 = [
md("""# Inadimplência no Pix Parcelado — 3. Modelo simples de IA

Pra fechar o estudo, tentei montar um modelo básico de regressão logística (o
mais simples e mais usado em crédito) pra ver se dá pra prever, com as
informações que tenho, quem tem mais chance de atrasar. Não usei nada muito
avançado de propósito — a ideia aqui é mostrar que entendo a lógica de montar e
avaliar um modelo, não ficar caçando o modelo mais complexo possível.
"""),

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/pix_parcelado.csv")
df.head()
"""),

md("""## Preparando os dados

Só uso informação que eu teria **antes** de saber se o cliente vai atrasar ou
não (não faz sentido usar coisa como "recuperou", que só existe depois que a
pessoa já atrasou — isso seria trapaça/vazamento de dado).
"""),

code("""features = ["idade", "renda_estimada", "valor_compra", "qtd_parcelas",
            "ja_usou_pix_parcelado_antes", "score_credito"]

X = df[features]
y = df["inadimplente"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                      random_state=7, stratify=y)

print(f"treino: {len(X_train)} | teste: {len(X_test)}")
print(f"taxa inadimplência treino: {y_train.mean():.1%} | teste: {y_test.mean():.1%}")
"""),

code("""scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = LogisticRegression(class_weight="balanced")
modelo.fit(X_train_scaled, y_train)

proba = modelo.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"AUC do modelo: {auc:.3f}")
"""),

md("""## Curva ROC

Uma forma visual de ver o quão bom o modelo é pra separar quem paga de quem
atrasa. Quanto mais a curva "sobe" longe da linha pontilhada (que representa um
modelo aleatório), melhor.
"""),

code("""fpr, tpr, _ = roc_curve(y_test, proba)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(fpr, tpr, label=f"Regressão logística (AUC={auc:.2f})", color="#FF7A00")
ax.plot([0,1],[0,1], linestyle="--", color="gray", label="modelo aleatório")
ax.set_xlabel("Taxa de falso positivo")
ax.set_ylabel("Taxa de verdadeiro positivo")
ax.set_title("Curva ROC")
ax.legend()
plt.tight_layout()
plt.savefig("../images/curva_roc.png", dpi=120)
plt.show()
"""),

code("""pred = modelo.predict(X_test_scaled)
print(classification_report(y_test, pred, target_names=["Pagou", "Atrasou"]))

fig, ax = plt.subplots(figsize=(4.5,4.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Pagou","Atrasou"],
                                         cmap="Blues", ax=ax)
plt.tight_layout()
plt.savefig("../images/matriz_confusao.png", dpi=120)
plt.show()
"""),

md("""## Quais variáveis pesaram mais

Olhando o coeficiente de cada variável no modelo, dá pra ter uma ideia de quais
delas mais "empurram" a previsão pra inadimplência ou pra pagamento.
"""),

code("""coef = pd.Series(modelo.coef_[0], index=features).sort_values()

fig, ax = plt.subplots(figsize=(7,4))
coef.plot(kind="barh", color="#FF7A00", ax=ax)
ax.set_title("Peso de cada variável no modelo")
plt.tight_layout()
plt.savefig("../images/coeficientes_modelo.png", dpi=120)
plt.show()
"""),

md("""## O que eu tirei desse estudo (juntando os 3 notebooks)

- Quanto mais parcelado, maior a inadimplência — o número de parcelas parece um
  bom sinal de risco
- Cliente que já usou o Pix parcelado antes atrasa menos, e isso não é só
  coincidência (qui-quadrado confirmou)
- Score de crédito separa bem quem paga de quem atrasa (visualmente e no teste t)
- No teste A/B, o lembrete de cobrança com opção de renegociar parece recuperar
  mais cliente do que o lembrete simples
- O modelo de regressão logística consegue captar esse padrão (AUC acima de 0.5,
  ou seja, melhor que chute aleatório), e as variáveis que mais pesaram fazem
  sentido com o que já tinha visto nos gráficos

Se fosse levar isso pra um cenário real, o próximo passo seria testar limites de
parcelamento por faixa de score/renda, rodar o teste A/B com mais tempo antes de
migrar todo mundo pro lembrete novo, e acompanhar esses números por safra ao
longo do tempo pra ver se o padrão se mantém.
"""),
]

nbf.write(nb(cells1), "notebooks/01_analise_exploratoria.ipynb")
nbf.write(nb(cells2), "notebooks/02_estatistica_teste_ab.ipynb")
nbf.write(nb(cells3), "notebooks/03_modelo_ia.ipynb")
print("notebooks criados")
