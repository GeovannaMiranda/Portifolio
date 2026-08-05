"""
Gera uma base fake de compras no Pix parcelado, pra eu poder praticar análise
de inadimplência sem usar dado real de ninguém. Os números não representam
nada real, é só pra simular um cenário parecido com o que rola num banco digital.
"""

import numpy as np
import pandas as pd

np.random.seed(7)

n = 3000

# idade do cliente
idade = np.random.randint(18, 65, n)

# renda estimada (bem simples, sem muita firula)
renda = np.round(np.random.lognormal(mean=7.6, sigma=0.5, size=n), 2)
renda = np.clip(renda, 900, 15000)

# valor da compra parcelada no pix
valor_compra = np.round(np.random.uniform(50, 3000, n), 2)

# quantidade de parcelas (o pix parcelado geralmente é até 12x)
parcelas = np.random.choice([2, 3, 4, 6, 9, 12], size=n, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1])

# se o cliente já tinha usado pix parcelado antes (cliente recorrente ou não)
ja_usou_antes = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

# score de crédito simplificado (0 a 1000, tipo os scores que os bancos usam)
score = np.clip(np.random.normal(600, 150, n), 300, 1000).astype(int)

# regra simples pra decidir quem ficou inadimplente (mais parcelas + valor alto em
# relação a renda + score baixo = mais chance de atrasar)
comprometimento_renda = valor_compra / renda
risco = (
    -0.006 * (score - 600)
    + 1.8 * comprometimento_renda
    + 0.08 * parcelas
    - 0.3 * ja_usou_antes  # cliente recorrente tende a ser mais organizado
    + np.random.normal(0, 1, n)
)
prob = 1 / (1 + np.exp(-(risco - 3.2)))
inadimplente = np.random.binomial(1, prob)

df = pd.DataFrame({
    "cliente_id": [f"CLI{1000+i}" for i in range(n)],
    "idade": idade,
    "renda_estimada": renda,
    "valor_compra": valor_compra,
    "qtd_parcelas": parcelas,
    "ja_usou_pix_parcelado_antes": ja_usou_antes,
    "score_credito": score,
    "inadimplente": inadimplente,
})

# ---- parte extra: simulando um teste A/B de lembrete de cobrança -----------------
# pra quem ficou inadimplente, o banco (fictício) testou duas formas de lembrar o
# cliente de pagar: um lembrete "simples" (notificação padrão) e um lembrete
# "com opção de renegociar parcela" direto no app. Quero ver qual funciona melhor
# pra recuperar o cliente.
df["estrategia_lembrete"] = "N/A"
df.loc[df["inadimplente"] == 1, "estrategia_lembrete"] = np.random.choice(
    ["Lembrete simples", "Lembrete com opção de renegociar"],
    size=(df["inadimplente"] == 1).sum(),
    p=[0.5, 0.5]
)

df["recuperou"] = 0
mask_inad = df["inadimplente"] == 1
prob_recup = np.where(
    df.loc[mask_inad, "estrategia_lembrete"] == "Lembrete com opção de renegociar",
    0.40, 0.29
)
df.loc[mask_inad, "recuperou"] = np.random.binomial(1, prob_recup)

df.to_csv("data/pix_parcelado.csv", index=False)
print(df.shape)
print("taxa de inadimplência:", round(df["inadimplente"].mean() * 100, 1), "%")
