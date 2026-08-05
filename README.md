# Inadimplência no Pix Parcelado (estudo)

Esse é um projeto pessoal que fiz pra praticar análise de dados aplicada a crédito,
usando como tema o **Pix parcelado**. Simulei uma base de compras parceladas
(os dados são fictícios, gerados por script) e fui atrás de responder algumas
perguntas simples que fariam sentido pra um time de crédito/cobrança de um banco
digital:

- O número de parcelas influencia na inadimplência?
- Cliente que já usou o produto antes atrasa menos?
- Entre duas formas de lembrete de cobrança, qual recupera mais cliente? (teste A/B)
- Dá pra usar um modelo simples pra prever quem tem mais chance de atrasar?

## O que tem aqui

- `data/gerar_dados.py` → gera a base fictícia (3.000 compras, com um cenário de
  teste A/B de lembrete de cobrança pra quem ficou inadimplente)
- `notebooks/01_analise_exploratoria.ipynb` → primeira olhada nos dados, gráficos e
  hipóteses iniciais
- `notebooks/02_estatistica_teste_ab.ipynb` → testes estatísticos (qui-quadrado,
  teste t) pra confirmar as hipóteses do notebook 1, e um teste A/B comparando duas
  estratégias de lembrete de cobrança
- `notebooks/03_modelo_ia.ipynb` → modelo simples de regressão logística pra prever
  inadimplência, com curva ROC e matriz de confusão
- `sql/analises_pix_parcelado.sql` → as mesmas perguntas de negócio resolvidas em SQL

## Como rodar

```bash
pip install -r requirements.txt
python data/gerar_dados.py
jupyter notebook notebooks/
```

## Algumas conclusões que cheguei

- Inadimplência sobe conforme aumenta o número de parcelas
- Cliente recorrente (que já usou o produto antes) atrasa menos, e isso não é só
  coincidência (testei com qui-quadrado)
- Score de crédito separa bem quem paga de quem atrasa (teste t confirmou)
- No teste A/B, o lembrete de cobrança com opção de renegociar recuperou mais
  cliente do que o lembrete simples, com diferença estatisticamente significativa
- O modelo simples de regressão logística já consegue captar esse padrão

Ferramentas usadas: Python, pandas, SQL, scikit-learn, scipy e statsmodels (testes
estatísticos e teste A/B), matplotlib/seaborn.

---
*Projeto de estudo/portfólio, feito com dados fictícios.*
