# Predição de probabilidade de vencimento de lotes farmacêuticos

Código do Trabalho de Conclusão de Curso *"Predição de probabilidade de vencimento de lotes farmacêuticos com aprendizado de máquina e explicabilidade via SHAP"*, do curso de Sistemas de Informação da PUC Minas, campus Contagem.

- **Autora:** Alice Machado Villalba Costa
- **Orientadora:** Profa. Maria Inês Lage de Paula

## Sobre o projeto

O projeto estima, para cada lote em estoque, a probabilidade de vencer antes de ser totalmente consumido. Três algoritmos são comparados entre si e com uma regra de referência inspirada na política FEFO (*First Expire, First Out*): Regressão Logística, Random Forest e XGBoost. As predições são explicadas com SHAP, e uma aplicação web em Streamlit apresenta o risco e a explicação de cada lote.

Principais etapas:

1. Carregamento do *dataset* público e geração de 5.000 lotes farmacêuticos sintéticos;
2. Diagnóstico das variáveis, exclusão das que causariam vazamento de dados e criação das variáveis preditoras;
3. Divisão treino/teste por produto (80/20);
4. Ajuste de hiperparâmetros e validação cruzada estratificada e agrupada por produto (k = 10), com SMOTE e correção *a priori* das probabilidades;
5. Avaliação no teste (AUC-ROC com IC 95%, AUC-PR, F1, Brier, *log-loss*) e comparação com a regra inspirada no FEFO;
6. Cenários experimentais (A, A-estoque, A-sem-FEFO, A-temporal, B e C) e análise de sensibilidade dos dados sintéticos;
7. Explicabilidade com SHAP e aplicação web.

## Dados

O projeto usa o *dataset* público **Managing Perishable Inventory Data** (Fatima, 2023), disponível no Kaggle:

https://www.kaggle.com/datasets/minahilfatima12328/managing-perishable-inventory-data

O arquivo não está incluído no repositório. Baixe-o e salve-o como:

```
data/perishable_goods_management.csv
```

## Estrutura

```
tcc-vencimento-lotes/
├── data/                  # dataset do Kaggle (baixar separadamente)
├── outputs/               # resultados gerados pelo pipeline (tabelas, gráficos e modelo final)
├── src/
│   ├── data_loader.py     # carregamento do dataset e geração dos dados sintéticos
│   ├── features.py        # diagnóstico e criação das variáveis preditoras
│   ├── train.py           # divisão dos dados, ajuste de hiperparâmetros e validação cruzada
│   ├── evaluate.py        # métricas, regra inspirada no FEFO e gráficos
│   ├── sensitivity.py     # análise de sensibilidade dos dados sintéticos
│   ├── explainability.py  # explicações com SHAP
│   ├── exemplo.py         # gera os lotes de exemplo usados pela aplicação
│   ├── pipeline.py        # executa todas as etapas
│   └── app.py             # aplicação web (Streamlit)
├── .streamlit/            # configuração visual da aplicação
└── requirements.txt
```

## Como executar

Requer Python 3.12.

**1. Instalar as dependências**

```bash
pip install -r requirements.txt
```

**2. Rodar o pipeline completo**

```bash
python src/pipeline.py
```

A execução completa leva cerca de 90 minutos e salva todos os resultados em `outputs/`. Para apenas testar se tudo funciona, use o modo rápido (poucos minutos, resultados não definitivos):

```bash
python src/pipeline.py --rapido
```

**3. Abrir a aplicação web**

```bash
streamlit run src/app.py
```

A aplicação usa o modelo salvo em `outputs/modelo_final.joblib` e os lotes de exemplo em `outputs/lotes_exemplo.csv`, que já estão no repositório. Por isso, ela funciona sem rodar o pipeline antes. Na aba **Lotes do estoque**, é possível enviar um CSV de lotes ou usar os exemplos. Na aba **Simular um lote**, é possível informar os dados de um lote manualmente.

## Principais resultados

| Cenário | Melhor modelo | AUC-ROC (IC 95%) | Regra FEFO |
|---|---|---|---|
| A (dados reais) | Regressão Logística | 0,588 [0,578; 0,597] | 0,505 |
| A-temporal | Regressão Logística | 0,589 [0,579; 0,597] | 0,509 |
| B (dados sintéticos) | XGBoost | 0,982 [0,975; 0,989] | 0,801 |

Os modelos superam a regra de referência de forma consistente, mas com ganho modesto nos dados reais. Os experimentos de controle indicam que essa limitação está nos dados, e não nos algoritmos. A tabela completa está em `outputs/resumo_cenarios.csv`.

## Reprodutibilidade

Todas as etapas aleatórias usam semente fixa (42). As versões das bibliotecas estão em `requirements.txt`.
