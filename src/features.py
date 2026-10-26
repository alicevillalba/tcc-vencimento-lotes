import os
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET = 'was_spoiled'

# -----------------------------------------------------------------------------
# Classificação das colunas do dataset
# -----------------------------------------------------------------------------

# Só se conhecem DEPOIS do resultado do lote -> usar seria vazamento de dados
COLUNAS_VAZAMENTO = [
    'units_sold', 'units_wasted', 'waste_pct', 'waste_cost',
    'revenue', 'profit', 'profit_margin_pct',
]

# Decisões tomadas durante a vida do lote, possivelmente em reação ao risco
COLUNAS_DECISAO_POSTERIOR = ['markdown_applied', 'discount_pct', 'selling_price']

# Escores prontos do autor do dataset, sem documentação de como foram calculados
COLUNAS_ESCORE_PRONTO = ['spoilage_risk', 'quality_grade']

# Identificadores e datas brutas (não descrevem o lote em si)
COLUNAS_IDENTIFICACAO = [
    'record_id', 'product_id', 'product_name', 'store_id', 'supplier_id',
    'transaction_date', 'expiration_date',
]

# Quase cópias de days_until_expiry (correlação de 0,99): mantê-las faria o modelo
# dividir o mesmo efeito entre colunas redundantes e distorceria a leitura do SHAP.
# A informação da vida útil continua no modelo por meio de urgencia_vencimento.
COLUNAS_REDUNDANTES = ['shelf_life_days', 'days_remaining_at_purchase']

# Variáveis de estoque: existem nos dados reais E nos sintéticos
FEATURES_ESTOQUE = [
    'days_until_expiry', 'daily_demand', 'initial_quantity', 'demand_variability',
    'spoilage_sensitivity',
]

# Variáveis de condição do lote: conhecidas antes do vencimento, só nos dados reais
FEATURES_CONDICAO = [
    'storage_temp', 'temp_deviation', 'temp_abuse_events', 'distribution_hours',
    'handling_score', 'packaging_score', 'supplier_score', 'is_promoted',
]

# Códigos fixos das categorias do dataset (ordem alfabética)
CATEGORIAS = ['Bakery', 'Beverages', 'Dairy', 'Deli', 'Frozen_Meals', 'Meat',
              'Pharmaceuticals', 'Produce', 'Ready_to_Eat', 'Seafood']
CODIGOS_CATEGORIA = {c: i for i, c in enumerate(CATEGORIAS)}

# Variáveis derivadas criadas neste trabalho
FEATURES_DERIVADAS = ['giro_estoque', 'urgencia_vencimento', 'risco_acumulado']


# -----------------------------------------------------------------------------
# Parte 1: Diagnóstico dos dados
# -----------------------------------------------------------------------------

def diagnosticar(df: pd.DataFrame, pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Mede a relação de cada variável numérica com o rótulo (AUC-ROC univariada)
    e indica se ela é usada no modelo ou excluída, e por quê.
    AUC = 0,5 significa nenhuma relação; quanto mais longe de 0,5, mais forte.
    """
    def classificar(coluna):
        if coluna in COLUNAS_VAZAMENTO:         return 'excluída: vazamento'
        if coluna in COLUNAS_DECISAO_POSTERIOR: return 'excluída: decisão posterior'
        if coluna in COLUNAS_ESCORE_PRONTO:     return 'excluída: escore sem documentação'
        if coluna in COLUNAS_IDENTIFICACAO:     return 'excluída: identificador'
        if coluna in COLUNAS_REDUNDANTES:       return 'excluída: redundante'
        if coluna in FEATURES_ESTOQUE:          return 'usada: estoque'
        if coluna in FEATURES_CONDICAO:         return 'usada: condição'
        if coluna in FEATURES_DERIVADAS:        return 'usada: derivada'
        if coluna == 'category_enc':            return 'usada: categoria'
        return 'não usada'

    linhas = []
    for coluna in df.select_dtypes('number').columns:
        if coluna == TARGET:
            continue
        valores = df[coluna].fillna(df[coluna].median())
        auc = roc_auc_score(df[TARGET], valores)
        linhas.append({
            'variavel': coluna,
            'auc_univariada': round(auc, 3),
            'forca_relacao': round(abs(auc - 0.5), 3),
            'situacao': classificar(coluna),
        })

    tabela = pd.DataFrame(linhas).sort_values('forca_relacao', ascending=False)
    os.makedirs(pasta_saida, exist_ok=True)
    tabela.to_csv(f'{pasta_saida}/diagnostico_variaveis.csv', index=False)

    print("[features] Diagnóstico (AUC univariada de cada variável):")
    print(tabela.to_string(index=False))

    # Pares de variáveis quase idênticas (|correlação| > 0,95)
    numericas = [c for c in df.select_dtypes('number').columns if c != TARGET]
    corr = df[numericas].corr()
    pares = [{'variavel_1': a, 'variavel_2': b, 'correlacao': round(corr.loc[a, b], 3)}
             for i, a in enumerate(numericas) for b in numericas[i + 1:]
             if abs(corr.loc[a, b]) > 0.95]
    pd.DataFrame(pares).to_csv(f'{pasta_saida}/diagnostico_redundancia.csv', index=False)
    print("\n[features] Pares de variáveis redundantes (|correlação| > 0,95):")
    print(pd.DataFrame(pares).to_string(index=False))
    return tabela


def resumir_dataset(df: pd.DataFrame, pasta_saida: str = 'outputs') -> pd.DataFrame:
    """Números do conjunto real: produtos, lotes, período, positivos e dependência."""
    datas = pd.to_datetime(df['transaction_date'])
    lotes_por_produto = df.groupby('product_id').size()
    farma = df[df['category'] == 'Pharmaceuticals']
    linhas = [
        ('Lotes (registros)', len(df)),
        ('Produtos distintos', df['product_id'].nunique()),
        ('Lotes por produto (média)', round(lotes_por_produto.mean(), 2)),
        ('Lotes por produto (máximo)', int(lotes_por_produto.max())),
        ('Produtos com mais de um lote (%)', round((lotes_por_produto > 1).mean() * 100, 1)),
        ('Categorias', df['category'].nunique()),
        ('Lojas', df['store_id'].nunique()),
        ('Período (início)', datas.min().date()),
        ('Período (fim)', datas.max().date()),
        ('Lotes vencidos', int(df[TARGET].sum())),
        ('Lotes vencidos (%)', round(df[TARGET].mean() * 100, 1)),
        ('Lotes farmacêuticos', len(farma)),
        ('Lotes farmacêuticos vencidos (%)', round(farma[TARGET].mean() * 100, 1)),
    ]
    tabela = pd.DataFrame(linhas, columns=['item', 'valor'])
    os.makedirs(pasta_saida, exist_ok=True)
    tabela.to_csv(f'{pasta_saida}/resumo_dataset.csv', index=False)
    print("[features] Resumo do dataset real:")
    print(tabela.to_string(index=False))
    return tabela


# -----------------------------------------------------------------------------
# Parte 2: Criação das variáveis derivadas
# -----------------------------------------------------------------------------

def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Giro de estoque: em quantos dias o lote esgota no ritmo atual de demanda.
    # Soma 1 na demanda para evitar divisão por zero (lotes com demanda nula).
    df['giro_estoque'] = df['initial_quantity'] / (df['daily_demand'] + 1)

    # Urgência de vencimento: fração da vida útil já consumida (0 = novo, 1 = vencendo)
    df['urgencia_vencimento'] = (
        1 - df['days_until_expiry'] / df['shelf_life_days'].clip(lower=1)
    ).clip(0, 1)

    # Risco acumulado: dias para esgotar divididos pelos dias até vencer
    # (> 1 indica que o lote não esgota a tempo), ampliado pela variabilidade da demanda
    df['risco_acumulado'] = (
        df['giro_estoque'] / df['days_until_expiry'].clip(lower=1)
    ) * (1 + df['demand_variability'])

    # Categoria como número, com códigos FIXOS: assim a mesma categoria recebe o
    # mesmo código em qualquer conjunto (reais, sintéticos, combinados ou na interface)
    df['category_enc'] = df['category'].map(CODIGOS_CATEGORIA).fillna(-1).astype(int)

    return df


# Variáveis que reproduzem a lógica do baseline FEFO (giro / dias até vencer)
FEATURES_SOBREPOSTAS_FEFO = ['giro_estoque', 'risco_acumulado']


def listar_features(incluir_condicao: bool, sem_sobreposicao_fefo: bool = False) -> list:
    """
    Variáveis usadas pelo modelo.
    incluir_condicao=False: só as comuns a reais e sintéticos (cenários B e C)
    incluir_condicao=True:  inclui as de condição (só dados reais, cenário A)
    sem_sobreposicao_fefo=True: remove as variáveis que reproduzem a regra do baseline
    """
    features = FEATURES_ESTOQUE + FEATURES_DERIVADAS + ['category_enc']
    if incluir_condicao:
        features += FEATURES_CONDICAO
    if sem_sobreposicao_fefo:
        features = [f for f in features if f not in FEATURES_SOBREPOSTAS_FEFO]
    return features