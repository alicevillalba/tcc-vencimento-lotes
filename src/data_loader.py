import numpy as np
import pandas as pd

RANDOM_STATE = 42

# Multiplicador da demanda diária para cada nível de velocidade de saída
VELOCIDADES = {'baixa': 0.5, 'media': 1.0, 'alta': 1.5}

# Dias de histórico de vendas usados para estimar a demanda diária de cada lote
DIAS_HISTORICO = 14


def carregar_dataset(caminho: str) -> pd.DataFrame:
    """Lê o dataset original (Fatima - Kaggle) e marca a origem."""
    df = pd.read_csv(caminho)
    df['origem'] = 'real'
    print(f"[data_loader] Dataset real: {df.shape[0]} linhas | "
          f"Vencidos: {df['was_spoiled'].mean()*100:.1f}%")
    return df


def gerar_sinteticos(df_real: pd.DataFrame, n: int = 5000,
                     prevalencia: float = None, horizonte: int = None,
                     velocidade: str = 'media', semente: int = RANDOM_STATE,
                     mostrar: bool = True) -> pd.DataFrame:
    """
    Gera lotes farmacêuticos sintéticos simulando consumo e perda por
    vencimento (outdating), conforme a dinâmica descrita por Nahmias (1982).

    prevalencia: proporção de lotes vencidos (padrão: taxa real dos farmacêuticos)
    horizonte:   dias até o vencimento; se None, usa os dias dos lotes reais sorteados
    velocidade:  'baixa', 'media' ou 'alta' (multiplica a demanda diária)
    """
    rng = np.random.default_rng(semente)
    farma = df_real[df_real['category'] == 'Pharmaceuticals']

    # Sorteia lotes farmacêuticos reais inteiros (com reposição) e usa os valores
    # deles: assim as distribuições E as relações entre as variáveis (ex.: vida
    # útil x dias até o vencimento) são as mesmas dos dados reais
    linhas = farma.iloc[rng.integers(0, len(farma), n)]

    def sortear(coluna):
        return linhas[coluna].values

    shelf_life  = sortear('shelf_life_days')
    demand_var  = sortear('demand_variability')
    initial_qty = sortear('initial_quantity')

    if horizonte is None:
        days_expiry = sortear('days_until_expiry').clip(1)
    else:
        days_expiry = rng.integers(1, horizonte + 1, n)

    # Taxa média de demanda do lote. A demanda no dataset é um inteiro arredondado
    # ("0" = menos de meia unidade por dia, não "nenhuma venda"); por isso cada valor
    # é espalhado dentro do seu intervalo de arredondamento (ex.: 2 -> entre 1,5 e 2,5).
    demanda_inteira = sortear('daily_demand')
    taxa = (demanda_inteira + rng.uniform(-0.5, 0.5, n)).clip(min=0.05)
    taxa = taxa * VELOCIDADES[velocidade]

    # O que o sistema enxerga é a demanda ESTIMADA a partir de um histórico curto de
    # vendas (Poisson em DIAS_HISTORICO dias). Produtos de baixo giro têm estimativa
    # mais imprecisa -- como na prática.
    daily_demand = rng.poisson(taxa * DIAS_HISTORICO) / DIAS_HISTORICO

    # Incerteza sobre o futuro:
    # 1) a taxa futura difere da passada (fator Gama com média 1 e coeficiente de
    #    variação = variabilidade da demanda; modelo Poisson-Gama / binomial negativa);
    # 2) flutuação diária em torno dessa taxa; a soma de muitos dias é aproximada
    #    por uma Normal (Teorema Central do Limite).
    cv = demand_var.clip(min=0.05)
    taxa_futura = taxa * rng.gamma(shape=1 / cv**2, scale=cv**2)
    consumo_medio  = taxa_futura * days_expiry
    consumo_desvio = np.sqrt(days_expiry) * taxa_futura * demand_var
    consumo = rng.normal(consumo_medio, consumo_desvio).clip(0)

    # Proporção do lote que sobrou no vencimento: max(quantidade - consumo, 0) / quantidade
    sobra = ((initial_qty - consumo) / initial_qty).clip(0, 1)

    # Marca como vencidos os lotes com maior sobra, até atingir a prevalência desejada.
    # Empates (ex.: vários lotes com sobra de 100%) são desempatados por sorteio.
    if prevalencia is None:
        prevalencia = farma['was_spoiled'].mean()
    k = round(prevalencia * n)
    ordem = np.lexsort((rng.random(n), -sobra))  # maior sobra primeiro
    was_spoiled = np.zeros(n, dtype=int)
    was_spoiled[ordem[:k]] = 1

    df_sint = pd.DataFrame({
        'product_id':                 [f'SINT-{semente}-{i:05d}' for i in range(n)],
        'category':                   'Pharmaceuticals',
        'shelf_life_days':            shelf_life,
        'days_until_expiry':          days_expiry,
        'daily_demand':               daily_demand,
        'initial_quantity':           initial_qty,
        'demand_variability':         demand_var,
        'days_remaining_at_purchase': days_expiry + rng.integers(0, 4, n),
        'spoilage_sensitivity':       sortear('spoilage_sensitivity'),
        'was_spoiled':                was_spoiled,
        'origem':                     'sintetico',
    })

    if mostrar:
        print(f"[data_loader] Sintéticos: {n} linhas | Vencidos: {was_spoiled.mean()*100:.1f}% "
              f"(alvo: {prevalencia*100:.1f}%) | horizonte={horizonte} | velocidade={velocidade}")
    return df_sint


def comparar_com_reais(df_real: pd.DataFrame, df_sint: pd.DataFrame,
                       pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Compara as distribuições das variáveis dos sintéticos com as dos lotes
    farmacêuticos reais (média, mediana e estatística KS de Kolmogorov-Smirnov:
    0 = distribuições iguais, 1 = totalmente diferentes).
    """
    import os
    from scipy.stats import ks_2samp
    farma = df_real[df_real['category'] == 'Pharmaceuticals']
    linhas = []
    for coluna in ['initial_quantity', 'daily_demand', 'demand_variability',
                   'days_until_expiry', 'spoilage_sensitivity', 'was_spoiled']:
        sint = df_sint[coluna]
        if coluna == 'daily_demand':
            # No dataset real a demanda é inteira; a sintética é uma estimativa contínua.
            # Para comparar na mesma escala, arredonda-se a sintética.
            sint = sint.round()
        linhas.append({
            'variavel': coluna,
            'media_real': round(farma[coluna].mean(), 3),
            'media_sintetico': round(sint.mean(), 3),
            'mediana_real': round(farma[coluna].median(), 3),
            'mediana_sintetico': round(sint.median(), 3),
            'ks': round(ks_2samp(farma[coluna], sint).statistic, 3),
        })
    tabela = pd.DataFrame(linhas)
    os.makedirs(pasta_saida, exist_ok=True)
    tabela.to_csv(f'{pasta_saida}/validacao_sinteticos.csv', index=False)
    print("[data_loader] Sintéticos x farmacêuticos reais:")
    print(tabela.to_string(index=False))
    return tabela


def combinar(df_real: pd.DataFrame, df_sint: pd.DataFrame) -> pd.DataFrame:
    """Junta reais e sintéticos mantendo só as colunas em comum."""
    colunas = [c for c in df_sint.columns if c in df_real.columns]
    df = pd.concat([df_real[colunas], df_sint[colunas]], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[data_loader] Combinado: {df.shape[0]} linhas | "
          f"Vencidos: {df['was_spoiled'].mean()*100:.1f}%")
    return df