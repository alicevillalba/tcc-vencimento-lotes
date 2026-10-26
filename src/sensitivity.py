import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

from data_loader import gerar_sinteticos
from features import criar_features, listar_features
from train import dividir_por_produto, criar_modelos, prever_probabilidade, TARGET, RANDOM_STATE
from evaluate import ESTILOS

# Parâmetros predefinidos (definidos antes de ver os resultados)
PROPORCOES_RISCO = [0.10, 0.20, 0.30]
HORIZONTES = [15, 30, 45]
VELOCIDADES = ['baixa', 'media', 'alta']
REPETICOES = 3
LIMITE_ROBUSTEZ = 0.05  # variação máxima aceitável da AUC-ROC


def rodar_sensibilidade(df_real: pd.DataFrame, n: int = 5000, repeticoes: int = REPETICOES,
                        hiperparametros: dict = None, pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Para cada combinação (proporção de risco x horizonte x velocidade) e cada repetição:
    gera um conjunto sintético novo, divide por produto, treina os três modelos e
    mede AUC-ROC e Brier no teste. Cada repetição usa uma semente diferente
    (42, 43, 44), para separar o efeito dos parâmetros do efeito do acaso.
    """
    features = listar_features(incluir_condicao=False)
    combinacoes = list(itertools.product(PROPORCOES_RISCO, HORIZONTES, VELOCIDADES))
    linhas = []

    print(f"[sensitivity] {len(combinacoes)} combinações x {repeticoes} repetições...")
    for i, (prop, horizonte, velocidade) in enumerate(combinacoes, 1):
        for rep in range(repeticoes):
            semente = RANDOM_STATE + rep
            df = criar_features(gerar_sinteticos(df_real, n=n, prevalencia=prop,
                                                 horizonte=horizonte, velocidade=velocidade,
                                                 semente=semente, mostrar=False))
            treino, teste = dividir_por_produto(df, semente=semente, mostrar=False)
            prev = treino[TARGET].mean()

            for nome, modelo in criar_modelos(semente, True, hiperparametros).items():
                modelo.fit(treino[features], treino[TARGET])
                prob = prever_probabilidade(modelo, teste[features], prev, True)
                linhas.append({
                    'prop_risco': prop, 'horizonte': horizonte, 'velocidade': velocidade,
                    'repeticao': rep + 1, 'modelo': nome,
                    'auc_roc': roc_auc_score(teste[TARGET], prob),
                    'brier': brier_score_loss(teste[TARGET], prob),
                })
        print(f"[sensitivity] {i}/{len(combinacoes)} combinações concluídas")

    resultados = pd.DataFrame(linhas)
    os.makedirs(pasta_saida, exist_ok=True)
    resultados.to_csv(f'{pasta_saida}/sensibilidade_bruto.csv', index=False)
    return resultados


def resumir_sensibilidade(resultados: pd.DataFrame, pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Dois critérios de robustez para cada modelo:
    1) Estabilidade entre repetições: maior desvio-padrão da AUC-ROC entre as
       3 repetições de uma mesma combinação (mede o efeito do acaso).
    2) Efeito de cada parâmetro: diferença entre a maior e a menor AUC-ROC média
       entre os níveis daquele parâmetro (mede o quanto o parâmetro muda o resultado).
    Robusto = todos os valores abaixo de LIMITE_ROBUSTEZ.
    """
    linhas = []
    for nome, grupo in resultados.groupby('modelo'):
        desvio_rep = grupo.groupby(['prop_risco', 'horizonte', 'velocidade'])['auc_roc'].std().max()
        linha = {'modelo': nome,
                 'auc_roc_media': grupo['auc_roc'].mean(),
                 'auc_roc_min': grupo['auc_roc'].min(),
                 'auc_roc_max': grupo['auc_roc'].max(),
                 'maior_desvio_repeticoes': desvio_rep}
        efeitos = []
        for parametro in ['prop_risco', 'horizonte', 'velocidade']:
            medias = grupo.groupby(parametro)['auc_roc'].mean()
            efeito = medias.max() - medias.min()
            linha[f'efeito_{parametro}'] = efeito
            efeitos.append(efeito)
        # Com uma única repetição (modo rápido) não há desvio entre repetições
        valores = efeitos + ([] if pd.isna(desvio_rep) else [desvio_rep])
        linha['robusto'] = 'sim' if max(valores) < LIMITE_ROBUSTEZ else 'não'
        linhas.append(linha)

    resumo = pd.DataFrame(linhas)
    num = resumo.select_dtypes('number').columns
    resumo[num] = resumo[num].round(4)
    resumo.to_csv(f'{pasta_saida}/sensibilidade_resumo.csv', index=False)
    print("\n[sensitivity] Resumo da análise de sensibilidade:")
    print(resumo.to_string(index=False))
    return resumo


def grafico_sensibilidade(resultados: pd.DataFrame, pasta_saida: str = 'outputs'):
    """Um painel por parâmetro: AUC-ROC média (e desvio) de cada modelo em cada nível."""
    paineis = [('prop_risco', 'Proporção de lotes em risco', PROPORCOES_RISCO,
                ['10%', '20%', '30%']),
               ('horizonte', 'Horizonte até o vencimento (dias)', HORIZONTES,
                ['15', '30', '45']),
               ('velocidade', 'Velocidade de saída', VELOCIDADES,
                ['baixa', 'média', 'alta'])]

    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, (coluna, titulo, niveis, rotulos) in zip(eixos, paineis):
        for nome, grupo in resultados.groupby('modelo'):
            estat = grupo.groupby(coluna)['auc_roc'].agg(['mean', 'std']).reindex(niveis)
            ax.errorbar(range(len(niveis)), estat['mean'], yerr=estat['std'], capsize=4,
                        markersize=7, label=nome, **ESTILOS[nome])
        ax.set_xticks(range(len(niveis)))
        ax.set_xticklabels(rotulos)
        ax.set_title(titulo, fontsize=11)
    # Eixo com amplitude fixa de 0,10 para não exagerar visualmente diferenças pequenas
    # (o dobro do limite de robustez de 0,05)
    topo = min(1.0, resultados['auc_roc'].max() + 0.01)
    eixos[0].set_ylim(topo - 0.10, topo)
    eixos[0].set_ylabel('AUC-ROC (média ± desvio)')
    eixos[0].legend(frameon=False, loc='lower left')
    fig.suptitle('Análise de sensibilidade (dados sintéticos)', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{pasta_saida}/sensibilidade.png', dpi=200)
    plt.close(fig)
    print("[sensitivity] Gráfico salvo: sensibilidade.png")