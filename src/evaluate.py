import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # gera imagens sem abrir janela
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, precision_score, recall_score, precision_recall_curve, roc_curve,
)
from sklearn.calibration import calibration_curve

from train import prever_probabilidade, TARGET, RANDOM_STATE

# Cores fixas por série + estilo de linha (legível também impresso em preto e branco)
ESTILOS = {
    'Regressão Logística': dict(color='#2a78d6', linestyle='-',  marker='o'),
    'Random Forest':       dict(color='#eb6834', linestyle='--', marker='s'),
    'XGBoost':             dict(color='#1baf7a', linestyle='-.', marker='^'),
    'Baseline FEFO':       dict(color='#6b6b6b', linestyle=':',  marker='D'),
}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'lines.linewidth': 2})


# -----------------------------------------------------------------------------
# Escolhas feitas SEM olhar o teste
# -----------------------------------------------------------------------------

def escolher_melhor(resultado_treino: dict) -> str:
    """Melhor modelo = maior AUC-ROC média na validação cruzada (não no teste)."""
    tabela = resultado_treino['tabela_cv']
    return tabela.sort_values('auc_roc_media', ascending=False).iloc[0]['modelo']


def escolher_limiar(y: np.ndarray, prob: np.ndarray) -> float:
    """
    Limiar de decisão que maximiza o F1 nas previsões fora da dobra do treino.
    Acima do limiar, o lote é classificado como 'em risco'.
    """
    precisao, recall, limiares = precision_recall_curve(y, prob)
    f1 = 2 * precisao * recall / np.clip(precisao + recall, 1e-12, None)
    return float(limiares[np.argmax(f1[:-1])])


# -----------------------------------------------------------------------------
# Intervalo de confiança por bootstrap (reamostrando produtos, não lotes)
# -----------------------------------------------------------------------------

def intervalo_confianca(y, prob, grupos, metrica=roc_auc_score, n=1000,
                        semente=RANDOM_STATE):
    """
    IC de 95% da métrica. Sorteia produtos com reposição (coerente com a divisão
    por produto) e recalcula a métrica n vezes, usando pesos por lote.
    """
    rng = np.random.default_rng(semente)
    codigos, grupo_do_lote = np.unique(np.asarray(grupos), return_inverse=True)
    n_grupos = len(codigos)
    valores = []
    for _ in range(n):
        contagem = np.bincount(rng.integers(0, n_grupos, n_grupos), minlength=n_grupos)
        pesos = contagem[grupo_do_lote]
        if len(np.unique(y[pesos > 0])) < 2:
            continue
        valores.append(metrica(y, prob, sample_weight=pesos))
    return np.percentile(valores, 2.5), np.percentile(valores, 97.5)


# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------

def calcular_metricas(y, prob, limiar, grupos, e_probabilidade=True) -> dict:
    pred = (prob >= limiar).astype(int)
    ic_inf, ic_sup = intervalo_confianca(y, prob, grupos)
    return {
        'limiar': round(limiar, 3),
        'precisao': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'auc_roc': roc_auc_score(y, prob),
        'auc_roc_ic95': f'[{ic_inf:.3f}; {ic_sup:.3f}]',
        'auc_pr': average_precision_score(y, prob),
        # Brier e log-loss só fazem sentido para probabilidades
        'brier': brier_score_loss(y, prob) if e_probabilidade else np.nan,
        'log_loss': log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)) if e_probabilidade else np.nan,
    }


def pontuacao_fefo(df: pd.DataFrame) -> np.ndarray:
    """
    Baseline: regra determinística inspirada no FEFO.
    Razão entre os dias necessários para esgotar o lote (no ritmo atual de demanda)
    e os dias até o vencimento. Razão > 1: o lote não esgota a tempo -> em risco.
    Não é uma probabilidade, por isso Brier e log-loss não se aplicam.
    """
    return (df['giro_estoque'] / df['days_until_expiry'].clip(lower=1)).values


def avaliar_teste(resultado_treino: dict, teste: pd.DataFrame,
                  nome_cenario: str, pasta_saida: str = 'outputs') -> tuple:
    """
    Avalia todos os modelos e o baseline no conjunto de teste.
    Retorna (tabela de métricas, dicionário de probabilidades por modelo).
    """
    r = resultado_treino
    X, y = teste[r['features']], teste[TARGET].values
    grupos = teste['product_id'].values
    linhas, probs = [], {}

    # Referência sem habilidade: prevê sempre a proporção de vencidos do treino.
    # Um modelo útil precisa ter Brier menor que esta linha.
    prob_ref = np.full(len(y), r['prevalencia'])
    linhas.append({'modelo': 'Referência (prevalência)', 'limiar': np.nan,
                   'brier': brier_score_loss(y, prob_ref),
                   'log_loss': log_loss(y, prob_ref), 'auc_roc': 0.5})

    # Baseline FEFO: limiar fixo em 1 (definição da regra)
    prob_fefo = pontuacao_fefo(teste)
    probs['Baseline FEFO'] = prob_fefo
    linhas.append({'modelo': 'Baseline FEFO',
                   **calcular_metricas(y, prob_fefo, 1.0, grupos, e_probabilidade=False)})

    # Modelos: limiar escolhido nas previsões fora da dobra do treino
    for nome, modelo in r['modelos'].items():
        prob = prever_probabilidade(modelo, X, r['prevalencia'], r['usar_smote'])
        limiar = escolher_limiar(r['y_treino'], r['oof'][nome])
        probs[nome] = prob
        linhas.append({'modelo': nome, **calcular_metricas(y, prob, limiar, grupos)})

    ordem = ['modelo', 'auc_roc', 'auc_roc_ic95', 'auc_pr', 'f1', 'precisao', 'recall',
             'limiar', 'brier', 'log_loss']
    tabela = pd.DataFrame(linhas)[ordem]
    colunas_num = tabela.select_dtypes('number').columns
    tabela[colunas_num] = tabela[colunas_num].round(4)

    os.makedirs(pasta_saida, exist_ok=True)
    tabela.to_csv(f'{pasta_saida}/metricas_teste_{nome_cenario}.csv', index=False)
    print(f"\n[evaluate] Métricas no teste - cenário {nome_cenario}:")
    print(tabela.to_string(index=False))
    return tabela, probs


# -----------------------------------------------------------------------------
# Comparação: com SMOTE (bruto), com SMOTE (corrigido) e sem SMOTE
# -----------------------------------------------------------------------------

def comparar_smote(res_smote: dict, res_sem: dict, nome_modelo: str,
                   teste: pd.DataFrame, nome_cenario: str,
                   pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Mostra o efeito do SMOTE na calibração das probabilidades do melhor modelo.
    """
    X, y = teste[res_smote['features']], teste[TARGET].values
    modelo_smote = res_smote['modelos'][nome_modelo]
    variantes = {
        'Com SMOTE (bruto)':     modelo_smote.predict_proba(X)[:, 1],
        'Com SMOTE (corrigido)': prever_probabilidade(modelo_smote, X,
                                                      res_smote['prevalencia'], True),
        'Sem SMOTE':             prever_probabilidade(res_sem['modelos'][nome_modelo], X,
                                                      res_sem['prevalencia'], False),
    }

    linhas = []
    for rotulo, prob in variantes.items():
        linhas.append({'variante': rotulo,
                       'prob_media_prevista': prob.mean(),
                       'taxa_real_vencidos': y.mean(),
                       'auc_roc': roc_auc_score(y, prob),
                       'brier': brier_score_loss(y, prob),
                       'log_loss': log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))})
    tabela = pd.DataFrame(linhas).round(4)
    tabela.to_csv(f'{pasta_saida}/comparacao_smote_{nome_cenario}.csv', index=False)
    print(f"\n[evaluate] Efeito do SMOTE na calibração ({nome_modelo}, cenário {nome_cenario}):")
    print(tabela.to_string(index=False))

    estilos_var = [dict(color='#eb6834', linestyle='--', marker='s'),
                   dict(color='#2a78d6', linestyle='-', marker='o'),
                   dict(color='#1baf7a', linestyle='-.', marker='^')]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color='#9a9a9a', linestyle=':', linewidth=1.5,
            label='Calibração perfeita')
    for (rotulo, prob), estilo in zip(variantes.items(), estilos_var):
        frac, media = calibration_curve(y, prob, n_bins=10, strategy='quantile')
        ax.plot(media, frac, markersize=7, label=rotulo, **estilo)
    ax.set_xlabel('Probabilidade prevista (média do grupo)')
    ax.set_ylabel('Proporção real de vencidos')
    ax.set_title(f'Calibração - {nome_modelo} (cenário {nome_cenario})')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'{pasta_saida}/calibracao_smote_{nome_cenario}.png', dpi=200)
    plt.close(fig)
    return tabela


# -----------------------------------------------------------------------------
# Gráficos do teste
# -----------------------------------------------------------------------------

def gerar_graficos(teste: pd.DataFrame, probs: dict, nome_cenario: str,
                   pasta_saida: str = 'outputs'):
    y = teste[TARGET].values

    # Curva ROC
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color='#9a9a9a', linestyle=':', linewidth=1.5,
            label='Aleatório (AUC = 0.5)')
    for nome, prob in probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        estilo = {k: v for k, v in ESTILOS[nome].items() if k != 'marker'}
        ax.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc_score(y, prob):.3f})', **estilo)
    ax.set_xlabel('Taxa de falsos positivos')
    ax.set_ylabel('Taxa de verdadeiros positivos (recall)')
    ax.set_title(f'Curva ROC - cenário {nome_cenario}')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{pasta_saida}/curva_roc_{nome_cenario}.png', dpi=200)
    plt.close(fig)

    # Curva precisão-recall (mais informativa com classes desbalanceadas)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.axhline(y.mean(), color='#9a9a9a', linestyle=':', linewidth=1.5,
               label=f'Aleatório (= {y.mean():.2f})')
    for nome, prob in probs.items():
        precisao, recall, _ = precision_recall_curve(y, prob)
        estilo = {k: v for k, v in ESTILOS[nome].items() if k != 'marker'}
        ax.plot(recall, precisao,
                label=f'{nome} (AUC-PR = {average_precision_score(y, prob):.3f})', **estilo)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precisão')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'Curva precisão-recall - cenário {nome_cenario}')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{pasta_saida}/curva_pr_{nome_cenario}.png', dpi=200)
    plt.close(fig)

    # Curva de calibração (só modelos; o baseline não gera probabilidades)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color='#9a9a9a', linestyle=':', linewidth=1.5,
            label='Calibração perfeita')
    for nome, prob in probs.items():
        if nome == 'Baseline FEFO':
            continue
        frac, media = calibration_curve(y, prob, n_bins=10, strategy='quantile')
        ax.plot(media, frac, markersize=7, label=nome, **ESTILOS[nome])
    ax.set_xlabel('Probabilidade prevista (média do grupo)')
    ax.set_ylabel('Proporção real de vencidos')
    ax.set_title(f'Curva de calibração - cenário {nome_cenario}')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'{pasta_saida}/calibracao_{nome_cenario}.png', dpi=200)
    plt.close(fig)

    print(f"[evaluate] Gráficos salvos: curva_roc_, curva_pr_, calibracao_{nome_cenario}.png")


# -----------------------------------------------------------------------------
# Matriz de confusão do melhor modelo
# -----------------------------------------------------------------------------

def matriz_confusao(resultado_treino: dict, teste: pd.DataFrame, nome_modelo: str,
                    nome_cenario: str, pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Quantos lotes vencidos o modelo identificou e quantos alarmes falsos gerou,
    usando o limiar escolhido na validação cruzada.
    """
    from sklearn.metrics import confusion_matrix
    r = resultado_treino
    y = teste[TARGET].values
    prob = prever_probabilidade(r['modelos'][nome_modelo], teste[r['features']],
                                r['prevalencia'], r['usar_smote'])
    limiar = escolher_limiar(r['y_treino'], r['oof'][nome_modelo])
    matriz = confusion_matrix(y, (prob >= limiar).astype(int))

    rotulos = ['Não venceu', 'Venceu']
    tabela = pd.DataFrame(matriz, index=[f'Real: {r_}' for r_ in rotulos],
                          columns=[f'Previsto: {r_}' for r_ in rotulos])
    tabela.to_csv(f'{pasta_saida}/matriz_confusao_{nome_cenario}.csv')

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(matriz, cmap='Blues')
    ax.grid(False)
    total_linha = matriz.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            escuro = matriz[i, j] > matriz.max() * 0.5
            ax.text(j, i, f'{matriz[i, j]:,}'.replace(',', '.') +
                    f'\n({matriz[i, j] / total_linha[i, 0] * 100:.1f}% da linha)',
                    ha='center', va='center', fontsize=11,
                    color='white' if escuro else '#1a1a1a')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(rotulos)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(rotulos)
    ax.set_xlabel('Previsto pelo modelo')
    ax.set_ylabel('Situação real do lote')
    ax.set_title(f'Matriz de confusão - {nome_modelo}\n(cenário {nome_cenario}, limiar = {limiar:.3f})',
                 fontsize=11)
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{pasta_saida}/matriz_confusao_{nome_cenario}.png', dpi=200)
    plt.close(fig)
    print(f"\n[evaluate] Matriz de confusão ({nome_modelo}, cenário {nome_cenario}):")
    print(tabela.to_string())
    return tabela