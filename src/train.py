import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.stats import loguniform
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
TARGET = 'was_spoiled'


# -----------------------------------------------------------------------------
# Divisão treino/teste por produto
# -----------------------------------------------------------------------------

def dividir_por_produto(df: pd.DataFrame, test_size: float = 0.2,
                        semente: int = RANDOM_STATE, mostrar: bool = True):
    """
    Separa treino e teste garantindo que todos os lotes de um mesmo produto
    fiquem do mesmo lado (evita dependência entre observações do mesmo produto).
    """
    divisor = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=semente)
    idx_treino, idx_teste = next(divisor.split(df, df[TARGET], groups=df['product_id']))
    treino, teste = df.iloc[idx_treino], df.iloc[idx_teste]
    if mostrar:
        print(f"[train] Treino: {len(treino)} lotes ({treino[TARGET].mean()*100:.1f}% vencidos) | "
              f"Teste: {len(teste)} lotes ({teste[TARGET].mean()*100:.1f}% vencidos)")
    return treino, teste


def dividir_por_tempo(df: pd.DataFrame, fracao_treino: float = 0.8, mostrar: bool = True):
    """
    Validação temporal: treina com os lotes mais antigos e testa com os mais
    recentes (corte na data que deixa `fracao_treino` dos lotes no treino).
    Retorna (treino, teste, data_de_corte).
    """
    datas = pd.to_datetime(df['transaction_date'])
    corte = datas.quantile(fracao_treino)
    treino, teste = df[datas < corte], df[datas >= corte]
    if mostrar:
        print(f"[train] Corte temporal em {corte.date()} | Treino: {len(treino)} lotes | "
              f"Teste: {len(teste)} lotes ({teste[TARGET].mean()*100:.1f}% vencidos)")
    return treino, teste, corte


# -----------------------------------------------------------------------------
# Modelos
# -----------------------------------------------------------------------------

def criar_modelos(semente: int = RANDOM_STATE, usar_smote: bool = True,
                  hiperparametros: dict = None) -> dict:
    """
    Cada modelo é um pipeline: [padronização] -> [SMOTE] -> classificador.
    Colocar o SMOTE dentro do pipeline garante que, na validação cruzada,
    ele seja aplicado só na parte de treino de cada fold, nunca na de validação.
    hiperparametros: resultado de ajustar_hiperparametros (None = valores padrão).
    """
    classificadores = {
        'Regressão Logística': LogisticRegression(max_iter=1000, random_state=semente),
        'Random Forest': RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=semente),
        'XGBoost': XGBClassifier(n_estimators=200, eval_metric='logloss',
                                 n_jobs=-1, random_state=semente, verbosity=0),
    }

    modelos = {}
    for nome, clf in classificadores.items():
        etapas = []
        if nome == 'Regressão Logística':
            etapas.append(('padronizacao', StandardScaler()))  # LR é sensível à escala
        if usar_smote:
            etapas.append(('smote', SMOTE(random_state=semente)))
        etapas.append(('modelo', clf))
        modelos[nome] = Pipeline(etapas)
        if hiperparametros and nome in hiperparametros:
            modelos[nome].set_params(**hiperparametros[nome])
    return modelos


# -----------------------------------------------------------------------------
# Ajuste de hiperparâmetros (busca aleatória)
# -----------------------------------------------------------------------------

# Valores possíveis de cada hiperparâmetro (o prefixo 'modelo__' indica a etapa do pipeline)
ESPACOS_BUSCA = {
    'Regressão Logística': {
        'modelo__C': loguniform(1e-3, 10),  # força da regularização
    },
    'Random Forest': {
        'modelo__n_estimators': [200, 400],
        'modelo__max_depth': [4, 6, 8, 12, None],
        'modelo__min_samples_leaf': [1, 5, 20, 50],
        'modelo__max_features': ['sqrt', 0.5],
    },
    'XGBoost': {
        'modelo__n_estimators': [200, 400, 600],
        'modelo__max_depth': [2, 3, 4, 6],
        'modelo__learning_rate': [0.01, 0.03, 0.1],
        'modelo__subsample': [0.7, 0.8, 1.0],
        'modelo__colsample_bytree': [0.6, 0.8, 1.0],
        'modelo__min_child_weight': [1, 5, 20, 50],
    },
}


def ajustar_hiperparametros(treino: pd.DataFrame, features: list, n_iter: int = 15,
                            k: int = 3, amostra_max: int = 30000,
                            semente: int = RANDOM_STATE, usar_smote: bool = True,
                            pasta_saida: str = 'outputs') -> dict:
    """
    Busca aleatória de hiperparâmetros, feita SÓ no conjunto de treino, com
    validação cruzada agrupada por produto e critério AUC-ROC.
    Para reduzir o tempo, usa uma amostra de até `amostra_max` lotes
    (sorteando produtos inteiros, para manter a regra da divisão por produto).
    Retorna {nome_do_modelo: melhores_hiperparametros}.
    """
    if len(treino) > amostra_max:
        amostra, _ = dividir_por_produto(treino, test_size=1 - amostra_max / len(treino),
                                         semente=semente, mostrar=False)
    else:
        amostra = treino
    X, y, grupos = amostra[features], amostra[TARGET], amostra['product_id']
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semente)

    melhores, linhas = {}, []
    for nome, modelo in criar_modelos(semente, usar_smote).items():
        print(f"[train] Ajuste de hiperparâmetros ({n_iter} configurações): {nome}...")
        busca = RandomizedSearchCV(modelo, ESPACOS_BUSCA[nome], n_iter=n_iter, cv=cv,
                                   scoring='roc_auc', random_state=semente, n_jobs=1)
        busca.fit(X, y, groups=grupos)
        melhores[nome] = busca.best_params_
        linhas.append({'modelo': nome, 'auc_roc_busca': round(busca.best_score_, 4),
                       'hiperparametros': {p.replace('modelo__', ''): v
                                           for p, v in busca.best_params_.items()}})

    tabela = pd.DataFrame(linhas)
    os.makedirs(pasta_saida, exist_ok=True)
    sufixo = 'com_smote' if usar_smote else 'sem_smote'
    tabela.to_csv(f'{pasta_saida}/hiperparametros_{sufixo}.csv', index=False)
    print(tabela.to_string(index=False))
    return melhores


# -----------------------------------------------------------------------------
# Correção das probabilidades após o SMOTE
# -----------------------------------------------------------------------------

def corrigir_prior(prob: np.ndarray, prevalencia: float) -> np.ndarray:
    """
    O SMOTE treina o modelo com 50% de vencidos, mas a proporção real é menor.
    Isso faz o modelo superestimar o risco. Esta função reajusta as probabilidades
    para a proporção real (correção de probabilidade a priori, Saerens et al., 2002):
        p_corrigida = p*pi / (p*pi + (1-p)*(1-pi))
    A ordem dos lotes não muda (AUC igual); só a escala das probabilidades.
    """
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return p * prevalencia / (p * prevalencia + (1 - p) * (1 - prevalencia))


def prever_probabilidade(modelo, X, prevalencia: float, usar_smote: bool) -> np.ndarray:
    """Probabilidade de vencimento, já corrigida quando o modelo usou SMOTE."""
    prob = modelo.predict_proba(X)[:, 1]
    return corrigir_prior(prob, prevalencia) if usar_smote else prob


# -----------------------------------------------------------------------------
# Validação cruzada e treino final
# -----------------------------------------------------------------------------

def validar_e_treinar(treino: pd.DataFrame, features: list, k: int = 10,
                      semente: int = RANDOM_STATE, usar_smote: bool = True,
                      hiperparametros: dict = None):
    """
    1) Validação cruzada k-fold, estratificada e agrupada por produto.
       Guarda a probabilidade prevista para cada lote quando ele estava na
       parte de validação (previsões "fora da dobra"), usadas depois para
       escolher o limiar de decisão sem tocar no conjunto de teste.
    2) Treina cada modelo no conjunto de treino inteiro.

    Retorna um dicionário com: modelos treinados, tabela da validação cruzada,
    previsões fora da dobra, rótulos do treino e prevalência do treino.
    """
    X, y, grupos = treino[features], treino[TARGET].values, treino['product_id']
    prevalencia = y.mean()
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semente)
    modelos = criar_modelos(semente, usar_smote, hiperparametros)

    linhas, previsoes_oof = [], {}
    for nome, modelo in modelos.items():
        print(f"[train] Validação cruzada ({k} folds, SMOTE={usar_smote}): {nome}...")
        oof = np.zeros(len(y))
        aucs, aps, briers = [], [], []

        for idx_fit, idx_val in cv.split(X, y, groups=grupos):
            m = clone(modelo).fit(X.iloc[idx_fit], y[idx_fit])
            prob = prever_probabilidade(m, X.iloc[idx_val], prevalencia, usar_smote)
            oof[idx_val] = prob
            aucs.append(roc_auc_score(y[idx_val], prob))
            aps.append(average_precision_score(y[idx_val], prob))
            briers.append(brier_score_loss(y[idx_val], prob))

        previsoes_oof[nome] = oof
        linhas.append({
            'modelo': nome,
            'auc_roc_media': np.mean(aucs), 'auc_roc_desvio': np.std(aucs),
            'auc_pr_media': np.mean(aps),   'auc_pr_desvio': np.std(aps),
            'brier_media': np.mean(briers), 'brier_desvio': np.std(briers),
        })

        modelo.fit(X, y)  # treino final com todo o conjunto de treino

    tabela_cv = pd.DataFrame(linhas).round(4)
    print(tabela_cv.to_string(index=False))

    return {
        'modelos': modelos,
        'tabela_cv': tabela_cv,
        'oof': previsoes_oof,
        'y_treino': y,
        'prevalencia': prevalencia,
        'usar_smote': usar_smote,
        'features': features,
    }