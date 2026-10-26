import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from train import prever_probabilidade, RANDOM_STATE

# Para o teste de sanidade no cenário sintético: papel de cada variável no gerador
VARIAVEIS_MECANISMO = ['initial_quantity', 'daily_demand', 'days_until_expiry',
                       'demand_variability', 'giro_estoque', 'risco_acumulado']
VARIAVEIS_PROXY = ['urgencia_vencimento']  # derivada do prazo e da vida útil
VARIAVEIS_SEM_PAPEL = ['spoilage_sensitivity', 'category_enc']


def calcular_shap(resultado_treino: dict, nome_modelo: str, teste: pd.DataFrame,
                  n_amostra: int = 2000, semente: int = RANDOM_STATE) -> shap.Explanation:
    """
    Calcula os valores SHAP do modelo para uma amostra de lotes do teste.
    O SHAP explica a saída do classificador antes da correção de probabilidade
    do SMOTE; como essa correção só desloca todas as previsões igualmente
    (na escala log-odds), a importância e o sentido de cada variável não mudam.
    """
    features = resultado_treino['features']
    pipeline = resultado_treino['modelos'][nome_modelo]
    modelo = pipeline.named_steps['modelo']

    amostra = teste.sample(min(n_amostra, len(teste)), random_state=semente)
    X = amostra[features]
    X_modelo = X.values.astype(float)
    if 'padronizacao' in pipeline.named_steps:  # a regressão logística vê os dados padronizados
        X_modelo = pipeline.named_steps['padronizacao'].transform(X)

    if nome_modelo == 'Regressão Logística':
        explicador = shap.LinearExplainer(modelo, X_modelo)
        valores = explicador(X_modelo)
    else:
        explicador = shap.TreeExplainer(modelo)
        valores = explicador(X_modelo)
        if valores.values.ndim == 3:  # Random Forest devolve uma saída por classe
            valores = valores[:, :, 1]

    # Mostra os valores originais (não padronizados) nos gráficos
    return shap.Explanation(values=valores.values, base_values=valores.base_values,
                            data=X.values, feature_names=features), amostra


def importancia_global(valores: shap.Explanation) -> pd.DataFrame:
    """Importância média de cada variável = média do |valor SHAP|."""
    media_abs = np.abs(valores.values).mean(axis=0)
    tabela = pd.DataFrame({'variavel': valores.feature_names,
                           'importancia_media_abs': media_abs})
    tabela['participacao_%'] = 100 * tabela['importancia_media_abs'] / media_abs.sum()
    return tabela.sort_values('importancia_media_abs', ascending=False).round(4)


def explicar(resultado_treino: dict, nome_modelo: str, teste: pd.DataFrame,
             nome_cenario: str, pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Gera:
    - gráfico de resumo (beeswarm): importância e sentido de cada variável;
    - gráfico waterfall do lote de maior risco: por que o modelo o classificou assim;
    - tabela de importância global.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    valores, amostra = calcular_shap(resultado_treino, nome_modelo, teste)

    tabela = importancia_global(valores)
    tabela.to_csv(f'{pasta_saida}/shap_importancia_{nome_cenario}.csv', index=False)
    print(f"\n[explainability] Importância SHAP ({nome_modelo}, cenário {nome_cenario}):")
    print(tabela.to_string(index=False))

    # Resumo global
    plt.figure()
    shap.plots.beeswarm(valores, max_display=12, show=False)
    plt.xlabel('Valor SHAP (impacto na previsão, escala log-odds)')
    plt.title(f'SHAP - {nome_modelo} (cenário {nome_cenario})', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{pasta_saida}/shap_resumo_{nome_cenario}.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Explicação local: um lote claramente em risco, mas não um caso extremo
    # (o de probabilidade mais próxima de 75%), para o gráfico ser representativo
    r = resultado_treino
    prob = prever_probabilidade(r['modelos'][nome_modelo], amostra[r['features']],
                                r['prevalencia'], r['usar_smote'])
    i = int(np.argmin(np.abs(prob - 0.75)))
    plt.figure()
    shap.plots.waterfall(valores[i], max_display=10, show=False)
    plt.title(f'Lote {amostra.iloc[i]["product_id"]} - probabilidade de vencer: {prob[i]*100:.1f}%',
              fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{pasta_saida}/shap_lote_{nome_cenario}.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[explainability] Gráficos salvos: shap_resumo_{nome_cenario}.png, "
          f"shap_lote_{nome_cenario}.png")
    return tabela


def verificar_mecanismo(tabela_importancia: pd.DataFrame, nome_cenario: str = 'B',
                        pasta_saida: str = 'outputs') -> pd.DataFrame:
    """
    Teste de sanidade do SHAP no cenário sintético, onde se sabe o que causa o
    vencimento. Soma a participação na importância de cada grupo de variáveis.
    Esperado: quase toda a importância nas variáveis do mecanismo (e nas suas
    quase cópias), e quase nenhuma nas variáveis sem papel no gerador.
    """
    t = tabela_importancia.set_index('variavel')['participacao_%']
    grupos = {'Mecanismo (quantidade, demanda, prazo, variabilidade e derivadas)':
                  VARIAVEIS_MECANISMO,
              'Derivada do prazo e da vida útil': VARIAVEIS_PROXY,
              'Sem papel no gerador': VARIAVEIS_SEM_PAPEL}
    resumo = pd.DataFrame([{'grupo': g, 'participacao_%': round(float(t.reindex(v).fillna(0).sum()), 1)}
                           for g, v in grupos.items()])
    resumo.to_csv(f'{pasta_saida}/shap_verificacao_mecanismo_{nome_cenario}.csv', index=False)
    print(f"\n[explainability] Verificação do mecanismo (cenário {nome_cenario}):")
    print(resumo.to_string(index=False))
    return resumo