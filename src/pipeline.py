# =============================================================================
# TCC - Predição de probabilidade de vencimento de lotes farmacêuticos
#       com aprendizado de máquina e explicabilidade via SHAP
# Autora: Alice Machado Villalba Costa
#
# Uso (dentro da pasta do repositório):
#   python src/pipeline.py            -> execução completa (resultados do TCC)
#   python src/pipeline.py --rapido   -> execução reduzida, só para testar
# =============================================================================

import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd

from data_loader import carregar_dataset, gerar_sinteticos, combinar, comparar_com_reais
from features import diagnosticar, resumir_dataset, criar_features, listar_features
from train import (dividir_por_produto, dividir_por_tempo, ajustar_hiperparametros,
                   validar_e_treinar, TARGET)
from evaluate import (escolher_melhor, escolher_limiar, avaliar_teste, gerar_graficos,
                      comparar_smote, matriz_confusao)
from sensitivity import rodar_sensibilidade, resumir_sensibilidade, grafico_sensibilidade
from explainability import explicar, verificar_mecanismo
from exemplo import salvar_lotes_exemplo

CAMINHO_DADOS = 'data/perishable_goods_management.csv'
PASTA = 'outputs'


def titulo(texto):
    print('\n' + '=' * 70 + f'\n{texto}\n' + '=' * 70)


def linha_resumo(cenario, descricao, tabela_teste, melhor):
    """Extrai a linha do melhor modelo e do baseline para a tabela-resumo."""
    m = tabela_teste.set_index('modelo')
    return {'cenario': cenario, 'descricao': descricao, 'melhor_modelo': melhor,
            'auc_roc': m.loc[melhor, 'auc_roc'], 'auc_roc_ic95': m.loc[melhor, 'auc_roc_ic95'],
            'auc_pr': m.loc[melhor, 'auc_pr'], 'f1': m.loc[melhor, 'f1'],
            'brier': m.loc[melhor, 'brier'],
            'brier_referencia': m.loc['Referência (prevalência)', 'brier'],
            'auc_roc_baseline_fefo': m.loc['Baseline FEFO', 'auc_roc']}


def main(rapido: bool):
    cfg = dict(k=3, n_iter=4, amostra=10000, repeticoes=1) if rapido else \
          dict(k=10, n_iter=15, amostra=30000, repeticoes=3)
    os.makedirs(PASTA, exist_ok=True)
    inicio = time.time()
    print(f"Modo: {'RÁPIDO (só teste)' if rapido else 'COMPLETO'} | configuração: {cfg}")

    # -------------------------------------------------------------------------
    titulo('ETAPA 1: Dados e diagnóstico')
    df_real = criar_features(carregar_dataset(CAMINHO_DADOS))
    resumir_dataset(df_real, PASTA)
    diagnosticar(df_real, PASTA)
    df_sint = criar_features(gerar_sinteticos(df_real))
    comparar_com_reais(df_real, df_sint, PASTA)

    feat_completo = listar_features(incluir_condicao=True)   # só dados reais
    feat_estoque = listar_features(incluir_condicao=False)   # comuns a reais e sintéticos
    resumo = []

    # -------------------------------------------------------------------------
    titulo('CENÁRIO A (principal): treino e teste com dados reais')
    treino_a, teste_a = dividir_por_produto(df_real)
    hp_a = ajustar_hiperparametros(treino_a, feat_completo, cfg['n_iter'], amostra_max=cfg['amostra'],
                                   usar_smote=True, pasta_saida=PASTA)
    hp_a_sem = ajustar_hiperparametros(treino_a, feat_completo, cfg['n_iter'],
                                       amostra_max=cfg['amostra'], usar_smote=False,
                                       pasta_saida=PASTA)
    res_a = validar_e_treinar(treino_a, feat_completo, cfg['k'], hiperparametros=hp_a)
    res_a_sem = validar_e_treinar(treino_a, feat_completo, cfg['k'], usar_smote=False,
                                  hiperparametros=hp_a_sem)
    res_a['tabela_cv'].to_csv(f'{PASTA}/validacao_cruzada_A.csv', index=False)
    res_a_sem['tabela_cv'].to_csv(f'{PASTA}/validacao_cruzada_A_sem_smote.csv', index=False)

    melhor_a = escolher_melhor(res_a)
    print(f"\nMelhor modelo no cenário A (pela validação cruzada): {melhor_a}")
    tab_a, probs_a = avaliar_teste(res_a, teste_a, 'A', PASTA)
    avaliar_teste(res_a_sem, teste_a, 'A_sem_smote', PASTA)
    gerar_graficos(teste_a, probs_a, 'A', PASTA)
    comparar_smote(res_a, res_a_sem, melhor_a, teste_a, 'A', PASTA)
    matriz_confusao(res_a, teste_a, melhor_a, 'A', PASTA)
    explicar(res_a, melhor_a, teste_a, 'A', PASTA)
    resumo.append(linha_resumo('A', 'Reais (estoque + condição)', tab_a, melhor_a))

    # -------------------------------------------------------------------------
    titulo('CENÁRIO A-estoque: dados reais só com variáveis de estoque')
    # Referência justa para o cenário C, que não tem as variáveis de condição
    res_ae = validar_e_treinar(treino_a, feat_estoque, cfg['k'], hiperparametros=hp_a)
    melhor_ae = escolher_melhor(res_ae)
    tab_ae, _ = avaliar_teste(res_ae, teste_a, 'A_estoque', PASTA)
    resumo.append(linha_resumo('A-estoque', 'Reais (só estoque)', tab_ae, melhor_ae))

    # -------------------------------------------------------------------------
    titulo('CENÁRIO A-sem-FEFO: dados reais sem as variáveis que reproduzem o baseline')
    # O giro de estoque e o risco acumulado reproduzem a razão usada pela regra FEFO.
    # Sem eles, a comparação com o baseline mede só o ganho das demais variáveis.
    feat_sem_fefo = listar_features(incluir_condicao=True, sem_sobreposicao_fefo=True)
    res_sf = validar_e_treinar(treino_a, feat_sem_fefo, cfg['k'], hiperparametros=hp_a)
    melhor_sf = escolher_melhor(res_sf)
    tab_sf, _ = avaliar_teste(res_sf, teste_a, 'A_sem_fefo', PASTA)
    resumo.append(linha_resumo('A-sem-FEFO', 'Reais, sem variáveis do baseline', tab_sf, melhor_sf))

    # -------------------------------------------------------------------------
    titulo('CENÁRIO A-temporal: treino com lotes antigos, teste com lotes recentes')
    # Tudo (ajuste de hiperparâmetros, validação cruzada e limiar) usa apenas os
    # lotes anteriores ao corte, para não usar informação do "futuro".
    treino_t, teste_t, _ = dividir_por_tempo(df_real)
    hp_t = ajustar_hiperparametros(treino_t, feat_completo, cfg['n_iter'],
                                   amostra_max=cfg['amostra'], pasta_saida=f'{PASTA}/temporal')
    res_t = validar_e_treinar(treino_t, feat_completo, cfg['k'], hiperparametros=hp_t)
    melhor_t = escolher_melhor(res_t)
    tab_t, _ = avaliar_teste(res_t, teste_t, 'A_temporal', PASTA)
    resumo.append(linha_resumo('A-temporal', 'Reais, teste em período posterior', tab_t, melhor_t))
    # Subconjunto do teste só com produtos que não aparecem no treino
    novos = teste_t[~teste_t['product_id'].isin(treino_t['product_id'])]
    print(f"[pipeline] Teste temporal só com produtos novos: {len(novos)} lotes")
    tab_tn, _ = avaliar_teste(res_t, novos, 'A_temporal_produtos_novos', PASTA)
    resumo.append(linha_resumo('A-temporal (produtos novos)',
                               'Reais, período posterior, só produtos novos', tab_tn, melhor_t))

    # -------------------------------------------------------------------------
    titulo('CENÁRIO B (controle): treino e teste com dados sintéticos')
    treino_b, teste_b = dividir_por_produto(df_sint)
    hp_b = ajustar_hiperparametros(treino_b, feat_estoque, cfg['n_iter'], k=3,
                                   amostra_max=cfg['amostra'], pasta_saida=f'{PASTA}/B')
    res_b = validar_e_treinar(treino_b, feat_estoque, cfg['k'], hiperparametros=hp_b)
    res_b['tabela_cv'].to_csv(f'{PASTA}/validacao_cruzada_B.csv', index=False)
    melhor_b = escolher_melhor(res_b)
    tab_b, probs_b = avaliar_teste(res_b, teste_b, 'B', PASTA)
    gerar_graficos(teste_b, probs_b, 'B', PASTA)
    matriz_confusao(res_b, teste_b, melhor_b, 'B', PASTA)
    imp_b = explicar(res_b, melhor_b, teste_b, 'B', PASTA)
    verificar_mecanismo(imp_b, 'B', PASTA)
    resumo.append(linha_resumo('B', 'Sintéticos (mecanismo conhecido)', tab_b, melhor_b))

    # -------------------------------------------------------------------------
    titulo('CENÁRIO C (complemento): treino com reais + sintéticos, teste só com reais')
    treino_c = combinar(treino_a, df_sint)   # mesmos produtos reais de treino do cenário A
    res_c = validar_e_treinar(treino_c, feat_estoque, cfg['k'], hiperparametros=hp_a)
    melhor_c = escolher_melhor(res_c)
    tab_c, _ = avaliar_teste(res_c, teste_a, 'C', PASTA)   # mesmo teste real do cenário A
    resumo.append(linha_resumo('C', 'Reais + sintéticos (teste só reais)', tab_c, melhor_c))

    # -------------------------------------------------------------------------
    titulo('ANÁLISE DE SENSIBILIDADE (dados sintéticos)')
    sens = rodar_sensibilidade(df_real, repeticoes=cfg['repeticoes'], hiperparametros=hp_b,
                               pasta_saida=PASTA)
    resumir_sensibilidade(sens, PASTA)
    grafico_sensibilidade(sens, PASTA)

    # -------------------------------------------------------------------------
    titulo('TABELA-RESUMO DOS CENÁRIOS')
    tabela_resumo = pd.DataFrame(resumo)
    tabela_resumo.to_csv(f'{PASTA}/resumo_cenarios.csv', index=False)
    print(tabela_resumo.to_string(index=False))

    # -------------------------------------------------------------------------
    titulo('SALVANDO O MODELO FINAL (usado pela interface)')
    joblib.dump({
        'modelo': res_a['modelos'][melhor_a],
        'nome_modelo': melhor_a,
        'features': feat_completo,
        'prevalencia': res_a['prevalencia'],
        'usar_smote': res_a['usar_smote'],
        'limiar': escolher_limiar(res_a['y_treino'], res_a['oof'][melhor_a]),
    }, f'{PASTA}/modelo_final.joblib')
    # Lotes de exemplo para demonstrar a interface (do conjunto de teste, sem o rótulo)
    salvar_lotes_exemplo(teste_a, f'{PASTA}/lotes_exemplo.csv')
    print(f"Salvos: {PASTA}/modelo_final.joblib e {PASTA}/lotes_exemplo.csv")

    print(f"\nConcluído em {(time.time() - inicio) / 60:.1f} minutos. Resultados em {PASTA}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rapido', action='store_true',
                        help='execução reduzida, só para verificar se tudo funciona')
    main(parser.parse_args().rapido)