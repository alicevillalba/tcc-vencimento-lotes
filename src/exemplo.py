# =============================================================================
# Gera os lotes de exemplo usados pela interface (outputs/lotes_exemplo.csv).
# Os lotes vêm do conjunto de TESTE do cenário A: o modelo nunca os viu no treino.
#
# Rodar sozinho (alguns segundos, sem treinar nada):
#   python src/exemplo.py
# =============================================================================

import pandas as pd

from data_loader import carregar_dataset
from features import criar_features, FEATURES_ESTOQUE, FEATURES_CONDICAO
from train import dividir_por_produto

COLUNAS_EXEMPLO = ['product_id', 'category', 'shelf_life_days'] + FEATURES_ESTOQUE + FEATURES_CONDICAO


def salvar_lotes_exemplo(teste: pd.DataFrame, caminho: str = 'outputs/lotes_exemplo.csv',
                         n_farma: int = 200, n_outros: int = 100, semente: int = 42):
    """Sorteia lotes farmacêuticos (foco do trabalho) e alguns de outras categorias."""
    farma = teste[teste['category'] == 'Pharmaceuticals']
    outros = teste[teste['category'] != 'Pharmaceuticals']
    amostra = pd.concat([
        farma.sample(min(n_farma, len(farma)), random_state=semente),
        outros.sample(min(n_outros, len(outros)), random_state=semente),
    ])
    amostra[COLUNAS_EXEMPLO].to_csv(caminho, index=False)
    print(f"Lotes de exemplo salvos em {caminho}: {len(amostra)} lotes "
          f"({min(n_farma, len(farma))} farmacêuticos)")


if __name__ == '__main__':
    # Refaz a mesma divisão treino/teste do pipeline (mesma semente)
    df_real = criar_features(carregar_dataset('data/perishable_goods_management.csv'))
    _, teste_a = dividir_por_produto(df_real)
    salvar_lotes_exemplo(teste_a)