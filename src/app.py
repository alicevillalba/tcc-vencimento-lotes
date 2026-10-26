# =============================================================================
# Interface web: priorização de lotes por risco de vencimento
#
# Uso (dentro da pasta do repositório, depois de rodar o pipeline):
#   streamlit run src/app.py
# =============================================================================

import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import streamlit as st

sys.path.append(os.path.dirname(__file__))  # permite importar os módulos de src/
from features import criar_features, CATEGORIAS
from train import prever_probabilidade

CAMINHO_MODELO = 'outputs/modelo_final.joblib'
CAMINHO_EXEMPLO = 'outputs/lotes_exemplo.csv'

# Nomes amigáveis para as variáveis exibidas na tela
NOMES = {
    'days_until_expiry': 'Dias até o vencimento',
    'daily_demand': 'Demanda diária',
    'initial_quantity': 'Quantidade do lote',
    'demand_variability': 'Variabilidade da demanda',
    'spoilage_sensitivity': 'Sensibilidade à deterioração',
    'giro_estoque': 'Giro de estoque (dias)',
    'urgencia_vencimento': 'Urgência de vencimento',
    'risco_acumulado': 'Risco acumulado',
    'category_enc': 'Categoria',
    'storage_temp': 'Temperatura de armazenagem',
    'temp_deviation': 'Desvio de temperatura',
    'temp_abuse_events': 'Eventos de temperatura',
    'distribution_hours': 'Horas de distribuição',
    'handling_score': 'Nota de manuseio',
    'packaging_score': 'Nota de embalagem',
    'supplier_score': 'Nota do fornecedor',
    'is_promoted': 'Em promoção',
}

# Colunas que o usuário precisa informar (as derivadas são calculadas)
COLUNAS_ENTRADA = ['product_id', 'category', 'shelf_life_days', 'days_until_expiry',
                   'daily_demand', 'initial_quantity', 'demand_variability',
                   'spoilage_sensitivity', 'storage_temp', 'temp_deviation',
                   'temp_abuse_events', 'distribution_hours', 'handling_score',
                   'packaging_score', 'supplier_score', 'is_promoted']


LEGENDA_SHAP = ('Barras vermelhas aumentam o risco do lote; azuis diminuem. Os valores à '
                'esquerda são os dados do lote. O eixo está na escala interna do modelo '
                '(antes da correção de probabilidade); o sentido e o peso relativo de cada '
                'fator são os mesmos.')


# -----------------------------------------------------------------------------
# Carregamento (feito uma vez e guardado em cache)
# -----------------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


@st.cache_data
def carregar_exemplo():
    return pd.read_csv(CAMINHO_EXEMPLO)


def prever(pacote, lotes: pd.DataFrame) -> pd.DataFrame:
    df = criar_features(lotes)
    df['probabilidade'] = prever_probabilidade(pacote['modelo'], df[pacote['features']],
                                               pacote['prevalencia'], pacote['usar_smote'])
    df['situacao'] = np.where(df['probabilidade'] >= pacote['limiar'], 'Em risco', 'Normal')
    return df.sort_values('probabilidade', ascending=False).reset_index(drop=True)


def explicacao_shap(pacote, df: pd.DataFrame, indice: int, fundo: pd.DataFrame):
    """Valores SHAP de um lote, usando os lotes carregados como referência."""
    pipeline = pacote['modelo']
    modelo = pipeline.named_steps['modelo']
    features = pacote['features']

    def preparar(X):
        if 'padronizacao' in pipeline.named_steps:
            return pipeline.named_steps['padronizacao'].transform(X)
        return X.values.astype(float)

    X_fundo = preparar(criar_features(fundo)[features])
    X_lote = preparar(df.loc[[indice], features])
    if pacote['nome_modelo'] == 'Regressão Logística':
        valores = shap.LinearExplainer(modelo, X_fundo)(X_lote)
    else:
        valores = shap.TreeExplainer(modelo)(X_lote)
        if valores.values.ndim == 3:
            valores = valores[:, :, 1]
    return shap.Explanation(values=valores.values[0], base_values=valores.base_values[0],
                            data=df.loc[indice, features].values.astype(float),
                            feature_names=[NOMES.get(f, f) for f in features])


def formatar(valor) -> str:
    """Número no formato brasileiro: inteiro sem casas, decimal com vírgula."""
    valor = float(valor)
    if valor.is_integer():
        return f'{valor:.0f}'
    return f'{valor:.2f}'.replace('.', ',')


def pct(valor: float) -> str:
    """Percentual com uma casa decimal e vírgula (ex.: 0.3597 -> '36,0%')."""
    return f'{valor * 100:.1f}%'.replace('.', ',')


def descrever_valor(feature: str, valor: float, media: float) -> str:
    """Valor do lote comparado com a média dos lotes de referência.
    A média é o ponto de partida do SHAP: cada fator empurra o risco para cima ou
    para baixo conforme o valor do lote se afasta dela."""
    if feature == 'category_enc':
        return CATEGORIAS[int(valor)] if 0 <= valor < len(CATEGORIAS) else 'desconhecida'
    if feature == 'is_promoted':
        return 'sim' if valor else 'não'
    if valor > media:
        comparacao = f'acima da média ({formatar(media)})'
    elif valor < media:
        comparacao = f'abaixo da média ({formatar(media)})'
    else:
        comparacao = 'igual à média'
    return f'{formatar(valor)}, {comparacao}'


def texto_explicacao(pacote, prob: float, situacao: str, explicacao,
                     features: list, medias: pd.Series, n: int = 3) -> str:
    """Resumo do risco + fatores que mais aumentam e mais reduzem o risco do lote."""
    media = pacote['prevalencia']
    razao = prob / media
    if razao >= 1.1:
        vezes = f'{razao:.1f}'.replace('.', ',')
        comparacao = f"cerca de **{vezes} {'vezes' if razao >= 2 else 'vez'}** o risco médio do estoque"
    elif razao <= 0.9:
        comparacao = 'abaixo do risco médio do estoque'
    else:
        comparacao = 'próximo do risco médio do estoque'
    linhas = [f'**Resumo:** probabilidade de vencer de **{pct(prob)}**, '
              f'{comparacao} ({pct(media)}). Situação: **{situacao}**.']

    fatores = sorted(zip(features, explicacao.feature_names, explicacao.data, explicacao.values),
                     key=lambda f: abs(f[3]), reverse=True)
    # Ignora fatores com peso desprezível (menos de 10% do fator mais forte)
    corte = 0.1 * abs(fatores[0][3])
    fatores = [f for f in fatores if abs(f[3]) >= corte]

    for titulo, grupo in [('Fatores que **aumentam** o risco', [f for f in fatores if f[3] > 0]),
                          ('Fatores que **reduzem** o risco', [f for f in fatores if f[3] < 0])]:
        if grupo:
            linhas.append(f'{titulo}:')
            linhas.append('\n'.join(
                f'- **{nome}**: {descrever_valor(feat, valor, medias[feat])}'
                for feat, nome, valor, _ in grupo[:n]))
    return '\n\n'.join(linhas)


def mostrar_explicacao(pacote, df: pd.DataFrame, indice: int, fundo: pd.DataFrame,
                       dados_informados: dict = None):
    features = pacote['features']
    explicacao = explicacao_shap(pacote, df, indice, fundo)
    medias = criar_features(fundo)[features].mean()
    if dados_informados:
        # Dados do lote simulado numa faixa compacta, para aparecerem no mesmo print
        itens = [f'{campo}: <b>{formatar(v) if isinstance(v, (int, float)) else v}</b>'
                 for campo, v in dados_informados.items()]
        st.markdown('<div style="font-size:0.85rem; line-height:1.6">'
                    '<b>Dados informados</b> · ' + ' · '.join(itens) + '</div>',
                    unsafe_allow_html=True)
        st.write('')
    # Texto à esquerda e gráfico à direita, com tamanho fixo: assim o resultado
    # cabe numa tela só e o gráfico não cresce quando o zoom do navegador muda
    col_texto, col_grafico = st.columns([1, 1.3], gap='large')
    col_texto.markdown(texto_explicacao(pacote, df.loc[indice, 'probabilidade'],
                                        df.loc[indice, 'situacao'], explicacao, features, medias))
    col_texto.caption(LEGENDA_SHAP + ' "Média" = média dos lotes de referência.')

    col_grafico.pyplot(grafico_lote(explicacao), use_container_width=False)


def grafico_lote(explicacao):
    fig = plt.figure(dpi=90)
    shap.plots.waterfall(explicacao, max_display=10, show=False)
    plt.xlabel('Contribuição para o risco (escala log-odds)')
    fig = plt.gcf()
    fig.set_size_inches(7, 4.5)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Tela
# -----------------------------------------------------------------------------

st.set_page_config(page_title='Risco de vencimento de lotes', layout='wide')

if not os.path.exists(CAMINHO_MODELO):
    st.error('Modelo não encontrado. Rode antes: python src/pipeline.py')
    st.stop()

pacote = carregar_modelo()
exemplo = carregar_exemplo()

st.title('Risco de vencimento de lotes')
st.caption(f"Modelo: {pacote['nome_modelo']} · lotes com probabilidade a partir de "
           f"{pacote['limiar']*100:.1f}% são marcados como **em risco**. "
           "As probabilidades apoiam a priorização da inspeção; não substituem a análise do "
           "responsável pelo estoque.")

aba_lotes, aba_manual = st.tabs(['Lotes do estoque', 'Simular um lote'])

# --- Aba 1: lista de lotes ----------------------------------------------------
with aba_lotes:
    arquivo = st.file_uploader('Envie uma planilha CSV de lotes (ou use os lotes de exemplo)',
                               type='csv')
    lotes = pd.read_csv(arquivo) if arquivo else exemplo
    faltando = [c for c in COLUNAS_ENTRADA if c not in lotes.columns]
    if faltando:
        st.error(f"Colunas ausentes na planilha: {', '.join(faltando)}")
        st.stop()

    resultado = prever(pacote, lotes)

    # Filtro por categoria (padrão: farmacêuticos, quando houver)
    disponiveis = sorted(resultado['category'].unique())
    padrao = ['Pharmaceuticals'] if 'Pharmaceuticals' in disponiveis else disponiveis
    categorias = st.multiselect('Categorias', disponiveis, default=padrao)
    resultado = resultado[resultado['category'].isin(categorias)].reset_index(drop=True)
    if resultado.empty:
        st.info('Selecione ao menos uma categoria.')
        st.stop()

    em_risco = resultado['situacao'].eq('Em risco')

    c1, c2, c3 = st.columns(3)
    c1.metric('Lotes analisados', f'{len(resultado)}')
    c2.metric('Lotes em risco', f'{em_risco.sum()} ({em_risco.mean()*100:.0f}%)')
    c3.metric('Probabilidade média', f"{resultado['probabilidade'].mean()*100:.1f}%")

    so_risco = st.toggle('Mostrar só lotes em risco', value=False)
    tabela = resultado[em_risco] if so_risco else resultado
    tabela_exibida = pd.DataFrame({
        'Lote': tabela['product_id'],
        'Categoria': tabela['category'],
        'Probabilidade de vencer': tabela['probabilidade'],
        'Situação': tabela['situacao'],
        'Dias até vencer': tabela['days_until_expiry'],
        'Quantidade': tabela['initial_quantity'],
        'Demanda diária': tabela['daily_demand'],
    })
    st.dataframe(
        tabela_exibida, hide_index=True, use_container_width=True, height=320,
        column_config={'Probabilidade de vencer': st.column_config.ProgressColumn(
            format='percent', min_value=0.0, max_value=1.0)},
    )

    st.subheader('Por que este lote tem esse risco?')
    escolhido = st.selectbox('Escolha um lote', tabela['product_id'].tolist())
    indice = resultado.index[resultado['product_id'] == escolhido][0]
    mostrar_explicacao(pacote, resultado, indice, lotes)

# --- Aba 2: simulação de um lote ----------------------------------------------
with aba_manual:
    st.write('Informe os dados de um lote (abaixo) para estimar o risco de vencimento.')
    # O resultado aparece ACIMA do formulário, para caber numa tela só
    area_resultado = st.container()
    base = exemplo.median(numeric_only=True)
    with st.form('lote_manual'):
        a, b, c = st.columns(3)
        categoria = a.selectbox('Categoria', CATEGORIAS, index=CATEGORIAS.index('Pharmaceuticals'))
        vida_util = a.number_input('Vida útil total (dias)', 1, 2000, int(base['shelf_life_days']))
        dias = a.number_input('Dias até o vencimento', 0, 2000, int(base['days_until_expiry']))
        quantidade = a.number_input('Quantidade do lote', 1, 100000, int(base['initial_quantity']))
        demanda = b.number_input('Demanda diária', 0.0, 10000.0, float(base['daily_demand']))
        variab = b.slider('Variabilidade da demanda', 0.0, 1.0, float(base['demand_variability']))
        sensib = b.slider('Sensibilidade à deterioração', 0.0, 1.0,
                          float(base['spoilage_sensitivity']))
        promo = b.checkbox('Em promoção')
        temp = c.number_input('Temperatura de armazenagem', -30.0, 40.0, float(base['storage_temp']))
        desvio = c.number_input('Desvio de temperatura', 0.0, 20.0, float(base['temp_deviation']))
        eventos = c.number_input('Eventos de temperatura', 0, 50, int(base['temp_abuse_events']))
        horas = c.number_input('Horas de distribuição', 0.0, 500.0,
                               float(base['distribution_hours']))
        manuseio = a.slider('Nota de manuseio (0 a 10)', 0.0, 10.0, float(base['handling_score']), 0.5)
        embalagem = b.slider('Nota de embalagem (0 a 10)', 0.0, 10.0, float(base['packaging_score']), 0.5)
        fornecedor = c.slider('Nota do fornecedor (0 a 10)', 0.0, 10.0, float(base['supplier_score']), 0.5)
        enviar = st.form_submit_button('Calcular risco', type='primary')

    if enviar:
        lote = pd.DataFrame([{
            'product_id': 'SIMULADO', 'category': categoria, 'shelf_life_days': vida_util,
            'days_until_expiry': dias, 'daily_demand': demanda, 'initial_quantity': quantidade,
            'demand_variability': variab, 'spoilage_sensitivity': sensib,
            'storage_temp': temp, 'temp_deviation': desvio, 'temp_abuse_events': eventos,
            'distribution_hours': horas, 'handling_score': manuseio,
            'packaging_score': embalagem, 'supplier_score': fornecedor, 'is_promoted': int(promo),
        }])
        res = prever(pacote, lote)
        dados = {
            'Categoria': categoria, 'Vida útil (dias)': vida_util,
            'Dias até vencer': dias, 'Quantidade': quantidade, 'Demanda diária': demanda,
            'Variabilidade da demanda': variab, 'Sensibilidade': sensib,
            'Em promoção': 'sim' if promo else 'não', 'Temperatura': temp,
            'Desvio de temperatura': desvio, 'Eventos de temperatura': eventos,
            'Horas de distribuição': horas, 'Nota de manuseio': manuseio,
            'Nota de embalagem': embalagem, 'Nota do fornecedor': fornecedor,
        }
        with area_resultado:
            mostrar_explicacao(pacote, res, 0, exemplo, dados_informados=dados)
            st.divider()