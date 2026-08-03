import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PARTE A — PREPARAÇÃO DOS DADOS (a partir dos brutos)
# ================================================================

def preparar_base():
    """Carrega brutos, limpa duplicatas, calcula erro e constrói TODAS as features."""
    df = pd.read_excel(r'C:\Users\Games\OneDrive\Desktop\osfstorage-archive\Features_2018\base_todas_long.xlsx')          # base bruta long (id = registro TSE p/ 2018)
    feat = pd.read_excel(r'C:\Users\Games\OneDrive\Desktop\osfstorage-archive\Features_2018\2018_pesquisas_features_falta_merge2.0.xlsx')        # planilha de features
    df['id_pesquisa'] = df['id_pesquisa'].astype(str).str.replace('\u2010', '-')
    feat['ID'] = feat['ID'].astype(str).str.replace('\u2010', '-')

    b = df[df['ano'] == 2018].copy()
    b['pct'] = b['percentual'].apply(lambda x: x*100 if x <= 1 else x)

    # --- limpeza das 2 pesquisas duplicadas ---
    rem = (((b['id_pesquisa']=='BR-01584/2018') & (b['quantidade_entrevistas']!=19552)) |
           ((b['id_pesquisa']=='BR-02410/2018') & (b['quantidade_entrevistas']!=2010)) |
           (b['id_pesquisa']=="BR-06298/2018") |
           (b['id_pesquisa']=="BR-0446/2018")
           
           
           
           
           
           )
    b = b[~rem].copy()

    # --- erro Método 2 (votos válidos), por registro+turno ---
    real = {1:{'bolsonaro':46.03,'haddad':29.28,'gomes':12.47,'alckmin':4.76,
               'amoedo':2.50,'meirelles':1.20,'silva':1.00,'dias':0.80},
            2:{'bolsonaro':55.13,'haddad':44.87}}
    b['soma_rt'] = b.groupby(['id_pesquisa','turno'])['pct'].transform('sum')
    b['percentual_valido'] = b['pct']/b['soma_rt']*100
    b['percentual_real'] = b.apply(lambda r: real.get(r['turno'],{}).get(r['nome_clean']), axis=1)
    b['vies'] = b['percentual_valido'] - b['percentual_real']
    b['abs_vies'] = b['vies'].abs()              # <<< VARIÁVEL DEPENDENTE

    # --- features temporais (CORRIGIDAS a partir de dias_ate_eleicao) ---
    d = b['dias_ate_eleicao']
    b['is_vespera']       = ((d>=0)&(d<=3)).astype(int)
    b['is_ultima_semana'] = ((d>=4)&(d<=7)).astype(int)
    b['is_15dias']        = ((d>=8)&(d<=15)).astype(int)
    b['is_30dias']        = ((d>=16)&(d<=30)).astype(int)
    b['is_60dias']        = ((d>=31)&(d<=60)).astype(int)
    # faixa categórica (multi-janela)
    b['faixa_dias'] = pd.cut(d, [-1,0,5,10,15,20,9999],
                             labels=['0','1-5','6-10','11-15','16-20','>20'])

    # --- features de amostra / precisão ---
    b['log_amostra'] = np.log(b['quantidade_entrevistas'])
    b['var_phat'] = (b['pct']/100)*(1-b['pct']/100)/b['quantidade_entrevistas'] #Variância amostral da estimativa do candidato - Já contém tamanho amostra!
    b['faixa_amostra'] = pd.cut(b['quantidade_entrevistas'], [0,500,1000,2000,999999],
                                labels=['ate500','501-1000','1001-2000','>2000'])

    # --- indecisos ---
    b['prop_indecisos'] = (100 - b['soma_rt']).clip(lower=0)
    b['faixa_indecisos'] = pd.cut(b['prop_indecisos'], [-1,1,5,10,100],
                                  labels=['0-1%','1-5%','5-10%','>10%'])

    # --- características da disputa ---
    b['cand_grande'] = (b['pct']>20).astype(int)
    marg = b.groupby(['id_pesquisa','turno']).apply(
        lambda g: (lambda v: v[0]-v[1] if len(v)>=2 else np.nan)(g.nlargest(2,'pct')['pct'].values)
    ).rename('competitividade')
    b = b.merge(marg, on=['id_pesquisa','turno'], how='left')

    # --- merge metodológicas da planilha ---
    fcols = ['ID','telefonica','Final_de_Semana','Dias_Campo','Auditoria=30',
             'conglomerados_3estagios','conglomerados_2estagios','Valor(R$)']
    b = b.merge(feat[fcols], left_on='id_pesquisa', right_on='ID', how='left')
    
    b['valor_pesquisa'] = pd.to_numeric(b['Valor(R$)'], errors='coerce')
    b['valor_pesquisa'] = b['valor_pesquisa'].fillna(b['valor_pesquisa'].median())
    b = b.rename(columns={'Final_de_Semana':'final_de_semana','Dias_Campo':'dias_campo',
                          'Auditoria=30':'auditoria_30'})
    for c in ['final_de_semana','dias_campo','auditoria_30','telefonica',
              'conglomerados_2estagios','conglomerados_3estagios']:
        b[c] = pd.to_numeric(b[c], errors='coerce').fillna(0)

    # --- (AJUSTE 1) remove linhas com NA nas variaveis usadas na modelagem ---
    #     inclui os nanicos sem 'real' (abs_vies NaN), que antes ficavam na base.
    COLS_MODELO = ['abs_vies', 'instituto', 'dias_ate_eleicao', 'is_vespera',
                   'is_ultima_semana', 'is_15dias', 'is_30dias', 'is_60dias',
                   'var_phat', 'log_amostra', 'prop_indecisos', 'cand_grande',
                   'competitividade', 'telefonica', 'final_de_semana', 'dias_campo',
                   'auditoria_30', 'valor_pesquisa']
    n_antes = len(b)
    b = b.dropna(subset=COLS_MODELO).copy()
    print(f"[NA] removidas {n_antes - len(b)} linhas com NA (restam {len(b)})")
    return b

# ================================================================
# PARTE B — MOTOR DE AVALIAÇÃO
# Você normalmente NÃO precisa mexer aqui.
# ================================================================

def montar_design(b, features, usar_instituto=True, usar_turno=True):
    """Monta a matriz X com as features escolhidas + controles."""
    partes = []
    # separa numéricas de categóricas (categóricas viram dummies)
    cat = ['faixa_dias','faixa_amostra','faixa_indecisos']
    num_feats = [f for f in features if f not in cat]
    cat_feats = [f for f in features if f in cat]
    if num_feats:
        partes.append(b[num_feats].apply(pd.to_numeric, errors='coerce').fillna(0))
    for c in cat_feats:
        partes.append(pd.get_dummies(b[c], prefix=c, drop_first=True).astype(float))
    if usar_instituto:
        partes.append(pd.get_dummies(b['instituto'], prefix='inst', drop_first=True).astype(float))
    if usar_turno:
        partes.append(pd.get_dummies(b['turno'], prefix='turno', drop_first=True).astype(float))
    X = pd.concat(partes, axis=1).astype(float)
    return X

def avaliar(b, features, nome, usar_instituto=True, usar_turno=True):
    """Ajusta o modelo e retorna métricas: R2, R2-adj, AIC, BIC, RMSE validação cruzada."""
    y = b['abs_vies']
    X = montar_design(b, features, usar_instituto, usar_turno)
    m = sm.OLS(y, sm.add_constant(X)).fit()
    kf = KFold(5, shuffle=True, random_state=42)
    rmse = -cross_val_score(LinearRegression(), X.values, y, cv=kf,
                            scoring='neg_root_mean_squared_error').mean()
    return {'modelo': nome, 'n_features': len(features), 'n_params': int(m.df_model),
            'R2': round(m.rsquared,3), 'R2_adj': round(m.rsquared_adj,3),
            'AIC': round(m.aic,1), 'BIC': round(m.bic,1), 'RMSE_cv': round(rmse,3)}

def vif(b, features):
    """Calcula VIF das features numéricas (diagnóstico de multicolinearidade)."""
    cat = ['faixa_dias','faixa_amostra','faixa_indecisos']
    num = [f for f in features if f not in cat]
    if len(num) < 2:
        return pd.DataFrame()
    X = sm.add_constant(b[num].apply(pd.to_numeric, errors='coerce').fillna(0))
    rows = []
    for i,c in enumerate(X.columns):
        if c=='const': continue
        rows.append({'feature':c, 'VIF':round(variance_inflation_factor(X.values,i),2)})
    return pd.DataFrame(rows)

# ----------------------------------------------------------------
# >>> ACRÉSCIMO: REGRESSÃO DETALHADA (coef, erro-padrão, t, p, IC, signif.)
# Devolve a tabela completa de coeficientes de um modelo + as
# estatísticas globais (R2, R2-adj, F, AIC, BIC, N).
# Por padrão, só das FEATURES de interesse (oculta as dummies de
# instituto); use incluir_controles=True para ver tudo.
# ----------------------------------------------------------------
def regressao_detalhada(b, features, incluir_controles=False):
    y = b['abs_vies']
    X = sm.add_constant(montar_design(b, features))
    m = sm.OLS(y, X).fit()

    def estrelas(p):
        return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))

    # quais linhas mostrar: const + features de interesse (e controles se pedido)
    cat = ['faixa_dias','faixa_amostra','faixa_indecisos']
    nomes_interesse = ['const']
    for f in features:
        if f in cat:
            nomes_interesse += [c for c in m.params.index if c.startswith(f + '_')]
        else:
            nomes_interesse.append(f)
    if incluir_controles:
        mostrar = list(m.params.index)
    else:
        mostrar = [c for c in m.params.index if c in nomes_interesse]

    ic = m.conf_int(alpha=0.05)
    linhas = []
    for c in mostrar:
        linhas.append({
            'variavel': c,
            'coef': round(m.params[c], 4),
            'erro_padrao': round(m.bse[c], 4),
            'estat_t': round(m.tvalues[c], 3),
            'p_valor': round(m.pvalues[c], 4),
            'signif': estrelas(m.pvalues[c]),
            'IC_2.5%': round(ic.loc[c, 0], 4),
            'IC_97.5%': round(ic.loc[c, 1], 4),
        })
    tab_coef = pd.DataFrame(linhas)

    estat = pd.DataFrame([
        {'estatistica':'N (observações)', 'valor': int(m.nobs)},
        {'estatistica':'R²',              'valor': round(m.rsquared, 4)},
        {'estatistica':'R²-ajustado',     'valor': round(m.rsquared_adj, 4)},
        {'estatistica':'F',               'valor': round(m.fvalue, 2)},
        {'estatistica':'p-valor (F)',     'valor': round(m.f_pvalue, 6)},
        {'estatistica':'AIC',             'valor': round(m.aic, 1)},
        {'estatistica':'BIC',             'valor': round(m.bic, 1)},
        {'estatistica':'nº parâmetros',   'valor': int(m.df_model)},
    ])
    return tab_coef, estat

# ----------------------------------------------------------------
# >>> ACRÉSCIMO: SELEÇÃO POR BUSCA (forward/backward/stepwise) x (AIC/BIC)
# Seleciona DENTRE as features do modelo informado. instituto e turno
# permanecem como controle fixo (não entram na seleção).
# ----------------------------------------------------------------
def _criterio_modelo(b, feats, crit):
    """AIC ou BIC do OLS com as feats dadas + controles (instituto+turno)."""
    y = b['abs_vies']
    X = sm.add_constant(montar_design(b, feats))
    m = sm.OLS(y, X).fit()
    return m.aic if crit == 'aic' else m.bic

def _forward(b, candidatas, crit):
    sel, rest = [], list(candidatas)
    melhor = _criterio_modelo(b, sel, crit)
    mudou = True
    while mudou and rest:
        mudou = False
        scores = sorted((_criterio_modelo(b, sel+[c], crit), c) for c in rest)
        if scores[0][0] < melhor:
            melhor = scores[0][0]; sel.append(scores[0][1]); rest.remove(scores[0][1]); mudou = True
    return sel

def _backward(b, candidatas, crit):
    sel = list(candidatas)
    melhor = _criterio_modelo(b, sel, crit)
    mudou = True
    while mudou and len(sel) > 0:
        mudou = False
        scores = sorted((_criterio_modelo(b, [x for x in sel if x != c], crit), c) for c in sel)
        if scores[0][0] < melhor:
            melhor = scores[0][0]; sel.remove(scores[0][1]); mudou = True
    return sel

def _stepwise(b, candidatas, crit):
    sel, rest = [], list(candidatas)
    melhor = _criterio_modelo(b, sel, crit)
    mudou = True
    while mudou:
        mudou = False
        movs = []
        for c in rest:
            movs.append((_criterio_modelo(b, sel+[c], crit), 'add', c))
        for c in sel:
            movs.append((_criterio_modelo(b, [x for x in sel if x != c], crit), 'rem', c))
        if not movs: break
        movs.sort(key=lambda t: t[0])
        if movs[0][0] < melhor:
            melhor, acao, var = movs[0]
            if acao == 'add':
                sel.append(var); rest.remove(var)
            else:
                sel.remove(var); rest.append(var)
            mudou = True
    return sel

def selecionar(b, features, nome):
    """Roda os 6 algoritmos (forward/backward/stepwise x AIC/BIC) sobre as
    features do modelo e devolve uma tabela: qual método manteve qual feature."""
    metodos = {
        'Forward_AIC' : _forward(b, features, 'aic'),
        'Forward_BIC' : _forward(b, features, 'bic'),
        'Backward_AIC': _backward(b, features, 'aic'),
        'Backward_BIC': _backward(b, features, 'bic'),
        'Stepwise_AIC': _stepwise(b, features, 'aic'),
        'Stepwise_BIC': _stepwise(b, features, 'bic'),
    }
    linhas = []
    for f in features:
        linha = {'feature': f}
        for met, sel in metodos.items():
            linha[met] = 1 if f in sel else 0
        linha['n_metodos'] = sum(linha[m] for m in metodos)
        linhas.append(linha)
    tab = pd.DataFrame(linhas).sort_values('n_metodos', ascending=False)
    # resumo: nº de variáveis selecionadas por método
    resumo = pd.DataFrame([{'metodo': m, 'n_selecionadas': len(s),
                            'features': ', '.join(s) if s else '(nenhuma)'}
                           for m, s in metodos.items()])
    return tab, resumo

# ================================================================
# PARTE C — PAINEL DE CONTROLE  <<<<< EDITE AQUI >>>>>
# É AQUI que você mexe para testar suas combinações.
# ================================================================

if __name__ == '__main__':
    b = preparar_base()
    print(f"Base preparada: {len(b)} linhas, {b['id_pesquisa'].nunique()} pesquisas\n")

    # -----------------------------------------------------------
    # TESTAR MODELOS

    # -----------------------------------------------------------
    MINHAS_COMBINACOES = {

        "Modelo3":
            ['dias_ate_eleicao', "is_30dias",'is_60dias','var_phat'],

        "Modelo5":
            ['dias_ate_eleicao', "is_30dias",'is_60dias','var_phat',"cand_grande","competitividade","log_amostra"],

        "Modelo1":
            ['dias_ate_eleicao','is_vespera','is_ultima_semana','is_15dias','is_30dias','is_60dias','var_phat'],

        "Modelo2":
            ['dias_ate_eleicao', "final_de_semana",'var_phat', "dias_campo","auditoria_30",'telefonica','prop_indecisos','valor_pesquisa'],

        "Modelo4":
            ['dias_ate_eleicao', "is_30dias",'is_60dias','var_phat', "log_amostra","competitividade","cand_grande", 'is_ultima_semana','is_15dias', 'is_vespera',"auditoria_30"]



    }

    # -----------------------------------------------------------
    # TESTAR TODAS CANDIDATAS

    # -----------------------------------------------------------
    POOL_SELECAO = [
        # --- temporais ---
        'dias_ate_eleicao',
        'is_vespera',
        'is_ultima_semana',
        'is_15dias',
        'is_30dias',
        'is_60dias',
        # --- amostra / precisão ---
        'var_phat',
        'log_amostra',
        # --- indecisos / disputa ---
        'prop_indecisos',
        'cand_grande',
        'competitividade',
        # --- metodológicas ---
        'telefonica',
        'final_de_semana',
        'dias_campo',
        'auditoria_30',
        #'conglomerados_3estagios',
        'valor_pesquisa',
    ]

    # -----------------------------------------------------------
    # CAMINHO 1 — MODELOS NA MÃO: só avaliam (NÃO passam pela seleção,
    # pois já são uma escolha teórica/humana proposital).
    # -----------------------------------------------------------
    resultados = []
    for nome, feats in MINHAS_COMBINACOES.items():
        resultados.append(avaliar(b, feats, nome))

    tabela = pd.DataFrame(resultados)
    print("\n" + "="*90)
    print("COMPARAÇÃO DE MODELOS  (R2_adj maior = melhor ajuste | RMSE_cv menor = melhor previsão)")
    print("="*90)
    print(tabela.to_string(index=False))

    # VIF do modelo final
    print("\n" + "="*50)
    print("VIF do primeiro modelo da lista (multicolinearidade)")
    print("="*50)
    primeiro = list(MINHAS_COMBINACOES.values())[0]
    print(vif(b, primeiro).to_string(index=False))

    # >>> ACRÉSCIMO: regressão detalhada do primeiro modelo <<<
    nome_primeiro = list(MINHAS_COMBINACOES.keys())[0]
    tab_coef, estat = regressao_detalhada(b, primeiro, incluir_controles=False)
    print("\n" + "="*70)
    print(f"REGRESSÃO DETALHADA — {nome_primeiro} (features de interesse)")
    print("Signif.: *** p<0.01 | ** p<0.05 | * p<0.1")
    print("="*70)
    print(tab_coef.to_string(index=False))
    print("\n>> Estatísticas globais:")
    print(estat.to_string(index=False))

    # Janela de previsibilidade (erro médio por janela)
    print("\n" + "="*50)
    print("JANELA DE PREVISIBILIDADE (erro médio por faixa de dias)")
    print("="*50)
    jan = b.groupby('faixa_dias')['abs_vies'].agg(['mean','count']).round(2)
    print(jan.to_string())

    # -----------------------------------------------------------
    # CAMINHO 2 — SELEÇÃO AUTOMÁTICA sobre o POOL completo.

    # -----------------------------------------------------------
    tab_sel, resumo_sel = selecionar(b, POOL_SELECAO, 'POOL')
    print("\n" + "#"*70)
    print("SELEÇÃO AUTOMÁTICA — a partir do POOL completo de features")
    print("#"*70)
    print(">> Resumo (quantas variáveis cada método manteve):")
    print(resumo_sel.to_string(index=False))
    print("\n>> Matriz feature x método (1 = mantida | n_metodos = robustez):")
    print(tab_sel.to_string(index=False))

    # Exportar
    with pd.ExcelWriter('resultados_meus_testes.xlsx', engine='openpyxl') as w:
        tabela.to_excel(w, sheet_name='comparacao_modelos', index=False)
        vif(b, primeiro).to_excel(w, sheet_name='vif', index=False)
        # >>> regressão detalhada do primeiro modelo <<<
        tab_coef.to_excel(w, sheet_name='regressao_coeficientes', index=False)
        estat.to_excel(w, sheet_name='regressao_estatisticas', index=False)
        jan.to_excel(w, sheet_name='janela_previsibilidade')
        # >>> seleção automática sobre o pool <<<
        tab_sel.to_excel(w, sheet_name='selecao_automatica', index=False)
        resumo_sel.to_excel(w, sheet_name='selecao_resumo', index=False)
        b.to_excel(w, sheet_name='base_completa', index=False)
    print("\nExportado: resultados_meus_testes.xlsx")
    print("  Abas: comparacao_modelos | regressao_coeficientes | regressao_estatisticas |")
    print("        vif | janela_previsibilidade | selecao_automatica | selecao_resumo | base_completa")