# -*- coding: utf-8 -*-
"""
================================================================================
TRÊS CAMADAS DE MODELAGEM — TCC Capacidade Preditiva das Pesquisas Eleitorais
Gabriel Follmer de Mattos — UFRGS

Camada 1 — REFERÊNCIA .... MQO com erros-padrão robustos (HC1)
Camada 2 — FLEXIBILIDADE . GAM (suaves em dias_ate_eleicao e prop_indecisos)
                           + kernel local linear como robustez
Camada 3 — AVALIAÇÃO ..... EQM fora da amostra (K-fold e treino 2018 -> 2022)

Dependente: abs_vies (erro absoluto da pesquisa)
================================================================================

DUAS PERGUNTAS QUE O SCRIPT RESPONDE
    (1) Associação: como as características da pesquisa se associam ao erro?
        -> Camada 1 dá os betas; Camada 2 mostra se a relação é mesmo linear.
    (2) Previsão: a flexibilidade extra melhora a previsão fora da amostra?
        -> Camada 3. Curva mais complexa não é automaticamente mais útil.

INSTALAÇÃO
    pip install pandas numpy statsmodels scikit-learn matplotlib openpyxl pygam
================================================================================
"""

import os
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # salva os gráficos em arquivo, não abre janela
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, train_test_split

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# CONFIG — o único bloco que você precisa mexer
# ==============================================================================

CAMINHO_2018 = r"C:\Users\Games\Downloads\setembro\Dados\base_completa_2018.xlsx"
CAMINHO_2022 = r"C:\Users\Games\Downloads\setembro\Dados\base_completa_2022.xlsx"

PASTA_SAIDA = "resultados_regressão"

DEPENDENTE = "abs_vies"

# --- especificação -----------------------------------------------------------
# Cole aqui o conjunto que saiu da sua seleção (stepwise/BIC, por exemplo).
# A ORDEM importa: no GAM, o primeiro suave é f1, o segundo é f2, etc.

SUAVES = ["dias_ate_eleicao"]   # f1(.) e f2(.) , "prop_indecisos"
#LINEARES = ["log_amostra", "competitividade", "dias_campo", "valor_pesquisa"]
DUMMIES = ["turno"]       #"turno"         # eleicao_2022 é acrescentada sozinha - presencial

# --- avaliação ---------------------------------------------------------------
SEMENTE = 100
N_FOLDS = 5
TESTE_TAMANHO = 0.3

RODAR_KERNEL = True      # kernel local linear sobre as variáveis de SUAVES
KERNEL_MAX_N = 1200      # acima disso, a banda é escolhida em subamostra

LAM_GRID = np.logspace(-3, 3, 13)   # grade de penalização do GAM


# ==============================================================================
# 1. LEITURA E PREPARO
# ==============================================================================

def carrega_pool():
    """Empilha 2018 e 2022 e cria a dummy de eleição."""
    a = pd.read_excel(CAMINHO_2018); a["eleicao"] = 2018
    b = pd.read_excel(CAMINHO_2022); b["eleicao"] = 2022
    return pd.concat([a, b], ignore_index=True)


def prepara(df):
    """Seleciona colunas, normaliza turno, cria eleicao_2022 e dropa NA uma vez."""
    regressores = SUAVES  + DUMMIES
    colunas = [DEPENDENTE] + regressores + ["eleicao"]

    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas ausentes na base: {faltando}")

    dados = df[colunas].copy()

    # turno como dummy 0/1 (0 = 1º turno, 1 = 2º turno)
    #if dados["turno"].nunique() == 2:
    #    dados["turno"] = (dados["turno"] == dados["turno"].max()).astype(int)

    dados["eleicao_2022"] = (dados["eleicao"] == 2022).astype(int)

    antes = len(dados)
    dados = dados.dropna().reset_index(drop=True)
    print(f"[preparo] {antes} linhas -> {len(dados)} após dropna "
          f"({antes - len(dados)} removidas)")
    return dados


def monta_matriz(dados, incluir_dummy_eleicao=True):
    """Devolve X (DataFrame, ordem fixa) e y. A ordem das colunas define os
    índices dos termos do GAM, então é montada num só lugar."""
    cols = SUAVES  + DUMMIES
    if incluir_dummy_eleicao:
        cols = cols + ["eleicao_2022"]
    return dados[cols].astype(float), dados[DEPENDENTE].astype(float)


def diagnostico(dados, X):
    print("\n" + "=" * 78)
    print("DIAGNÓSTICO DA BASE")
    print("=" * 78)
    print(f"\nShape: {dados.shape}")
    print(f"\nDependente ({DEPENDENTE}):")
    print(dados[DEPENDENTE].describe().to_string())
    print(f"\nObservações por eleição:")
    print(dados["eleicao"].value_counts().sort_index().to_string())

    Xc = sm.add_constant(X, has_constant="add")
    vif = pd.DataFrame({
        "variavel": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
    })
    print("\nVIF (>10 sugere colinearidade preocupante):")
    print(vif[vif.variavel != "const"].to_string(index=False))


# ==============================================================================
# 2. CAMADA 1 — MQO (modelo de referência)
# ==============================================================================

def ajusta_mqo(X, y, robusto=True):
    Xc = sm.add_constant(X, has_constant="add")
    mod = sm.OLS(y, Xc)
    return mod.fit(cov_type="HC1") if robusto else mod.fit()


def camada1_mqo(X, y):
    print("\n" + "=" * 78)
    print("CAMADA 1 — MQO COM ERROS-PADRÃO ROBUSTOS (HC1)")
    print("=" * 78)

    mqo = ajusta_mqo(X, y, robusto=True)
    print(mqo.summary())

    # Os testes de especificação precisam do ajuste sem correção robusta.
    simples = ajusta_mqo(X, y, robusto=False)

    bp = het_breuschpagan(simples.resid, simples.model.exog)
    reset = linear_reset(simples, power=3, use_f=True)

    print("\n--- Testes de especificação " + "-" * 50)
    print(f"Breusch-Pagan (heterocedasticidade) : p = {bp[1]:.4f}")
    print(f"RESET de Ramsey (forma funcional)   : p = {reset.pvalue:.4f}")
    if reset.pvalue < 0.05:
        print("  -> RESET rejeita a forma linear. Isso já justifica a Camada 2.")
    else:
        print("  -> RESET não rejeita a linearidade. A Camada 2 vira robustez.")
    return mqo


# ==============================================================================
# 3. CAMADA 2 — GAM (flexibilidade)
# ==============================================================================

def constroi_termos(colunas):
    """s() para as variáveis de SUAVES, l() para o resto, na ordem de `colunas`."""
    from pygam import s, l
    termos = []
    for i, nome in enumerate(colunas):
        termos.append(s(i, n_splines=10) if nome in SUAVES else l(i))
    return reduce(lambda a, b: a + b, termos)


def ajusta_gam(X, y, lam_grid=LAM_GRID, verbose=False):
    """GAM com penalização escolhida por GCV numa grade de lambda."""
    from pygam import LinearGAM
    gam = LinearGAM(constroi_termos(list(X.columns)))
    gam.gridsearch(X.to_numpy(float), y.to_numpy(float),
                   lam=lam_grid, progress=verbose)
    return gam


def ajusta_gam_linear(X, y, lam_grid=LAM_GRID):
    """Mesmo GAM, mas com TODOS os termos lineares. Serve de contrafactual:
    se o GAM suave não vence este aqui em AIC/GCV, a não linearidade não paga."""
    from pygam import LinearGAM, l
    termos = reduce(lambda a, b: a + b, [l(i) for i in range(X.shape[1])])
    gam = LinearGAM(termos)
    gam.gridsearch(X.to_numpy(float), y.to_numpy(float),
                   lam=lam_grid, progress=False)
    return gam


def camada2_gam(X, y):
    print("\n" + "=" * 78)
    print("CAMADA 2 — GAM (suaves em: " + ", ".join(SUAVES) + ")")
    print("=" * 78)

    gam = ajusta_gam(X, y)
    gam.summary()

    gam_lin = ajusta_gam_linear(X, y)
    print("\n--- GAM suave vs. GAM todo linear " + "-" * 43)
    print(f"{'':<14}{'AIC':>12}{'GCV':>12}{'EDoF':>10}")
    for nome, g in [("suave", gam), ("linear", gam_lin)]:
        st = g.statistics_
        print(f"{nome:<14}{st['AIC']:>12.2f}{st['GCV']:>12.6f}{st['edof']:>10.2f}")
    if gam.statistics_["AIC"] < gam_lin.statistics_["AIC"]:
        print("  -> O GAM suave tem AIC menor: há não linearidade relevante.")
    else:
        print("  -> O GAM suave NÃO melhora o AIC: a linearidade dá conta.")

    print("\nLeitura dos graus de liberdade efetivos (EDoF por termo no summary):")
    print("  EDoF ~ 1  => o termo é essencialmente uma reta")
    print("  EDoF > 2  => há curvatura que o MQO não captura")
    return gam


def grafico_suaves(gam, X, nome_arquivo="gam_funcoes_suaves.png"):
    """Plota f1, f2, ... com intervalo de confiança de 95%."""
    cols = list(X.columns)
    idx_suaves = [i for i, c in enumerate(cols) if c in SUAVES]
    if not idx_suaves:
        return

    fig, axes = plt.subplots(1, len(idx_suaves),
                             figsize=(5.5 * len(idx_suaves), 4.2))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx_suaves):
        grade = gam.generate_X_grid(term=i)
        pdep, ic = gam.partial_dependence(term=i, X=grade, width=0.95)
        ax.plot(grade[:, i], pdep, lw=2)
        ax.plot(grade[:, i], ic, ls="--", c="gray", lw=1)
        ax.axhline(0, c="black", lw=0.6, alpha=0.5)
        ax.set_xlabel(cols[i])
        ax.set_ylabel(f"efeito parcial sobre {DEPENDENTE}")
        ax.set_title(f"f({cols[i]})")
    plt.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, nome_arquivo)
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"\n[gráfico] funções suaves salvas em {caminho}")


# ==============================================================================
# 3b. CAMADA 2 (robustez) — kernel local linear
# ==============================================================================

def ajusta_kernel(X, y):
    """Kernel local linear com banda por validação cruzada.
    Usa SÓ as variáveis de SUAVES: com muitos regressores contínuos o kernel
    sofre da maldição da dimensionalidade e vira ruído."""
    from statsmodels.nonparametric.kernel_regression import KernelReg
    Xk = X[SUAVES].to_numpy(float)
    yk = y.to_numpy(float)
    var_type = "c" * Xk.shape[1]

    if len(yk) > KERNEL_MAX_N:
        rng = np.random.default_rng(SEMENTE)
        idx = rng.choice(len(yk), KERNEL_MAX_N, replace=False)
        bw = KernelReg(yk[idx], Xk[idx], var_type=var_type,
                       reg_type="ll", bw="cv_ls").bw
        print(f"[kernel] banda escolhida em subamostra de {KERNEL_MAX_N} obs.")
        return KernelReg(yk, Xk, var_type=var_type, reg_type="ll", bw=bw)
    return KernelReg(yk, Xk, var_type=var_type, reg_type="ll", bw="cv_ls")


def camada2_kernel(X, y):
    print("\n" + "=" * 78)
    print("CAMADA 2 (robustez) — KERNEL LOCAL LINEAR em: " + ", ".join(SUAVES))
    print("=" * 78)

    kr = ajusta_kernel(X, y)
    print(f"Larguras de banda: "
          + ", ".join(f"{v}={b:.4f}" for v, b in zip(SUAVES, kr.bw)))
    print(f"R² (dentro da amostra): {kr.r_squared():.4f}")

    _, efeitos = kr.fit()
    emm = pd.Series(efeitos.mean(axis=0), index=SUAVES)
    print("\nEfeito marginal médio (o análogo dos betas no não paramétrico):")
    print(emm.to_string())
    print("\nAtenção: aqui não existe 'beta'. A derivada varia ponto a ponto;")
    print("a média só resume a inclinação típica da curva.")
    return kr


# ==============================================================================
# 4. CAMADA 3 — AVALIAÇÃO FORA DA AMOSTRA
# ==============================================================================

def metricas(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    erro = y_true - y_pred
    eqm = float(np.mean(erro ** 2))
    return {"EQM": eqm, "RMSE": float(np.sqrt(eqm)),
            "MAE": float(np.mean(np.abs(erro)))}


def prediz_kernel(kr, X_novo):
    return kr.fit(X_novo[SUAVES].to_numpy(float))[0]


def avalia_kfold(X, y, n_folds=N_FOLDS):
    """Reajusta TODOS os modelos dentro de cada dobra. Reaproveitar um modelo
    treinado na amostra inteira contaminaria a comparação."""
    print("\n" + "=" * 78)
    print(f"CAMADA 3a — VALIDAÇÃO CRUZADA ({n_folds} dobras)")
    print("=" * 78)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEMENTE)
    acumulado = {"média (baseline)": [], "MQO": [], "GAM": []}
    if RODAR_KERNEL:
        acumulado["Kernel"] = []

    for k, (i_tr, i_te) in enumerate(kf.split(X), start=1):
        X_tr, X_te = X.iloc[i_tr], X.iloc[i_te]
        y_tr, y_te = y.iloc[i_tr], y.iloc[i_te]
        print(f"  dobra {k}/{n_folds} ...", end="", flush=True)

        # baseline: prever sempre a média do treino
        acumulado["média (baseline)"].append(
            metricas(y_te, np.full(len(y_te), y_tr.mean())))

        m = ajusta_mqo(X_tr, y_tr, robusto=False)
        acumulado["MQO"].append(
            metricas(y_te, m.predict(sm.add_constant(X_te, has_constant="add"))))

        g = ajusta_gam(X_tr, y_tr)
        acumulado["GAM"].append(metricas(y_te, g.predict(X_te.to_numpy(float))))

        if RODAR_KERNEL:
            kr = ajusta_kernel(X_tr, y_tr)
            acumulado["Kernel"].append(metricas(y_te, prediz_kernel(kr, X_te)))

        print(" ok")

    linhas = []
    for nome, lista in acumulado.items():
        media = {m: float(np.mean([d[m] for d in lista])) for m in ("EQM", "RMSE", "MAE")}
        linhas.append({"modelo": nome, **media})
    tab = pd.DataFrame(linhas)

    # R² fora da amostra: ganho sobre prever simplesmente a média
    eqm_base = tab.loc[tab.modelo == "média (baseline)", "EQM"].iloc[0]
    tab["R2_fora"] = 1 - tab["EQM"] / eqm_base

    print("\n" + tab.round(6).to_string(index=False))
    return tab


def avalia_temporal(dados):
    """Treina em 2018, prevê 2022. É o teste mais duro e mais parecido com o
    uso real: prever uma eleição com o que se aprendeu na anterior."""
    print("\n" + "=" * 78)
    print("CAMADA 3b — TREINO EM 2018, PREVISÃO DE 2022")
    print("=" * 78)

    treino = dados[dados["eleicao"] == 2018]
    teste = dados[dados["eleicao"] == 2022]
    print(f"treino (2018): {len(treino)} | teste (2022): {len(teste)}")

    # eleicao_2022 é constante dentro de cada subamostra: sai do modelo.
    X_tr, y_tr = monta_matriz(treino, incluir_dummy_eleicao=False)
    X_te, y_te = monta_matriz(teste, incluir_dummy_eleicao=False)

    resultados = {"média (baseline)": metricas(y_te, np.full(len(y_te), y_tr.mean()))}

    m = ajusta_mqo(X_tr, y_tr, robusto=False)
    resultados["MQO"] = metricas(y_te, m.predict(sm.add_constant(X_te, has_constant="add")))

    g = ajusta_gam(X_tr, y_tr)
    resultados["GAM"] = metricas(y_te, g.predict(X_te.to_numpy(float)))

    if RODAR_KERNEL:
        kr = ajusta_kernel(X_tr, y_tr)
        resultados["Kernel"] = metricas(y_te, prediz_kernel(kr, X_te))

    tab = pd.DataFrame([{"modelo": k, **v} for k, v in resultados.items()])
    eqm_base = tab.loc[tab.modelo == "média (baseline)", "EQM"].iloc[0]
    tab["R2_fora"] = 1 - tab["EQM"] / eqm_base
    print("\n" + tab.round(6).to_string(index=False))

    print("\nCuidado na leitura: o R² fora da amostra pode ser NEGATIVO. Isso")
    print("significa que o modelo prevê pior do que a média de 2018, ou seja,")
    print("que o padrão de erro de 2018 não se transporta para 2022.")
    return tab


def grafico_ajuste(y_true, preds, nome_arquivo="previsto_vs_observado.png"):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lim = [0, float(np.max(y_true)) * 1.05]
    for ax, (nome, yp) in zip(axes, preds.items()):
        ax.scatter(y_true, yp, s=14, alpha=0.5)
        ax.plot(lim, lim, c="black", lw=1, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("observado"); ax.set_ylabel("previsto")
        ax.set_title(nome)
    plt.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, nome_arquivo)
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"[gráfico] previsto vs. observado salvo em {caminho}")


# ==============================================================================
# 5. EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    dados = prepara(carrega_pool())
    X, y = monta_matriz(dados, incluir_dummy_eleicao=True)
    diagnostico(dados, X)

    # ---- Camada 1 -----------------------------------------------------------
    mqo = camada1_mqo(X, y)

    # ---- Camada 2 -----------------------------------------------------------
    try:
        gam = ajusta_gam(X, y)   # ajuste único, reaproveitado nos gráficos
        gam_ok = True
    except ImportError:
        print("\n[aviso] pygam não instalado. Rode: pip install pygam")
        gam_ok = False

    if gam_ok:
        gam = camada2_gam(X, y)
        grafico_suaves(gam, X)

    kr = camada2_kernel(X, y) if RODAR_KERNEL else None

    # ---- Camada 3 -----------------------------------------------------------
    tab_cv = avalia_kfold(X, y)
    tab_temporal = avalia_temporal(dados)

    # ---- Síntese ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SÍNTESE")
    print("=" * 78)
    comparacao = tab_cv.merge(tab_temporal, on="modelo",
                              suffixes=("_cv", "_2022"))
    print(comparacao[["modelo", "EQM_cv", "R2_fora_cv",
                      "EQM_2022", "R2_fora_2022"]].round(6).to_string(index=False))

    print("\nComo interpretar:")
    print("  · GAM melhor que MQO nas DUAS colunas -> a não linearidade é real")
    print("    e transportável; vale reportar as curvas f(.) como resultado.")
    print("  · GAM melhor só dentro da amostra -> é sobreajuste; o MQO é o")
    print("    modelo preferido por parcimônia e interpretabilidade.")
    print("  · Todos piores que o baseline em 2022 -> a evidência principal do")
    print("    TCC passa a ser a instabilidade entre eleições, e não os betas.")

    if gam_ok:
        preds = {"MQO": mqo.predict(sm.add_constant(X, has_constant="add")),
                 "GAM": gam.predict(X.to_numpy(float))}
        grafico_ajuste(y, preds)

    tab_cv.to_csv(os.path.join(PASTA_SAIDA, "comparacao_cv.csv"), index=False)
    tab_temporal.to_csv(os.path.join(PASTA_SAIDA, "comparacao_2022.csv"), index=False)
    print(f"\n[saída] tabelas gravadas em {PASTA_SAIDA}/")