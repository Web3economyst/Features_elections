# -*- coding: utf-8 -*-
"""
TCC — Precisão das pesquisas presidenciais
Padronização + seleção automática de features — ELEIÇÃO 2018

Roda a MESMA lógica que o script do outro ano; a única diferença é o
bloco CONFIG (arquivo de entrada/saída e ano). Mantenha o bloco
"CONFIG COMPARTILHADA" IDÊNTICO nos dois scripts, senão a comparação
entre anos perde a validade.

Requisitos: pandas, numpy, statsmodels, openpyxl.
"""
import re
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings("ignore")  # silencia avisos de rank do teste-F sob cluster

# ============================ CONFIG (por ano) ============================
ANO          = 2018
ARQ_ENTRADA  = "resultados_meus_testes.xlsx"      # deve conter a aba 'base_completa'
ABA_ENTRADA  = "base_completa"
ARQ_SAIDA    = "padronizado_2018.xlsx"

# ===================== CONFIG COMPARTILHADA (idêntica nos 2) ===============
DEP            = "abs_vies"          # variável dependente (viés absoluto)
CLUSTER_POR    = "id_pesquisa"       # nível de correlação (candidatos da mesma pesquisa)
FE_INSTITUTO   = True                # efeitos fixos de instituto (como no modelo original)

# (1) filtro de candidatos: mantém quem teve relevância eleitoral em ALGUM lado
#     (pesquisa OU resultado real) acima do limiar. Aplicado igual aos 2 anos.
LIMIAR_CANDIDATO = 1.0               # em pontos percentuais

# (2) cand_grande recomputada de forma consistente (a original está quebrada em 2022)
LIMIAR_CAND_GRANDE = 10.0            # real >= 10% -> candidato "grande"

# pool completo para a seleção automática (cand_grande entra recomputada)
POOL = ["var_phat", "dias_ate_eleicao", "is_vespera", "is_ultima_semana",
        "is_15dias", "is_30dias", "is_60dias", "log_amostra", "prop_indecisos",
        "cand_grande", "competitividade", "final_de_semana", "dias_campo",
        "auditoria_30", "telefonica", "valor_pesquisa"]

# especificação COMUM (teórica) rodada idêntica nos dois anos.
# tempo em dummies MUTUAMENTE EXCLUSIVAS (ref = >60 dias), SEM a contínua junto,
# para evitar a referência não-contígua que instabilizava os coeficientes.
ESPEC_COMUM = ["var_phat", "is_vespera", "is_ultima_semana", "is_15dias",
               "is_30dias", "is_60dias", "log_amostra", "telefonica", "dias_campo"]
# =========================================================================


def padronizar(df):
    """Deixa a base na base de comparação comum. Devolve (df, log)."""
    log = []
    n0 = len(df)

    # --- cenário principal (só existe em 2022) ---
    if "descricao_cenario" in df.columns:
        num = df["descricao_cenario"].map(
            lambda s: int(m.group(1)) if (m := re.search(r"cen[aá]rio\s+(\d+)", str(s), re.I)) else 1)
        df = df[num == 1].copy()
        log.append(f"cenário principal: -{n0 - len(df)} linhas (cenários secundários)")
    else:
        log.append("cenário: base sem coluna de cenário (nada a filtrar)")

    # --- filtro de candidatos (mesma regra nos 2 anos) ---
    n1 = len(df)
    relev = df[["percentual", "percentual_real"]].max(axis=1)
    df = df[relev >= LIMIAR_CANDIDATO].copy()
    log.append(f"filtro candidatos (max(pesq,real) >= {LIMIAR_CANDIDATO}%): -{n1 - len(df)} linhas")

    # --- cand_grande recomputada de forma consistente ---
    df["cand_grande"] = (df["percentual_real"] >= LIMIAR_CAND_GRANDE).astype(int)
    log.append(f"cand_grande recomputada: real >= {LIMIAR_CAND_GRANDE}%  "
               f"(n grandes = {int(df['cand_grande'].sum())})")

    # --- limpeza de NaN nas colunas usadas ---
    usar = [DEP, CLUSTER_POR, "instituto"] + POOL
    df = df.dropna(subset=usar).copy()
    log.append(f"após dropna: {len(df)} linhas ({df[CLUSTER_POR].nunique()} pesquisas)")
    return df, log


def _design(df, feats):
    """Monta X (const + FE instituto + feats) e y."""
    partes = [pd.Series(1.0, index=df.index, name="const")]
    if FE_INSTITUTO:
        fe = pd.get_dummies(df["instituto"].astype(str), prefix="inst", drop_first=True).astype(float)
        partes.append(fe)
    if feats:
        partes.append(df[feats].astype(float))
    X = pd.concat(partes, axis=1)
    y = df[DEP].astype(float)
    return X, y


def ajustar(df, feats, cluster=False):
    X, y = _design(df, feats)
    if cluster:
        return sm.OLS(y, X).fit(cov_type="cluster",
                                cov_kwds={"groups": df[CLUSTER_POR]})
    return sm.OLS(y, X).fit()


def _score(df, feats, criterio):
    r = ajustar(df, feats)
    return r.aic if criterio == "AIC" else r.bic


def forward(df, criterio):
    sel, rem = [], list(POOL)
    best = _score(df, sel, criterio)
    melhorou = True
    while rem and melhorou:
        melhorou = False
        tent = sorted((_score(df, sel + [f], criterio), f) for f in rem)
        s, f = tent[0]
        if s < best - 1e-9:
            best, melhorou = s, True
            sel.append(f); rem.remove(f)
    return sel


def backward(df, criterio):
    sel = list(POOL)
    best = _score(df, sel, criterio)
    melhorou = True
    while sel and melhorou:
        melhorou = False
        tent = sorted((_score(df, [x for x in sel if x != f], criterio), f) for f in sel)
        s, f = tent[0]
        if s < best - 1e-9:
            best, melhorou = s, True
            sel.remove(f)
    return sel


def stepwise(df, criterio):
    sel = []
    best = _score(df, sel, criterio)
    while True:
        mudou = False
        fora = [f for f in POOL if f not in sel]
        if fora:
            s, f = sorted((_score(df, sel + [f], criterio), f) for f in fora)[0]
            if s < best - 1e-9:
                sel.append(f); best = s; mudou = True
        if len(sel) > 1:
            s, f = sorted((_score(df, [x for x in sel if x != f], criterio), f) for f in sel)[0]
            if s < best - 1e-9:
                sel.remove(f); best = s; mudou = True
        if not mudou:
            break
    return sel


def rodar_selecao(df):
    metodos = {
        "Forward_AIC":  forward(df, "AIC"),  "Forward_BIC":  forward(df, "BIC"),
        "Backward_AIC": backward(df, "AIC"), "Backward_BIC": backward(df, "BIC"),
        "Stepwise_AIC": stepwise(df, "AIC"), "Stepwise_BIC": stepwise(df, "BIC"),
    }
    # matriz feature x método
    mat = pd.DataFrame({m: [int(f in feats) for f in POOL] for m, feats in metodos.items()},
                       index=POOL)
    mat["n_metodos"] = mat.sum(axis=1)
    mat = mat.sort_values("n_metodos", ascending=False).reset_index().rename(columns={"index": "feature"})
    resumo = pd.DataFrame([(m, len(f), ", ".join(f)) for m, f in metodos.items()],
                          columns=["metodo", "n_selecionadas", "features"])
    return mat, resumo


def tabela_coef(res):
    ci = res.conf_int()
    t = pd.DataFrame({
        "variavel": res.params.index, "coef": res.params.values,
        "erro_padrao": res.bse.values, "estat_t": res.tvalues.values,
        "p_valor": res.pvalues.values, "IC_2.5%": ci[0].values, "IC_97.5%": ci[1].values})
    t["signif"] = pd.cut(t["p_valor"], [-1, .01, .05, .1, 1], labels=["***", "**", "*", ""])
    # esconde as dummies de instituto para leitura
    t = t[~t["variavel"].str.startswith("inst_")].reset_index(drop=True)
    return t


def tabela_stats(res, df):
    try:
        fval, fp = round(float(res.fvalue), 2), round(float(res.f_pvalue), 4)
    except Exception:
        fval, fp = float("nan"), float("nan")  # F degenera sob cluster + FE colineares
    return pd.DataFrame({
        "estatistica": ["N (observações)", "N pesquisas (clusters)", "R²", "R²-ajustado",
                        "F", "p-valor (F)", "AIC", "BIC", "nº parâmetros"],
        "valor": [int(res.nobs), df[CLUSTER_POR].nunique(), round(res.rsquared, 4),
                  round(res.rsquared_adj, 4), fval, fp,
                  round(res.aic, 1), round(res.bic, 1), int(res.df_model + 1)]})


def tabela_vif(df, feats):
    X = sm.add_constant(df[feats].astype(float))
    v = [(c, round(variance_inflation_factor(X.values, i), 2))
         for i, c in enumerate(X.columns) if c != "const"]
    return pd.DataFrame(v, columns=["feature", "VIF"])


def janela(df):
    d = df["dias_ate_eleicao"]
    faixa = pd.cut(d, [-1, 0, 5, 10, 15, 20, 10**6],
                   labels=["0", "1-5", "6-10", "11-15", "16-20", ">20"])
    g = df.groupby(faixa, observed=True)[DEP].agg(["mean", "count"]).round(2).reset_index()
    return g.rename(columns={"index": "faixa_dias"})


def main():
    print(f"\n===== ELEIÇÃO {ANO} =====")
    bruto = pd.read_excel(ARQ_ENTRADA, sheet_name=ABA_ENTRADA)
    df, log = padronizar(bruto)
    print("Padronização:")
    for l in log:
        print("  -", l)

    mat, resumo = rodar_selecao(df)
    res_comum = ajustar(df, ESPEC_COMUM, cluster=True)  # SE robusto por pesquisa
    coef = tabela_coef(res_comum)
    stats = tabela_stats(res_comum, df)
    vif = tabela_vif(df, ESPEC_COMUM)
    jan = janela(df)

    print("\nEspecificação COMUM (SE clusterizado por pesquisa):")
    print(coef.to_string(index=False))
    print("\nRobustez da seleção (n_metodos, de 6):")
    print(mat[["feature", "n_metodos"]].to_string(index=False))

    with pd.ExcelWriter(ARQ_SAIDA, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="base_padronizada", index=False)
        mat.to_excel(w, sheet_name="selecao_matriz", index=False)
        resumo.to_excel(w, sheet_name="selecao_resumo", index=False)
        coef.to_excel(w, sheet_name="espec_comum_coef", index=False)
        stats.to_excel(w, sheet_name="espec_comum_stats", index=False)
        vif.to_excel(w, sheet_name="vif", index=False)
        jan.to_excel(w, sheet_name="janela_previsibilidade", index=False)
    print(f"\nSalvo em: {ARQ_SAIDA}\n")


if __name__ == "__main__":
    main()
