# -*- coding: utf-8 -*-
"""
================================================================================
SELEÇÃO DE VARIÁVEIS POR CRITÉRIO DE INFORMAÇÃO (AIC e BIC)
Forward · Backward · Stepwise

TCC — Capacidade Preditiva das Pesquisas Eleitorais
Variável dependente: abs_vies (erro absoluto da pesquisa)
Gabriel Follmer de Mattos — UFRGS
================================================================================

COMO USAR
    1. Ajuste o bloco CONFIG (caminhos, base, variáveis forçadas e candidatas).
    2. Rode o arquivo inteiro. Ele imprime, em ordem:
         - diagnóstico da base (shape, nulos, colinearidade)
         - o log passo a passo de cada algoritmo
         - a tabela comparativa dos 6 resultados (3 algoritmos × 2 critérios)
         - o summary do modelo final

IDEIA CENTRAL
    A cada passo o algoritmo testa todos os movimentos possíveis, calcula AIC e
    BIC de cada um, e executa o melhor movimento SEGUNDO O CRITÉRIO ESCOLHIDO.
    O outro critério é calculado junto e exibido ao lado, com uma marca [!]
    quando ele teria decidido diferente. É ali que se vê onde os dois discordam.
================================================================================
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# ==============================================================================
# CONFIG — o único bloco que você precisa mexer
# ==============================================================================

CAMINHO_2018 = r"C:\Users\Games\Downloads\setembro\Dados\base_completa_2018.xlsx"
CAMINHO_2022 = r"C:\Users\Games\Downloads\setembro\Dados\base_completa_2022.xlsx"

# "2018", "2022" ou "pool" (empilha as duas com uma dummy de eleição)
BASE = "pool"

DEPENDENTE = "abs_vies"

# Variáveis de interesse: entram em TODOS os modelos e nunca são removidas.
# É isto que garante que o efeito do turno sobrevive à seleção.
FORCADAS = [ "turno","dias_ate_eleicao"]

# Pool de controles sobre o qual a seleção efetivamente opera.
CANDIDATAS = [
    "dias_campo",
    "prop_indecisos",
    "competitividade",
    "log_amostra",
    #"margem_erro_pp",   # ver nota metodológica abaixo
    "valor_pesquisa",
    "presencial",
]

# NOTA METODOLÓGICA — margem_erro_pp
# A margem de erro declarada é registrada antes do encerramento do campo e não
# incorpora pós-estratificação, logo não reflete a margem real (Gramacho, 2013;
# Meireles & Russo, 2022). Se essa decisão vale, remova-a do pool acima.

TESTE_TAMANHO = 0.3
SEMENTE = 100
TOL = 1e-8           # exigência de melhora estrita (proteção contra ciclo)


# ==============================================================================
# 1. LEITURA E PREPARO
# ==============================================================================

def carrega_base(qual):
    if qual == "2018":
        df = pd.read_excel(CAMINHO_2018)
        df["eleicao"] = 2018
    elif qual == "2022":
        df = pd.read_excel(CAMINHO_2022)
        df["eleicao"] = 2022
    elif qual == "pool":
        a = pd.read_excel(CAMINHO_2018); a["eleicao"] = 2018
        b = pd.read_excel(CAMINHO_2022); b["eleicao"] = 2022
        df = pd.concat([a, b], ignore_index=True)
    else:
        raise ValueError("BASE deve ser '2018', '2022' ou 'pool'.")
    return df


def prepara(df):
    """Seleciona colunas por NOME, normaliza o turno e dropa NA uma única vez."""
    colunas = [DEPENDENTE] + FORCADAS + CANDIDATAS
    if BASE == "pool":
        colunas.append("eleicao")

    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas ausentes na base: {faltando}")

    dados = df[colunas].copy()

    # turno como dummy 0/1 (0 = 1º turno, 1 = 2º turno), qualquer que seja a codificação
    if dados["turno"].nunique() == 2:
        maior = dados["turno"].max()
        dados["turno"] = (dados["turno"] == maior).astype(int)

    if BASE == "pool":
        dados["eleicao_2022"] = (dados["eleicao"] == 2022).astype(int)
        dados = dados.drop(columns="eleicao")

    # CRÍTICO: AIC/BIC só são comparáveis entre modelos ajustados na MESMA amostra.
    # Por isso o dropna acontece aqui, uma vez, sobre todas as colunas do estudo.
    antes = len(dados)
    dados = dados.dropna()
    print(f"[preparo] {antes} linhas -> {len(dados)} após dropna "
          f"({antes - len(dados)} removidas)")
    return dados


def diagnostico(dados, X):
    print("\n" + "=" * 78)
    print("DIAGNÓSTICO DA BASE")
    print("=" * 78)
    print(f"\nShape: {dados.shape}")
    print(f"\nDependente ({DEPENDENTE}):")
    print(dados[DEPENDENTE].describe().to_string())
    print("\nNulos por coluna:")
    print(dados.isnull().sum().to_string())

    # VIF: turno, competitividade e prop_indecisos tendem a ser colineares,
    # porque o 2º turno tem 2 candidatos e disputa mais apertada por construção.
    Xc = sm.add_constant(X, has_constant="add")
    vif = pd.DataFrame({
        "variavel": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
    })
    print("\nVIF (>10 sugere colinearidade preocupante):")
    print(vif[vif.variavel != "const"].to_string(index=False))


# ==============================================================================
# 2. FUNÇÕES AUXILIARES — ajustar e pontuar
# ==============================================================================

def ajusta_modelo(y, X, termos):
    """Ajusta OLS de y contra os `termos` (lista de nomes) + constante."""
    if len(termos) == 0:
        X1 = pd.DataFrame({"const": np.ones(len(X))}, index=X.index)
    else:
        X1 = sm.add_constant(X[list(termos)], has_constant="add")
    return sm.OLS(y, X1).fit()


def criterios(modelo):
    """Devolve os DOIS critérios de uma vez."""
    return {"aic": modelo.aic, "bic": modelo.bic}


def _mostra(sinal, nome, c, atual):
    """Imprime uma candidata com AIC e BIC lado a lado, com os deltas."""
    print(f"    {sinal} {nome:<20}"
          f" AIC={c['aic']:10.3f} ({c['aic'] - atual['aic']:+8.3f})"
          f"   BIC={c['bic']:10.3f} ({c['bic'] - atual['bic']:+8.3f})")


def _linha_hist(passo, mov, var, atual, n):
    return {"passo": passo, "movimento": mov, "variavel": var,
            "aic": atual["aic"], "bic": atual["bic"], "n_termos": n}


# ==============================================================================
# 3. FORWARD
# ==============================================================================

def forward(y, X, forcadas=(), tipo="bic", verbose=True):
    outro = "aic" if tipo == "bic" else "bic"
    selecionadas = list(forcadas)
    candidatas = [v for v in X.columns if v not in selecionadas]

    atual = criterios(ajusta_modelo(y, X, selecionadas))
    historico = [_linha_hist(0, "inicio", None, atual, len(selecionadas))]
    if verbose:
        print(f"\nFORWARD ({tipo.upper()}) | início: {selecionadas} | "
              f"AIC={atual['aic']:.3f} BIC={atual['bic']:.3f}\n")

    passo = 0
    while candidatas:
        passo += 1
        resultados = []
        for v in candidatas:
            m = ajusta_modelo(y, X, selecionadas + [v])
            resultados.append({"var": v, **criterios(m)})

        resultados.sort(key=lambda r: r[tipo])
        vencedora = resultados[0]
        alt = min(resultados, key=lambda r: r[outro])

        if verbose:
            print(f"--- Passo {passo} --- decide por {tipo.upper()} | "
                  f"atual AIC={atual['aic']:.3f} BIC={atual['bic']:.3f}")
            for r in resultados:
                _mostra("+", r["var"], r, atual)
            if alt["var"] != vencedora["var"]:
                print(f"    [!] {outro.upper()} teria escolhido {alt['var']}")

        if vencedora[tipo] < atual[tipo] - TOL:
            selecionadas.append(vencedora["var"])
            candidatas.remove(vencedora["var"])
            atual = {"aic": vencedora["aic"], "bic": vencedora["bic"]}
            historico.append(_linha_hist(passo, "add", vencedora["var"],
                                         atual, len(selecionadas)))
            if verbose:
                print(f"    => ENTRA: {vencedora['var']}\n")
        else:
            if verbose:
                print(f"    => nenhuma candidata melhora o {tipo.upper()}. Parando.\n")
            break

    modelo = ajusta_modelo(y, X, selecionadas)
    return selecionadas, criterios(modelo), modelo, pd.DataFrame(historico)


# ==============================================================================
# 4. BACKWARD
# ==============================================================================

def backward(y, X, forcadas=(), tipo="bic", verbose=True):
    outro = "aic" if tipo == "bic" else "bic"
    forcadas = list(forcadas)
    selecionadas = list(X.columns)

    atual = criterios(ajusta_modelo(y, X, selecionadas))
    historico = [_linha_hist(0, "inicio", None, atual, len(selecionadas))]
    if verbose:
        print(f"\nBACKWARD ({tipo.upper()}) | início: {len(selecionadas)} termos | "
              f"AIC={atual['aic']:.3f} BIC={atual['bic']:.3f}\n")

    passo = 0
    while True:
        passo += 1
        removiveis = [v for v in selecionadas if v not in forcadas]
        if not removiveis:
            break

        resultados = []
        for v in removiveis:
            m = ajusta_modelo(y, X, [t for t in selecionadas if t != v])
            resultados.append({"var": v, **criterios(m)})

        resultados.sort(key=lambda r: r[tipo])
        perdedora = resultados[0]
        alt = min(resultados, key=lambda r: r[outro])

        if verbose:
            print(f"--- Passo {passo} --- decide por {tipo.upper()} | "
                  f"atual AIC={atual['aic']:.3f} BIC={atual['bic']:.3f}")
            for r in resultados:
                _mostra("-", r["var"], r, atual)
            if alt["var"] != perdedora["var"]:
                print(f"    [!] {outro.upper()} teria removido {alt['var']}")

        if perdedora[tipo] < atual[tipo] - TOL:
            selecionadas.remove(perdedora["var"])
            atual = {"aic": perdedora["aic"], "bic": perdedora["bic"]}
            historico.append(_linha_hist(passo, "drop", perdedora["var"],
                                         atual, len(selecionadas)))
            if verbose:
                print(f"    => SAI: {perdedora['var']}\n")
        else:
            if verbose:
                print(f"    => nenhuma remoção melhora o {tipo.upper()}. Parando.\n")
            break

    modelo = ajusta_modelo(y, X, selecionadas)
    return selecionadas, criterios(modelo), modelo, pd.DataFrame(historico)


# ==============================================================================
# 5. STEPWISE (bidirecional)
# ==============================================================================

def stepwise(y, X, forcadas=(), tipo="bic", verbose=True):
    outro = "aic" if tipo == "bic" else "bic"
    forcadas = list(forcadas)
    selecionadas = list(forcadas)
    candidatas = [v for v in X.columns if v not in selecionadas]

    atual = criterios(ajusta_modelo(y, X, selecionadas))
    historico = [_linha_hist(0, "inicio", None, atual, len(selecionadas))]
    if verbose:
        print(f"\nSTEPWISE ({tipo.upper()}) | início: {selecionadas} | "
              f"AIC={atual['aic']:.3f} BIC={atual['bic']:.3f}\n")

    passo = 0
    while True:
        passo += 1
        movimentos = []

        for v in candidatas:                                       # adicionar
            m = ajusta_modelo(y, X, selecionadas + [v])
            movimentos.append({"acao": "add", "var": v, **criterios(m)})

        for v in [t for t in selecionadas if t not in forcadas]:    # remover
            m = ajusta_modelo(y, X, [t for t in selecionadas if t != v])
            movimentos.append({"acao": "drop", "var": v, **criterios(m)})

        if not movimentos:
            break

        movimentos.sort(key=lambda r: r[tipo])
        melhor = movimentos[0]
        alt = min(movimentos, key=lambda r: r[outro])

        if verbose:
            print(f"--- Passo {passo} --- decide por {tipo.upper()} | "
                  f"atual AIC={atual['aic']:.3f} BIC={atual['bic']:.3f}")
            for r in movimentos:
                _mostra("+" if r["acao"] == "add" else "-", r["var"], r, atual)
            if (alt["var"], alt["acao"]) != (melhor["var"], melhor["acao"]):
                print(f"    [!] {outro.upper()} teria feito "
                      f"{alt['acao']} em {alt['var']}")

        if melhor[tipo] >= atual[tipo] - TOL:
            if verbose:
                print(f"    => nenhum movimento melhora o {tipo.upper()}. Parando.\n")
            break

        if melhor["acao"] == "add":
            selecionadas.append(melhor["var"])
            candidatas.remove(melhor["var"])
        else:
            selecionadas.remove(melhor["var"])
            candidatas.append(melhor["var"])

        atual = {"aic": melhor["aic"], "bic": melhor["bic"]}
        historico.append(_linha_hist(passo, melhor["acao"], melhor["var"],
                                     atual, len(selecionadas)))
        if verbose:
            print(f"    => {melhor['acao'].upper()}: {melhor['var']}\n")

    modelo = ajusta_modelo(y, X, selecionadas)
    return selecionadas, criterios(modelo), modelo, pd.DataFrame(historico)


# ==============================================================================
# 6. TABELA COMPARATIVA — 3 algoritmos × 2 critérios
# ==============================================================================

ALGORITMOS = [("forward", forward), ("backward", backward), ("stepwise", stepwise)]


def tabela_comparativa(y, X, forcadas=()):
    linhas = []
    for nome, func in ALGORITMOS:
        for tipo in ("aic", "bic"):
            sel, crit, mod, _ = func(y, X, forcadas=forcadas, tipo=tipo, verbose=False)
            linhas.append({
                "algoritmo": nome,
                "criterio": tipo.upper(),
                "n_termos": len(sel),
                "AIC": round(crit["aic"], 2),
                "BIC": round(crit["bic"], 2),
                "R2_aj": round(mod.rsquared_adj, 4),
                "variaveis": ", ".join(sel),
            })
    return pd.DataFrame(linhas)


def estabilidade_dos_betas(y, X, forcadas=()):
    """Coeficiente das variáveis de interesse em cada uma das 6 especificações.
    É a tabela de robustez: se turno e dias_ate_eleicao ficam estáveis,
    a conclusão não depende da especificação escolhida."""
    linhas = []
    for nome, func in ALGORITMOS:
        for tipo in ("aic", "bic"):
            sel, _, mod, _ = func(y, X, forcadas=forcadas, tipo=tipo, verbose=False)
            linha = {"algoritmo": nome, "criterio": tipo.upper()}
            for v in forcadas:
                linha[f"beta_{v}"] = round(mod.params.get(v, np.nan), 5)
                linha[f"p_{v}"] = round(mod.pvalues.get(v, np.nan), 4)
            linhas.append(linha)
    return pd.DataFrame(linhas)


# ==============================================================================
# 7. EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    bruto = carrega_base(BASE)
    dados = prepara(bruto)

    colunas_X = FORCADAS + CANDIDATAS
    if BASE == "pool":
        colunas_X = colunas_X + ["eleicao_2022"]

    X = dados[colunas_X]
    y = dados[DEPENDENTE]

    diagnostico(dados, X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TESTE_TAMANHO, random_state=SEMENTE
    )
    print(f"\n[split] treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

    # --- modelo de referência: só as variáveis de interesse -------------------
    print("\n" + "=" * 78)
    print("MODELO DE REFERÊNCIA (apenas as variáveis forçadas)")
    print("=" * 78)
    base_mod = ajusta_modelo(y_train, X_train, FORCADAS)
    print(base_mod.summary())

    # --- os três algoritmos, com log passo a passo ---------------------------
    for tipo in ("aic", "bic"):
        for nome, func in ALGORITMOS:
            print("\n" + "=" * 78)
            print(f"{nome.upper()} — decisão por {tipo.upper()}")
            print("=" * 78)
            func(y_train, X_train, forcadas=FORCADAS, tipo=tipo, verbose=True)

    # --- tabelas de síntese --------------------------------------------------
    print("\n" + "=" * 78)
    print("TABELA COMPARATIVA")
    print("=" * 78)
    tab = tabela_comparativa(y_train, X_train, forcadas=FORCADAS)
    print(tab.to_string(index=False))

    print("\n" + "=" * 78)
    print("ESTABILIDADE DOS COEFICIENTES DE INTERESSE")
    print("=" * 78)
    print(estabilidade_dos_betas(y_train, X_train, forcadas=FORCADAS).to_string(index=False))

    # --- modelo final: o mais parcimonioso (stepwise/BIC) --------------------
    print("\n" + "=" * 78)
    print("MODELO FINAL — stepwise/BIC")
    print("=" * 78)
    sel_final, crit_final, mod_final, hist_final = stepwise(
        y_train, X_train, forcadas=FORCADAS, tipo="bic", verbose=False
    )
    print(f"\nVariáveis: {sel_final}")
    print(f"AIC={crit_final['aic']:.3f} | BIC={crit_final['bic']:.3f}\n")
    print(mod_final.summary())

    print("\nHistórico de passos:")
    print(hist_final.to_string(index=False))

    # --- desempenho fora da amostra -----------------------------------------
    X_test_f = sm.add_constant(X_test[sel_final], has_constant="add")
    pred = mod_final.predict(X_test_f)
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - pred)))
    print(f"\n[fora da amostra] RMSE={rmse:.5f} | MAE={mae:.5f}")

    # tab.to_csv("resultados_tcc/selecao_comparativa.csv", index=False)