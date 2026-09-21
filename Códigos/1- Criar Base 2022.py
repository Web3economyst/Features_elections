#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base de analise do ciclo de 2022.  ->  python base_analise_2022.py

ENTRADAS
  base_pesquisas_2022.csv        formato LONGO (uma linha por candidato),
                                 ja com o vies calculado
  base_metodologias_2022_BR.csv  registros do TSE (dados abertos, sep ';')

SAIDA  base_analise_2022.xlsx
  base_longa     unidade de analise: candidato x pesquisa
  colunas_extra  o que nao entra na principal, ligado por id_pesquisa
  base_pesquisa  uma linha por pesquisa
  descartados    tudo que saiu, com o motivo
  evidencias     trecho do registro que sustenta cada feature
  log_auditoria  colisoes, divergencias e descartes

PARTICULARIDADES DE 2022
  1. Nao houve substituicao de candidato: Lula e Bolsonaro estavam postos
     desde o inicio. Nao existe corte equivalente ao de 11/09/2018, e
     dias_ate_eleicao mantem amplitude grande (1 a 49 dias).
  2. Segundo turno muito apertado (50,90 x 49,10). Vies absoluto menor e
     competitividade quase constante nas pesquisas pos-1o turno.
  3. Pablo Marcal e Roberto Jefferson aparecem em pesquisas mas nao tem
     resultado apurado (foram substituidos). Saem por falta de referencia.

CUIDADO COM OS NOMES DE COLUNA DO ARQUIVO DE ORIGEM
  'percentual_valido' NAO e o percentual da pesquisa: e o RESULTADO
  APURADO, constante por candidato e turno. O percentual da pesquisa
  renormalizado esta em 'percentual_pesquisa_valido'. Os nomes convidam ao
  engano, por isso aqui viram percentual_real e percentual_pesq_valido.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import nucleo_base_analise as nb

# ============================================================
# 1. CONFIGURACAO
# ============================================================
PATH_PESQUISAS = r"C:\Users\Games\Downloads\setembro\Dados\base_pesquisas_2022.csv"
PATH_METODOLOGIA = r"C:\Users\Games\Downloads\setembro\Dados\base_metodologias_2022_BR.csv"
DIR_SAIDA = Path(r"C:\Users\Games\Downloads\setembro\Dados")

ANO = 2022
DATAS_ELEICAO = {1: pd.Timestamp("2022-10-02"), 2: pd.Timestamp("2022-10-30")}

# Quem chegou ao 2o turno. Definicao EXOGENA de proposito: usar um limiar
# sobre o percentual da propria pesquisa tornaria a variavel funcao do
# mesmo numero que forma a dependente.
CAND_GRANDE = {"lula", "bolsonaro"}

# Pesquisas de 2o turno com campo anterior ao 1o mediam uma dupla que ainda
# nao estava definida. Nao sao comparaveis ao resultado de 30/10.
EXCLUIR_CENARIO_HIPOTETICO = True

# Modo de coleta atribuido manualmente, apos leitura do registro. Use so
# quando os padroes de texto nao resolvem e a leitura e conclusiva. Os dois
# abaixo (Verita) descrevem apenas "realizacao de entrevistas" e um plano
# amostral por PPT, sem citar setor censitario nem bairro. PPT sozinho NAO
# serve como regra geral: em 2022 ha 71 registros que combinam PPT com
# coleta telefonica. Dai a excecao ser manual e nominal.
MODO_MANUAL = {
    "BR-03033/2022": "presencial",
    "BR-01060/2022": "presencial",
}

# Colunas da aba principal, nesta ordem. id_pesquisa fica porque e a chave
# de agrupamento do erro-padrao cluster e do GroupKFold.
COLUNAS_PRINCIPAIS = [
    "id_pesquisa", "ano", "turno", "registro", "instituto", "contratante",
    "data", "data_eleicao", "dias_ate_eleicao", "dt_divulgacao",
    "dias_divulgacao_ate_eleicao", "dias_campo", "final_de_semana",
    "cenario_num", "nome_clean", "percentual", "soma_validos_pesquisa",
    "percentual_pesq_valido", "percentual_real", "vies_pp", "abs_vies",
    "cand_grande", "competitividade", "prop_indecisos", "amostra",
    "amostra_origem", "divergencia_amostra", "log_amostra", "margem_erro_pp",
    "valor_pesquisa", "telefonica", "presencial", "presencial_inferido",
    "online", "modo_manual", "pct_checagem", "pct_checagem_imputado",
    "divulgada_vespera", "amostra_tse", "amostra_base_original",
    "amostra_implausivel_no_tse", "metodologia_encontrada",
]


# ============================================================
# 2. LEITURA E LIMPEZA DAS PESQUISAS
#
# Sete conferencias feitas sobre o arquivo de origem, cada uma comentada no
# ponto onde e aplicada.
# ============================================================
def padroniza_cenario(descricao):
    """Extrai (numero do cenario, turno) das 21 grafias encontradas.

    Variam genero (estimulado/estimulada), ordem dos termos, maiuscula, e ha
    erros de digitacao: 'cenario 1.-', 'cenario 1-', espaco duplo,
    '-2 turno'. Todas estas caem no mesmo padrao:
        'cenario 1 - estimulado - 1o turno'
        'cenario 1 - 1o turno - estimulada'
        'Cenario  1.- estimulada -2o turno'
    """
    import re
    texto = re.sub(r"[^a-z0-9]+", " ", nb.sem_acento(descricao)).strip()
    numero = re.search(r"cenario\s*(\d+)", texto)
    turno = re.search(r"(\d)\s*o?\s*turno", texto)
    return (int(numero.group(1)) if numero else np.nan,
            int(turno.group(1)) if turno else np.nan)


def carrega_pesquisas():
    """Le, limpa e filtra. Devolve (pesquisas, descartados)."""
    df = pd.read_csv(PATH_PESQUISAS, encoding="latin-1")
    bruto = len(df)
    descartes = []

    def descartar(dados, mascara, motivo):
        if mascara.any():
            fora = dados[mascara].copy()
            fora["motivo_descarte"] = motivo
            descartes.append(fora)
        return dados[~mascara].copy()

    # --- (2) 'Unnamed: 11' e um numero_registro quebrado na importacao:
    # 1.590 nulos, e entre os valores aparecem a string 'numero_registro' e
    # '0'. Coincide com numero_registro em 1 linha de 1.758.
    df = df.drop(columns=["Unnamed: 11"], errors="ignore")

    # --- (5) turno_real e identica a turno nas 1.758 linhas
    if "turno_real" in df.columns:
        divergem = int((df["turno"] != df["turno_real"]).sum())
        if divergem == 0:
            df = df.drop(columns=["turno_real"])
        else:
            print(f"  [aviso] turno != turno_real em {divergem} linhas: mantida")

    # --- (6) nome_pesquisa e identica a nome_clean nas 1.758 linhas.
    # nome_tse e diferente (nome completo de urna) e permanece.
    if "nome_pesquisa" in df.columns:
        divergem = int((df["nome_pesquisa"].astype(str)
                        != df["nome_clean"].astype(str)).sum())
        if divergem == 0:
            df = df.drop(columns=["nome_pesquisa"])
        else:
            print(f"  [aviso] nome_pesquisa != nome_clean em {divergem} linhas")

    # --- (3) margem_mais e margem_menos sao a mesma margem. Onde divergem
    # (4 linhas, 2,0 contra 0,0) e erro de digitacao.
    df["margem_erro_pp"] = pd.to_numeric(df["margem_mais"], errors="coerce")
    df["margem_assimetrica"] = (
        pd.to_numeric(df["margem_mais"], errors="coerce")
        != pd.to_numeric(df["margem_menos"], errors="coerce")).astype(int)
    df = df.drop(columns=["margem_mais", "margem_menos"], errors="ignore")

    # --- (4) padroniza as 21 grafias de cenario
    cenarios = df["descricao_cenario"].map(padroniza_cenario)
    df["cenario_num"] = [c[0] for c in cenarios]
    df["cenario_turno"] = [c[1] for c in cenarios]
    df["cenario_padrao"] = [
        f"cenario {int(n)} - {int(t)}o turno" if pd.notna(n) and pd.notna(t)
        else np.nan for n, t in cenarios]

    # --- (1) so pesquisas de 2022
    df["data_referencia"] = pd.to_datetime(df["data_referencia"],
                                           errors="coerce")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = descartar(df, df["data_referencia"].dt.year != ANO,
                   f"data_referencia fora de {ANO}")

    # --- cenario original: MENOR numero dentro de cada pesquisa e turno.
    # Filtrar literalmente por "cenario 1" apagaria 12 combinacoes cujo
    # unico cenario esta numerado como 2, 3, 5 ou 9.
    menor = df.groupby(["id_pesquisa", "turno"])["cenario_num"].transform("min")
    df = descartar(df, df["cenario_num"] != menor, "cenario alternativo")

    df = descartar(df, df["categoria_voto"] != "candidato",
                   "nao e candidato (branco, nulo ou indeciso)")
    df = descartar(df, df["percentual_valido"].isna(),
                   "candidato sem resultado apurado")

    descartados = (pd.concat(descartes, ignore_index=True) if descartes
                   else pd.DataFrame())
    print(f"pesquisas: {bruto} linhas brutas -> {len(df)} apos limpeza")
    return df, descartados


# ============================================================
# 3. MONTAGEM
# ============================================================
def monta(df, met):
    """Junta metodologia, aplica excecoes e deriva as variaveis."""
    df["registro"] = [nb.normaliza_registro(v)[0]
                      for v in df["numero_registro"]]

    colunas = ["NR_CNPJ_EMPRESA", "NM_EMPRESA", "dt_ini", "dt_fim",
               "dt_divulgacao", "QT_ENTREVISTADOS", "VR_PESQUISA",
               "NM_ESTATISTICO_RESP", "telefonica", "presencial",
               "presencial_inferido", "online", "pct_checagem",
               "pct_checagem_imputado", "ev_modo_coleta", "ev_pct_checagem"]
    df, log = nb.liga_metodologia(df, met, "registro", "data", colunas)

    # --- excecoes manuais de modo de coleta (ver MODO_MANUAL) ----------
    df["modo_manual"] = 0
    for registro, modo in MODO_MANUAL.items():
        alvo = df["registro"] == registro
        if not alvo.any():
            continue
        df.loc[alvo, ["telefonica", "presencial", "online"]] = 0
        df.loc[alvo, modo] = 1
        df.loc[alvo, "modo_manual"] = 1

    df["ano"] = ANO
    df = nb.deriva_datas(df, DATAS_ELEICAO, "data")
    df = nb.deriva_amostra(df, col_base="quantidade_entrevistas")
    df = nb.deriva_custo(df)

    # --- variaveis da disputa -------------------------------------------
    # Ver o aviso no cabecalho: no arquivo de origem 'percentual_valido' e o
    # resultado apurado. Tudo vem em proporcao (0-1) e vira pontos aqui.
    df["percentual_real"] = df["percentual_valido"] * 100
    df["percentual_pesq_valido"] = df["percentual_pesquisa_valido"] * 100
    df["vies_pp"] = df["vies"] * 100
    df["abs_vies"] = df["vies_pp"].abs()
    # confere a identidade vies = pesquisa - real
    df["conferencia_vies"] = (df["percentual_pesq_valido"]
                              - df["percentual_real"] - df["vies_pp"]).abs()

    df["cand_grande"] = df["nome_clean"].isin(CAND_GRANDE).astype(int)
    df["competitividade"] = df.groupby(["id_pesquisa", "turno"])[
        "percentual_pesq_valido"].transform(
        lambda s: s.nlargest(2).iloc[0] - s.nlargest(2).iloc[1]
        if len(s) > 1 else np.nan)
    # soma_validos_pesquisa tambem vem em proporcao neste arquivo, ao
    # contrario da base de 2018, onde ja esta em pontos percentuais
    df["prop_indecisos"] = 100 - pd.to_numeric(
        df["soma_validos_pesquisa"], errors="coerce") * 100

    df["cenario_hipotetico"] = ((df["turno"] == 2)
                                & (df["data"] < DATAS_ELEICAO[1])).astype(int)
    return df, log


# ============================================================
# 4. EXECUCAO
# ============================================================
def main():
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    met = nb.carrega_metodologia(PATH_METODOLOGIA, separador=";")
    df, descartados = carrega_pesquisas()
    df, log = monta(df, met)

    if EXCLUIR_CENARIO_HIPOTETICO:
        fora = df[df["cenario_hipotetico"] == 1].copy()
        fora["motivo_descarte"] = "2o turno medido antes de 02/10"
        descartados = pd.concat([descartados, fora], ignore_index=True)
        df = df[df["cenario_hipotetico"] == 0].copy()

    constantes = [c for c in ["cenario_hipotetico"] if df[c].nunique() <= 1]
    df = df.drop(columns=constantes)

    principal, extra = nb.separa_colunas(df, COLUNAS_PRINCIPAIS)
    por_pesquisa = principal.drop_duplicates(subset=["id_pesquisa", "turno"])

    # ---- log -----------------------------------------------------------
    log = list(log)
    if len(descartados):
        log.append({"tipo": "descartes", "registro": "-", "detalhe": "; ".join(
            f"{k}={v}" for k, v in
            descartados["motivo_descarte"].value_counts().items())})
    for registro, modo in MODO_MANUAL.items():
        if (principal["registro"] == registro).any():
            log.append({"tipo": "modo_manual", "registro": registro,
                        "detalhe": f"modo atribuido manualmente: {modo}"})
    for tipo, sub, detalhe in [
            ("sem_metodologia",
             principal[principal["metodologia_encontrada"] == 0],
             "registro nao encontrado no TSE"),
            ("amostra_implausivel",
             principal[principal["amostra_implausivel_no_tse"] == 1],
             "QT_ENTREVISTADOS implausivel: usado valor da base"),
            ("pct_checagem_imputada",
             principal[principal["pct_checagem_imputado"] == 1],
             f"sem checagem declarada: assumido {nb.PCT_CHECAGEM_PADRAO:.0f}%"),
            ("margem_assimetrica", extra[extra["margem_assimetrica"] == 1]
             if "margem_assimetrica" in extra.columns else pd.DataFrame(),
             "margem_mais != margem_menos (erro de digitacao)")]:
        for _, linha in sub.drop_duplicates("id_pesquisa").iterrows():
            log.append({"tipo": tipo,
                        "registro": linha.get("registro", "-"),
                        "detalhe": detalhe})
    ruins = extra[extra["conferencia_vies"] > 0.01] \
        if "conferencia_vies" in extra.columns else pd.DataFrame()
    if len(ruins):
        log.append({"tipo": "vies_inconsistente", "registro": "-",
                    "detalhe": f"{len(ruins)} linhas onde vies != pesquisa - real"})
    log = pd.DataFrame(log)

    colunas_ev = ["id_pesquisa", "registro", "instituto"] + \
                 [c for c in df.columns if c.startswith("ev_")]

    nb.grava(DIR_SAIDA / f"base_completa_{ANO}.xlsx", {
        "base_longa": principal,
        "colunas_extra": extra,
        "base_pesquisa": por_pesquisa,
        "descartados": descartados,
        "evidencias": df[colunas_ev],
        "log_auditoria": log,
    })

    modo_ausente = int(((principal["telefonica"] == 0)
                        & (principal["presencial"] == 0)
                        & (principal["online"] == 0)).sum())
    print(f"ano............................ {ANO}")
    print(f"linhas na base longa........... {len(principal)}")
    print(f"  pesquisas.................... {len(por_pesquisa)}")
    print(f"  registros TSE................ {principal['registro'].nunique()}")
    print(f"  com metodologia.............. "
          f"{principal['metodologia_encontrada'].mean():.0%}")
    print(f"  por turno.................... "
          f"{principal['turno'].value_counts().to_dict()}")
    print(f"dias_ate_eleicao............... "
          f"{principal['dias_ate_eleicao'].min():.0f} a "
          f"{principal['dias_ate_eleicao'].max():.0f}")
    print(f"dias_campo nulos............... {int(principal['dias_campo'].isna().sum())}")
    print(f"amostra corrigida (TSE ruim)... "
          f"{int(principal['amostra_implausivel_no_tse'].sum())}")
    print(f"pct_checagem imputada.......... "
          f"{int(principal['pct_checagem_imputado'].sum())}")
    print(f"modo: telefonica={int(principal['telefonica'].sum())} "
          f"presencial={int(principal['presencial'].sum())} "
          f"online={int(principal['online'].sum())} "
          f"sem modo={modo_ausente}")
    print(f"vies absoluto medio............ {principal['abs_vies'].mean():.2f} pp")


if __name__ == "__main__":
    main()
