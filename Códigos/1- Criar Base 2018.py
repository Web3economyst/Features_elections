#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base de analise do ciclo de 2018.  ->  python base_analise_2018.py

ENTRADAS
  base_pesquisas_2018.xlsx   formato LARGO (uma linha por pesquisa/cenario,
                             uma coluna por candidato), abas 2018_1T e 2018_2T
  base_metodologias_2018.csv registros do TSE (dados abertos, separador ',')

SAIDA  base_analise_2018.xlsx
  base_longa          unidade de analise: candidato x pesquisa
  colunas_extra       o que nao entra na principal, ligado por id_pesquisa
  base_pesquisa       uma linha por pesquisa
  sem_metodologia     pesquisas sem registro no TSE (as de 2017)
  cenario_hipotetico  2o turno medido antes do 1o
  quadro_indefinido   campo anterior a 11/09 (ver particularidade 1)
  evidencias          trecho do registro que sustenta cada feature
  log_auditoria       colisoes, divergencias e descartes

PARTICULARIDADES DE 2018
  1. Substituicao Lula -> Haddad em 11/09/2018. Antes disso Haddad aparecia
     como substituto hipotetico (media de 4,2% contra 20,3% depois) e o
     eleitorado petista estava distribuido entre os demais e os indecisos.
     TODOS os candidatos ficam distorcidos, nao so Haddad: Marina passa de
     +18,4 para +4,8 de vies medio, Bolsonaro de -11,0 para -8,2. Por isso
     DATA_CORTE_QUADRO remove a pesquisa inteira. Custo: dias_ate_eleicao
     fica restrito a 1-26 dias.
  2. Dois registros digitados errado na base, ambos conferidos no TSE:
     BR-0446 -> BR-04446 (sequencial com 4 digitos) e
     BR-06298 -> BR-06928 (transposicao dos digitos 2 e 9).
  3. Colisao de protocolo: BR-02039 e BR-07829 sao usados por duas empresas
     cada. O nucleo desempata pela data de campo e loga as duas candidatas.
  4. O CSV de 2018 NAO traz DT_DIVULGACAO, entao divulgada_vespera fica
     sempre 0 neste ciclo. A data existe no PDF do registro, nao no dado
     estruturado.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import nucleo_base_analise as nb

# ============================================================
# 1. CONFIGURACAO
# ============================================================
PATH_PESQUISAS = r"C:\Users\Games\Downloads\setembro\Dados\base_pesquisas_2018.xlsx"
PATH_METODOLOGIA = r"C:\Users\Games\Downloads\setembro\Dados\base_metodologias_2018.csv"
DIR_SAIDA = Path(r"C:\Users\Games\Downloads\setembro\Dados")

ANO = 2018
DATAS_ELEICAO = {"1T": pd.Timestamp("2018-10-07"),
                 "2T": pd.Timestamp("2018-10-28")}

# % de votos validos apurados pelo TSE
RESULTADO_REAL = {
    "1T": {"bolsonaro": 46.03, "haddad": 29.28, "gomes": 12.47,
           "alckmin": 4.76, "amoedo": 2.50, "meirelles": 1.20,
           "silva": 1.00, "dias": 0.80},
    "2T": {"bolsonaro": 55.13, "haddad": 44.87},
}

# nome curto -> colunas onde o candidato aparece. A aba do 1o turno e a do
# 2o usam rotulos diferentes para a mesma pessoa.
CANDIDATOS = {
    "haddad": ["Fernando_Haddad", "Haddad"],
    "bolsonaro": ["Jair_Bolsonaro", "Bolsonaro"],
    "gomes": ["Ciro_Gomes"],
    "silva": ["Marina_Silva"],
    "meirelles": ["Henrique_Meirelles"],
    "dias": ["Alvaro_Dias"],
    "alckmin": ["Geraldo_Alckmin"],
    "amoedo": ["Joao_Amoedo"],
}

# Quem chegou ao 2o turno. Definicao EXOGENA de proposito: usar um limiar
# sobre o percentual da propria pesquisa tornaria a variavel funcao do
# mesmo numero que forma a dependente.
CAND_GRANDE = {"haddad", "bolsonaro"}

DATA_CORTE_QUADRO = pd.Timestamp("2018-09-11")      # particularidade 1
# Numeros de registro digitados errado na base de pesquisas (ver
# particularidade 2). Cada correcao foi conferida no CSV do TSE.
CORRECOES_REGISTRO = {
    # sequencial com 4 digitos em vez de 5
    "BR-0446/2018": "BR-04446/2018",
    # transposicao de digitos (06298 -> 06928). Conferido: BR-06928 e do
    # RealTime Big Data, campo em 28-29/09 com 3.200 entrevistas, e e a
    # UNICA pesquisa do instituto com essas datas no CSV do TSE.
    "BR-06298/2018": "BR-06928/2018",
}

EXCLUIR_CENARIO_HIPOTETICO = True
EXCLUIR_QUADRO_INDEFINIDO = True

# Colunas da aba principal, nesta ordem. id_pesquisa fica porque e a chave
# de agrupamento do erro-padrao cluster e do GroupKFold; sem ela a analise
# nao roda.
COLUNAS_PRINCIPAIS = [
    "id_pesquisa", "ano", "turno", "registro", "instituto", "cnpj",
    "empresa_tse", "data_fim_campo", "dias_ate_eleicao", "dias_campo",
    "final_de_semana", "nome_clean", "pct", "soma_rt", "percentual_valido",
    "percentual_real", "vies", "abs_vies", "cand_grande", "prop_indecisos",
    "competitividade", "amostra", "amostra_origem", "divergencia_amostra",
    "log_amostra", "margem_erro_pp", "valor_pesquisa", "telefonica",
    "presencial", "presencial_inferido", "pct_checagem",
]


# ============================================================
# 2. LEITURA DAS PESQUISAS (formato largo)
# ============================================================
def carrega_pesquisas():
    """Empilha as abas do arquivo e normaliza o numero de registro."""
    arquivo = pd.ExcelFile(PATH_PESQUISAS)
    pesquisas = pd.concat(
        [pd.read_excel(PATH_PESQUISAS, sheet_name=aba).assign(aba_origem=aba)
         for aba in arquivo.sheet_names], ignore_index=True, sort=False)

    pesquisas["registro_bruto"] = pesquisas["registro_tse"]
    pesquisas["registro_tse"] = pesquisas["registro_tse"].replace(
        CORRECOES_REGISTRO)
    normalizado = [nb.normaliza_registro(v) for v in pesquisas["registro_tse"]]
    pesquisas["registro"] = [x[0] for x in normalizado]
    pesquisas["registro_suspeito"] = [x[1] for x in normalizado]

    pesquisas["turno"] = pesquisas["turno"].astype(str).str.strip()
    pesquisas["data_fim_campo"] = pesquisas["periodo"].map(
        nb.data_fim_do_periodo)
    return pesquisas


# ============================================================
# 3. MONTAGEM (nivel da pesquisa)
# ============================================================
def monta_pesquisas(pesquisas, met):
    """Junta metodologia e deriva as variaveis de desenho."""
    colunas = ["NR_CNPJ_EMPRESA", "NM_EMPRESA", "dt_ini", "dt_fim",
               "dt_divulgacao", "QT_ENTREVISTADOS", "VR_PESQUISA",
               "NM_CONTRATANTE", "NM_ESTATISTICO_RESP", "telefonica",
               "presencial", "presencial_inferido", "online", "pct_checagem",
               "pct_checagem_imputado", "ev_modo_coleta", "ev_pct_checagem"
               ] + nb.CAMPOS_TEXTO

    df, log = nb.liga_metodologia(pesquisas, met, "registro",
                                  "data_fim_campo", colunas)

    df["ano"] = ANO
    df["turno_num"] = df["turno"].map({"1T": 1, "2T": 2})
    # a data do TSE cobre as pesquisas cujo texto de 'periodo' nao parseou
    df["data_fim_campo"] = df["data_fim_campo"].fillna(df["dt_fim"])

    df = nb.deriva_datas(df, DATAS_ELEICAO, "data_fim_campo")
    df = nb.deriva_amostra(df, col_base="amostra")
    df = nb.deriva_custo(df)

    # ---- marcadores de casos que nao sao previsao comparavel ----------
    df["cenario_hipotetico"] = (
        (df["turno"] == "2T")
        & (df["data_fim_campo"] < DATAS_ELEICAO["1T"])).astype(int)
    df["quadro_indefinido"] = (
        0 if DATA_CORTE_QUADRO is None
        else (df["data_fim_campo"] < DATA_CORTE_QUADRO).astype(int))

    # id unico: registro nao basta porque um registro comporta varios
    # cenarios dentro do mesmo turno
    df.insert(0, "id_pesquisa",
              df["registro"].fillna("SEM-REG") + "|" + df["turno"] + "|"
              + df.groupby(["registro", "turno"]).cumcount().add(1).astype(str))
    return df, log


# ============================================================
# 4. FORMATO LONGO (nivel candidato x pesquisa)
# ============================================================
def monta_base_longa(pesquisas):
    """Converte de largo para longo e calcula o vies de cada candidato.

    percentual_valido renormaliza o percentual bruto sobre a soma dos
    candidatos (soma_rt), para ficar comparavel ao resultado apurado, que e
    em votos validos. Sem isso toda pesquisa pareceria subestimar todo
    mundo, porque os brutos incluem indecisos e brancos.
    """
    linhas = []
    for _, pesquisa in pesquisas.iterrows():
        reais = RESULTADO_REAL.get(pesquisa["turno"], {})

        brutos = {}
        for candidato, origens in CANDIDATOS.items():
            if candidato not in reais:
                continue
            for coluna in origens:
                valor = pd.to_numeric(pesquisa.get(coluna), errors="coerce")
                if pd.notna(valor):
                    brutos[candidato] = float(valor)
                    break
        if not brutos:
            continue

        soma = sum(brutos.values())
        ordenado = sorted(brutos.values(), reverse=True)
        competitividade = (ordenado[0] - ordenado[1] if len(ordenado) > 1
                           else np.nan)

        for candidato, bruto in brutos.items():
            valido = 100 * bruto / soma if soma else np.nan
            real = reais[candidato]
            linhas.append({
                "id_pesquisa": pesquisa["id_pesquisa"],
                "registro": pesquisa["registro"],
                "ano": pesquisa["ano"], "turno": pesquisa["turno_num"],
                "instituto": pesquisa["instituto"],
                "cnpj": pesquisa["NR_CNPJ_EMPRESA"],
                "empresa_tse": pesquisa["NM_EMPRESA"],
                "data_fim_campo": pesquisa["data_fim_campo"],
                "dias_ate_eleicao": pesquisa["dias_ate_eleicao"],
                "dias_campo": pesquisa["dias_campo"],
                "final_de_semana": pesquisa["final_de_semana"],
                "nome_clean": candidato,
                "pct": bruto, "soma_rt": soma,
                "percentual_valido": valido, "percentual_real": real,
                "vies": valido - real, "abs_vies": abs(valido - real),
                "cand_grande": int(candidato in CAND_GRANDE),
                "prop_indecisos": pd.to_numeric(
                    pesquisa.get("Abst_Nao_Decid"), errors="coerce"),
                "competitividade": competitividade,
                "amostra": pesquisa["amostra"],
                "amostra_tse": pesquisa["amostra_tse"],
                "amostra_base_original": pesquisa["amostra_base_original"],
                "amostra_origem": pesquisa["amostra_origem"],
                "divergencia_amostra": pesquisa["divergencia_amostra"],
                "amostra_implausivel_no_tse":
                    pesquisa["amostra_implausivel_no_tse"],
                "log_amostra": pesquisa["log_amostra"],
                "margem_erro_pp": pd.to_numeric(
                    pesquisa.get("margem_erro_pp"), errors="coerce"),
                "valor_pesquisa": pesquisa["valor_pesquisa"],
                "custo_por_entrevista": pesquisa["custo_por_entrevista"],
                "telefonica": pesquisa["telefonica"],
                "presencial": pesquisa["presencial"],
                "presencial_inferido": pesquisa["presencial_inferido"],
                "pct_checagem": pesquisa["pct_checagem"],
                "pct_checagem_imputado": pesquisa["pct_checagem_imputado"],
                "divulgada_vespera": pesquisa["divulgada_vespera"],
                "dias_divulgacao_ate_eleicao":
                    pesquisa["dias_divulgacao_ate_eleicao"],
                "cenario_hipotetico": pesquisa["cenario_hipotetico"],
                "quadro_indefinido": pesquisa["quadro_indefinido"],
                "metodologia_encontrada": pesquisa["metodologia_encontrada"],
            })
    return pd.DataFrame(linhas)


# ============================================================
# 5. EXECUCAO
# ============================================================
def main():
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    met = nb.carrega_metodologia(PATH_METODOLOGIA, separador=",")
    pesquisas, log = monta_pesquisas(carrega_pesquisas(), met)
    longa = monta_base_longa(pesquisas)

    # ---- separa o que sai da analise, sempre com destino visivel -------
    sem_met = pesquisas[pesquisas["metodologia_encontrada"] == 0]
    hipoteticas = pesquisas[(pesquisas["metodologia_encontrada"] == 1)
                            & (pesquisas["cenario_hipotetico"] == 1)]
    indefinidas = pesquisas[(pesquisas["metodologia_encontrada"] == 1)
                            & (pesquisas["cenario_hipotetico"] == 0)
                            & (pesquisas["quadro_indefinido"] == 1)]

    manter = longa["metodologia_encontrada"] == 1
    if EXCLUIR_CENARIO_HIPOTETICO:
        manter &= longa["cenario_hipotetico"] == 0
    if EXCLUIR_QUADRO_INDEFINIDO:
        manter &= longa["quadro_indefinido"] == 0
    longa = longa[manter].copy()

    # com o filtro ligado essas colunas ficam constantes: saem da saida
    constantes = [c for c in ["metodologia_encontrada", "cenario_hipotetico",
                              "quadro_indefinido"] if longa[c].nunique() <= 1]
    longa = longa.drop(columns=constantes)

    principal, extra = nb.separa_colunas(longa, COLUNAS_PRINCIPAIS)
    por_pesquisa = principal.drop_duplicates(subset=["id_pesquisa"])

    # ---- log ----------------------------------------------------------
    log = list(log)
    for tipo, sub, detalhe in [
            ("sem_metodologia", sem_met, "registro nao encontrado no TSE"),
            ("registro_suspeito", pesquisas[pesquisas["registro_suspeito"]],
             "sequencial sem 5 digitos: zero-padding e chute"),
            ("divergencia_amostra",
             pesquisas[pesquisas["divergencia_amostra"] == 1],
             "amostra da base difere de QT_ENTREVISTADOS (vale o TSE)"),
            ("amostra_implausivel",
             pesquisas[pesquisas["amostra_implausivel_no_tse"] == 1],
             "QT_ENTREVISTADOS implausivel: usado valor da base"),
            ("cenario_hipotetico", hipoteticas,
             "2o turno medido antes do 1o"),
            ("quadro_indefinido", indefinidas,
             f"campo anterior a {DATA_CORTE_QUADRO.date()}"),
            ("pct_checagem_imputada",
             pesquisas[pesquisas["pct_checagem_imputado"] == 1],
             f"sem checagem declarada: assumido {nb.PCT_CHECAGEM_PADRAO:.0f}%")]:
        for _, linha in sub.drop_duplicates("id_pesquisa").iterrows():
            log.append({"tipo": tipo, "registro": linha["registro"],
                        "detalhe": detalhe})
    log = pd.DataFrame(log)

    colunas_ev = ["id_pesquisa", "registro", "instituto"] + \
                 [c for c in pesquisas.columns if c.startswith("ev_")]
    sem_texto = [c for c in nb.CAMPOS_TEXTO + [c for c in pesquisas.columns
                                               if c.startswith("ev_")]]

    nb.grava(DIR_SAIDA / f"base_completa_{ANO}.xlsx", {
        "base_longa": principal,
        "colunas_extra": extra,
        "base_pesquisa": por_pesquisa,
        "sem_metodologia": sem_met.drop(columns=sem_texto, errors="ignore"),
        "cenario_hipotetico": hipoteticas.drop(columns=sem_texto, errors="ignore"),
        "quadro_indefinido": indefinidas.drop(columns=sem_texto, errors="ignore"),
        "evidencias": pesquisas[colunas_ev],
        "log_auditoria": log,
    })

    print(f"ano............................ {ANO}")
    print(f"pesquisas lidas................ {len(pesquisas)}")
    print(f"  sem metodologia no TSE....... {len(sem_met)}")
    print(f"  cenario hipotetico........... {len(hipoteticas)}")
    print(f"  quadro indefinido............ {len(indefinidas)}")
    print(f"colisoes de protocolo.......... "
          f"{int((log['tipo'] == 'colisao_protocolo').sum()) if len(log) else 0}")
    print(f"linhas na base longa........... {len(principal)}")
    print(f"  pesquisas.................... {principal['id_pesquisa'].nunique()}")
    print(f"  dias_ate_eleicao............. {principal['dias_ate_eleicao'].min():.0f}"
          f" a {principal['dias_ate_eleicao'].max():.0f}")
    print(f"vies absoluto medio............ {principal['abs_vies'].mean():.2f} pp")


if __name__ == "__main__":
    main()
