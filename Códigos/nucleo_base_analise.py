#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NUCLEO -- ferramentas compartilhadas pelos dois ciclos eleitorais.

Nao rode este arquivo. Rode base_analise_2018.py ou base_analise_2022.py.

Aqui fica o que NAO depende do ano: normalizacao de numero de registro,
leitura da metodologia do TSE, extracao das features a partir do texto,
resolucao de colisao de protocolo, derivacao de amostra e datas, e escrita
do Excel. O que muda entre ciclos (datas da eleicao, resultado apurado,
formato do arquivo de pesquisas) fica nos arquivos de ano.

ORGANIZACAO
  secao 1  normalizacao de texto e chaves
  secao 2  padroes de texto (o "jogo de palavras")
  secao 3  extracao das features metodologicas
  secao 4  metodologia do TSE e colisao de protocolo
  secao 5  derivacoes comuns (amostra, datas, custo)
  secao 6  escrita do Excel
"""

import re
import unicodedata

import numpy as np
import pandas as pd

# Colunas de texto livre do registro do TSE.
CAMPOS_TEXTO = ["DS_METODOLOGIA_PESQUISA", "DS_PLANO_AMOSTRAL",
                "DS_SISTEMA_CONTROLE"]

MESES = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5,
         "junho": 6, "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10,
         "novembro": 11, "dezembro": 12}

# Amostra abaixo disso e erro de digitacao no registro, nao pesquisa real.
AMOSTRA_MINIMA_PLAUSIVEL = 300

# Assumido quando o registro nao declara percentual de checagem. E imputacao,
# nao medicao: pct_checagem_imputado marca quais linhas usaram este valor.
PCT_CHECAGEM_PADRAO = 20.0

TRAVESSOES = "\u2010\u2011\u2012\u2013\u2014\u2015"


# ============================================================
# 1. NORMALIZACAO DE TEXTO E CHAVES
# ============================================================
def sem_acento(texto):
    """Minusculas e sem acento, para os regex nao dependerem de grafia."""
    return "".join(c for c in unicodedata.normalize("NFD", str(texto))
                   if unicodedata.category(c) != "Mn").lower()


def limpa_texto(valor):
    """Normaliza um campo de texto do TSE. Devolve '' quando vazio."""
    if pd.isna(valor) or str(valor).strip() in ("", "#NULO#"):
        return ""
    # \u00bf (o caractere ¿) e lixo recorrente de encoding nos registros
    limpo = unicodedata.normalize("NFKC", str(valor).replace("\u00bf", " "))
    return re.sub(r"\s+", " ", limpo).strip()


def normaliza_registro(valor):
    """Converte qualquer grafia do protocolo para 'BR-00005/2018'.

    Aceita 'BR000052018' (formato do CSV do TSE), 'BR-00005/2018',
    'BR- 5/2018' e variantes com travessao unicode.

    Devolve (registro, suspeito). suspeito=True quando o sequencial nao tem
    5 digitos: nesse caso o zero-padding e chute e precisa de conferencia
    manual -- foi assim que 'BR-0446/2018' apareceu, sendo BR-04446/2018.
    """
    if pd.isna(valor):
        return None, False
    texto = str(valor).upper()
    for travessao in TRAVESSOES:
        texto = texto.replace(travessao, "-")
    texto = re.sub(r"[^A-Z0-9]", "", texto)

    casa = re.match(r"^BR(\d+?)(\d{4})$", texto)
    if not casa:
        return None, True
    sequencial, ano = casa.group(1), casa.group(2)
    return f"BR-{int(sequencial):05d}/{ano}", len(sequencial) != 5


def data_fim_do_periodo(periodo):
    """Extrai a data final de campo do texto livre da coluna 'periodo'.

    Formatos tratados:
      '8-11 de dezembro de 2017'      -> 2017-12-11
      '21 de junho de 2018'           -> 2018-06-21
      '2930 de setembro de 2018'      -> 2018-09-30
      '28 de setembro a 3 de outubro' -> pega o ultimo par dia+mes

    O caso '2930' acontece quando o travessao se perde na exportacao para
    CSV: o codigo parte o bloco de digitos ao meio e fica com a 2a metade.
    """
    if pd.isna(periodo):
        return pd.NaT
    texto = sem_acento(periodo)
    for travessao in TRAVESSOES:
        texto = texto.replace(travessao, "-")

    pares = re.findall(r"(\d{1,4})\s*(?:de\s+)?(" + "|".join(MESES) + r")",
                       texto)
    ano = re.search(r"(20\d{2})", texto)
    if not pares or not ano:
        return pd.NaT

    dia_texto, mes = pares[-1]
    if len(dia_texto) > 2:
        dia = int(dia_texto[len(dia_texto) // 2:])
    elif len(dia_texto) == 2 and int(dia_texto) > 31:
        dia = int(dia_texto[1])
    else:
        dia = int(dia_texto)

    try:
        return pd.Timestamp(year=int(ano.group(1)), month=MESES[mes], day=dia)
    except ValueError:
        return pd.NaT


def para_data(serie):
    """Converte para data aceitando os dois formatos que o TSE usa.

    O CSV de 2018 traz '01/10/2018' e o de 2022 traz ISO
    ('2022-04-27 00:00:00'). Fixar um formato faz o outro virar NaT em
    silencio -- foi assim que dias_campo ficou vazio em 2022 sem nenhum
    aviso, ate a conferencia manual encontrar.
    """
    dia_primeiro = pd.to_datetime(serie, format="%d/%m/%Y", errors="coerce")
    return dia_primeiro.fillna(pd.to_datetime(serie, errors="coerce"))


# ============================================================
# 2. PADROES DE TEXTO
#
# Cada padrao e deliberadamente estreito. O risco aqui nao e deixar de
# capturar -- isso vira imputacao ou NaN e aparece no log -- e sim capturar
# errado, que passa despercebido.
# ============================================================
PAD_TELEFONICA = (r"telefonic|por telefone|telefonia (fixa|movel)|"
                  r"numeros telefonicos|ligacoes automatizadas|"
                  r"estratificada por ddd")

PAD_PRESENCIAL = (r"entrevistas? pessoa|face-a-face|abordagem pessoal|"
                  r"entrevistas? domiciliar|domiciliares|ponto[s]? de fluxo")

# Questionario web autoaplicado: nao e presencial nem telefonico.
PAD_ONLINE = (r"questionario (?:estruturado )?web|coleta[^.]{0,30}web|"
              r"via web|online")

# "entrevistas pessoais telefonicas": a redacao junta os dois termos, mas a
# coleta e por telefone. Quando aparece, telefonica prevalece.
PAD_PESSOAL_TELEFONICA = r"entrevistas? pessoa\w*\s+telefonic"

# "controla e fiscaliza a aplicacao dos questionarios (por telefone)":
# descreve como o questionario e APLICADO, nao como e checado. Estreito de
# proposito para nao capturar "checados por telefone", que e verificacao de
# uma pesquisa presencial.
PAD_APLICACAO_TELEFONE = r"aplicacao dos questionarios[^.]{0,30}por telefone"

# Marco amostral territorial: sorteio de setor censitario, bairro ou ponto de
# abordagem. Implica coleta presencial -- so se sorteia um setor do IBGE para
# ir a campo nele; pesquisa telefonica sorteia numeros por DDD. Testado nos
# 657 registros de 2018 e nos 1.192 de 2022: nenhum caso de marco territorial
# junto com coleta telefonica.
PAD_TERRITORIAL = (r"setor(?:es)? censitario|sorteio[^.]{0,60}bairro|"
                   r"bairros ou setores|ponto[s]? de abordagem|"
                   r"conglomerados?[^.]{0,40}(?:municipio|bairro|setor)")

# Percentual de questionarios checados. Tres redacoes no texto do TSE:
#   verbo antes  -> "fiscalizacao de cerca de 20% dos questionarios"
#   numero antes -> "serao selecionados 20% dos questionarios aplicados
#                    ... para a verificacao das respostas"
#   por extenso  -> "garantindo um minimo de vinte por cento de checagem"
PAD_CHECAGEM = (r"(?:checa|fiscaliza|auditad|auditor|verifica|supervis|"
                r"conferencia|conferir|retorno a campo)[^.]{0,220}?"
                r"(\d+[,.]?\d*)\s*%")
# Exigir "% dos questionarios" evita capturar percentual solto, como em
# "nao atinjam os 100% em razao das dizimas periodicas".
PAD_CHECAGEM_REVERSO = (r"(\d+[,.]?\d*)\s*%[^.]{0,80}?(?:d[oa]s|de)\s+"
                        r"(?:questionarios|entrevistas|formularios|"
                        r"entrevistados)")
PAD_CHECAGEM_EXTENSO = (r"(?:checa|fiscaliza|auditad|auditor|verifica|"
                        r"supervis|conferencia|conferir)[^.]{0,220}?"
                        r"(dez|quinze|vinte|vinte e cinco|trinta|quarenta|"
                        r"cinquenta)\s+por cento")
EXTENSO_NUM = {"dez": 10, "quinze": 15, "vinte": 20, "vinte e cinco": 25,
               "trinta": 30, "quarenta": 40, "cinquenta": 50}


# ============================================================
# 3. EXTRACAO DAS FEATURES METODOLOGICAS
# ============================================================
def _trecho(original, inicio, fim, antes=60, depois=120):
    """Recorta a evidencia literal em volta do que o regex casou."""
    return original[max(0, inicio - antes):fim + depois].strip()


def extrai_features(linha):
    """Le o texto de um registro e devolve as features + a evidencia.

    Cada feature ganha uma coluna ev_<feature> com o trecho literal que a
    sustenta. E isso que torna a codificacao auditavel sem reabrir o PDF.

    O texto e separado em dois blocos com escopos diferentes:
      DESENHO  (metodologia + plano amostral) -> modo de coleta
      CONTROLE (sistema de controle)          -> percentual de checagem
    Sem essa separacao, "checados por telefone" na secao de controle fazia
    uma pesquisa presencial ser classificada como telefonica.
    """
    desenho_orig = " || ".join(filter(None, [
        limpa_texto(linha.get("DS_METODOLOGIA_PESQUISA")),
        limpa_texto(linha.get("DS_PLANO_AMOSTRAL"))]))
    controle_orig = limpa_texto(linha.get("DS_SISTEMA_CONTROLE"))
    desenho, controle = sem_acento(desenho_orig), sem_acento(controle_orig)
    saida = {}

    # ---- modo de coleta -------------------------------------------------
    online = bool(re.search(PAD_ONLINE, desenho))
    pessoal_tel = bool(re.search(PAD_PESSOAL_TELEFONICA, desenho))
    telefonica = (bool(re.search(PAD_TELEFONICA, desenho)) or pessoal_tel
                  or bool(re.search(PAD_APLICACAO_TELEFONE, controle)))
    presencial = bool(re.search(PAD_PRESENCIAL, desenho)) and not pessoal_tel
    territorial = bool(re.search(PAD_TERRITORIAL, desenho))

    if online:                        # os tres modos sao excludentes
        telefonica = presencial = territorial = False

    # Muitos registros dizem apenas "realizacao de entrevistas", sem o modo,
    # mas detalham o plano amostral. Marco territorial implica presencial: a
    # inferencia vem de um fato declarado (o sorteio de setores ou bairros),
    # nao do silencio. Fica marcada para poder ser revertida num teste de
    # robustez.
    saida["online"] = int(online)
    saida["telefonica"] = int(telefonica)
    saida["presencial"] = int(presencial or (territorial and not telefonica))
    saida["presencial_inferido"] = int(territorial and not presencial
                                       and not telefonica)

    casa = (re.search(PAD_TELEFONICA, desenho)
            or re.search(PAD_PRESENCIAL, desenho)
            or re.search(PAD_TERRITORIAL, desenho))
    saida["ev_modo_coleta"] = (_trecho(desenho_orig, casa.start(), casa.end())
                               if casa else "")

    # ---- percentual de checagem -----------------------------------------
    for padrao, extenso in [(PAD_CHECAGEM, False),
                            (PAD_CHECAGEM_REVERSO, False),
                            (PAD_CHECAGEM_EXTENSO, True)]:
        casa = re.search(padrao, controle)
        if not casa:
            continue
        bruto = casa.group(1)
        saida["pct_checagem"] = float(EXTENSO_NUM[bruto] if extenso
                                      else bruto.replace(",", "."))
        saida["pct_checagem_imputado"] = 0
        saida["ev_pct_checagem"] = _trecho(controle_orig, casa.start(),
                                           casa.end())
        break
    else:
        saida["pct_checagem"] = PCT_CHECAGEM_PADRAO
        saida["pct_checagem_imputado"] = 1
        saida["ev_pct_checagem"] = ""
    return saida


# ============================================================
# 4. METODOLOGIA DO TSE E COLISAO DE PROTOCOLO
# ============================================================
COLUNA_CHAVE_TSE = "NR_PROTOCOLO_REGISTRO"


def _le_csv_tse(caminho, separador=None):
    """Le o CSV do TSE detectando separador e codificacao.

    O arquivo do TSE ja apareceu com ';' e com ',', em latin-1 e em utf-8
    (as vezes com BOM). Com o separador errado o pandas le a linha inteira
    como UMA coluna, e o erro so aparece la na frente, como KeyError de uma
    coluna que "sumiu". Aqui testamos as combinacoes e ficamos com a que
    produz a coluna-chave.
    """
    separadores = [separador] if separador else [";", ",", "\t"]
    codificacoes = ["latin-1", "utf-8-sig", "utf-8"]
    tentativas = []

    for sep in separadores:
        for enc in codificacoes:
            try:
                df = pd.read_csv(caminho, encoding=enc, sep=sep, dtype=str)
            except Exception as erro:
                tentativas.append(f"sep={sep!r} enc={enc}: {type(erro).__name__}")
                continue
            if COLUNA_CHAVE_TSE in df.columns:
                if separador is None:
                    print(f"  [nota] metodologia lida com sep={sep!r}, enc={enc}")
                return df
            tentativas.append(f"sep={sep!r} enc={enc}: leu "
                              f"{len(df.columns)} coluna(s)")

    # se o separador foi imposto e falhou, tenta os demais antes de desistir
    if separador is not None:
        return _le_csv_tse(caminho, separador=None)

    raise SystemExit(
        f"Nao encontrei a coluna {COLUNA_CHAVE_TSE} em:\n  {caminho}\n"
        f"Tentativas: {'; '.join(tentativas)}\n"
        f"Confira se o arquivo e o de dados abertos do TSE "
        f"(pesquisa_eleitoral_ANO.csv), e nao outro export.")


def carrega_metodologia(caminho, separador=None):
    """Le o CSV de dados abertos do TSE e extrai as features de cada registro.

    Colapsa por registro + CNPJ. Um mesmo registro repete no arquivo quando
    ha varios contratantes, mas a metodologia e a mesma. O CNPJ entra na
    chave porque o protocolo NAO e unico: ha numeros usados por duas
    empresas diferentes.

    separador: deixe None para detectar sozinho. So informe se souber.
    """
    met = _le_csv_tse(caminho, separador)

    # o nome da coluna de amostra muda entre os anos
    for nome in ("QT_ENTREVISTADOS", "QT_ENTREVISTADO"):
        if nome in met.columns:
            met = met.rename(columns={nome: "QT_ENTREVISTADOS"})
            break

    met["registro"] = [normaliza_registro(v)[0]
                       for v in met["NR_PROTOCOLO_REGISTRO"]]
    for coluna, novo in [("DT_INICIO_PESQUISA", "dt_ini"),
                         ("DT_FIM_PESQUISA", "dt_fim"),
                         ("DT_DIVULGACAO", "dt_divulgacao")]:
        met[novo] = para_data(met[coluna]) if coluna in met.columns else pd.NaT

    met = (met.sort_values(["registro", "NR_CNPJ_EMPRESA"])
           .groupby(["registro", "NR_CNPJ_EMPRESA"], as_index=False).first())
    features = met.apply(extrai_features, axis=1, result_type="expand")
    return pd.concat([met, features], axis=1)


def escolhe_registro(met, registro, data_alvo):
    """Escolhe a linha da metodologia quando o protocolo colide.

    O protocolo do TSE nao e unico: BR-02039/2018, por exemplo, aparece para
    Vox do Brasil e para PoderData. Sem desempate o merge atribui a
    metodologia errada. Criterio: menor distancia entre a data de campo da
    pesquisa e a registrada no TSE.

    Devolve (indice_escolhido, linhas_de_log). O log traz TODAS as
    candidatas com a distancia, para a escolha poder ser conferida.
    """
    candidatas = met[met["registro"] == registro]
    if candidatas.empty:
        return None, []
    if len(candidatas) == 1:
        return candidatas.index[0], []

    distancia = (candidatas["dt_fim"] - data_alvo).abs()
    escolhido = (distancia.idxmin() if distancia.notna().any()
                 else candidatas.index[0])

    def descreve(i):
        fim = met.loc[i, "dt_fim"]
        dias = distancia.get(i)
        return (f"{met.loc[i, 'NM_EMPRESA']} "
                f"(cnpj {met.loc[i, 'NR_CNPJ_EMPRESA']}, "
                f"fim {fim.date() if pd.notna(fim) else '?'}, "
                f"dist {int(dias.days) if pd.notna(dias) else '?'}d) -> "
                f"{'ESCOLHIDA' if i == escolhido else 'descartada'}")

    log = [{"tipo": "colisao_protocolo", "registro": registro,
            "detalhe": descreve(i)} for i in candidatas.index]
    return escolhido, log


def liga_metodologia(pesquisas, met, col_registro, col_data, colunas):
    """Junta as colunas da metodologia a cada pesquisa, tratando colisoes.

    Devolve (dataframe_com_metodologia, linhas_de_log).
    """
    escolhas, log = {}, []
    for registro, grupo in pesquisas.groupby(col_registro):
        indice, entradas = escolhe_registro(met, registro,
                                            grupo[col_data].max())
        if indice is not None:
            escolhas[registro] = indice
        log.extend(entradas)

    if escolhas:
        ligacao = met.loc[list(escolhas.values()), colunas]
        ligacao.index = list(escolhas.keys())
        pesquisas = pesquisas.join(ligacao, on=col_registro)
    else:
        for coluna in colunas:
            pesquisas[coluna] = np.nan

    pesquisas["metodologia_encontrada"] = pesquisas["NM_EMPRESA"].notna().astype(int)
    return pesquisas, log


# ============================================================
# 5. DERIVACOES COMUNS
# ============================================================
def deriva_amostra(df, col_base):
    """Define a amostra oficial e guarda o rastro de onde ela veio.

    Regra: o QT_ENTREVISTADOS do TSE prevalece sobre a base de pesquisas.
    Excecao: valores implausiveis. Duas rodadas do PoderData em 2022 estao
    registradas com 3 entrevistados e margem de 2 pp, quando a base traz
    3.500 -- ali o registro do TSE esta corrompido e vale o valor original.
    """
    tse = pd.to_numeric(df["QT_ENTREVISTADOS"], errors="coerce")
    base = pd.to_numeric(df[col_base], errors="coerce")
    tse_ok = tse.where(tse >= AMOSTRA_MINIMA_PLAUSIVEL)

    df["amostra"] = tse_ok.fillna(base)
    df["amostra_tse"] = tse
    df["amostra_base_original"] = base
    df["amostra_origem"] = np.where(tse_ok.notna(), "TSE", "base_original")
    df["amostra_implausivel_no_tse"] = (
        tse.notna() & (tse < AMOSTRA_MINIMA_PLAUSIVEL)).astype(int)
    df["divergencia_amostra"] = (
        base.notna() & tse.notna() & (base != tse)).astype(int)
    df["log_amostra"] = np.log(df["amostra"])
    return df


def deriva_datas(df, datas_eleicao, col_data_fim, col_turno="turno"):
    """Distancia ate a eleicao, duracao do campo e marca de vespera.

    dias_ate_eleicao usa o FIM do campo, nao o inicio. Nem toda base traz a
    mesma definicao -- a de 2022 vinha contando do inicio e ainda tinha dois
    valores impossiveis; o valor recalculado aqui e o que vale.

    divulgada_vespera exige DT_DIVULGACAO, que existe no CSV de 2022 mas NAO
    no de 2018. Sem essa coluna a marca fica sempre 0.
    """
    df["data_eleicao"] = df[col_turno].map(datas_eleicao)
    df["dias_ate_eleicao"] = (df["data_eleicao"] - df[col_data_fim]).dt.days
    # +1 para contar os dois extremos: campo de 3 a 5/out sao 3 dias
    df["dias_campo"] = (df["dt_fim"] - df["dt_ini"]).dt.days + 1
    df["final_de_semana"] = (df[col_data_fim].dt.dayofweek >= 5).astype(int)

    df["dt_divulgacao"] = pd.to_datetime(df["dt_divulgacao"], errors="coerce")
    df["dias_divulgacao_ate_eleicao"] = (
        df["data_eleicao"] - df["dt_divulgacao"]).dt.days
    df["divulgada_vespera"] = (
        df["dias_divulgacao_ate_eleicao"] == 1).astype(int)
    return df


def deriva_custo(df):
    """Valor da pesquisa e custo por entrevista.

    VR_PESQUISA vem no padrao brasileiro ('205.000,00'): tira o ponto de
    milhar e troca a virgula decimal por ponto.
    """
    valor = pd.to_numeric(
        df["VR_PESQUISA"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False), errors="coerce")
    df["valor_pesquisa"] = valor
    df["custo_por_entrevista"] = (valor / df["amostra"]).round(2)
    return df


# ============================================================
# 6. ESCRITA
# ============================================================
def separa_colunas(df, principais):
    """Divide em (aba principal, aba colunas_extra).

    Nada e descartado: o que nao esta na lista vai para colunas_extra,
    ligada por id_pesquisa.
    """
    presentes = [c for c in principais if c in df.columns]
    resto = ["id_pesquisa"] + [c for c in df.columns if c not in presentes]
    return df[presentes].copy(), df[[c for c in resto if c in df.columns]].copy()


def grava(caminho, abas):
    """Grava {nome_da_aba: DataFrame}. Aba vazia vira um aviso legivel."""
    with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
        for nome, df in abas.items():
            if df is None or len(df) == 0:
                df = pd.DataFrame({"info": ["(nenhum registro)"]})
            df.to_excel(writer, sheet_name=nome, index=False)
    print(f"\nSalvo: {caminho}")
