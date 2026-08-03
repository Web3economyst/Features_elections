------------------------------------------------------------------------------
PASTA: 2018
------------------------------------------------------------------------------
Contem o processamento do ano de 2018.

Arquivos:

  - selecao1 (codigo Python)
        Script que faz o MERGE das bases e cria as features do ano de 2018.
        Combina as duas bases abaixo (base_todas_long + pesquisas features).

  - base_todas_long (Excel)
        Base extraida a partir do codigo em R.
        Serve como uma das entradas do script de merge.

  - pesquisas features
        Base com as caracteristicas (features) extraidas dos PDFs baixados
        do TSE. Segunda entrada do script de merge.

  Saida da etapa: base de 2018 com features consolidadas.

Os PDFS podem ser encontrados aqui: https://drive.google.com/drive/folders/1MMcd8l2Hp0OUV2XShF3fKmPXVS6ItPcR?usp=sharing

Nem todas as pesquisas tem PDFS, alguns foram perdidos quando tive problema com a memória, então haverão mais pesquisas com features que PDFs e algumas pesquisas não tinham PDFs


------------------------------------------------------------------------------
PASTA: 2022
------------------------------------------------------------------------------
Contem o processamento do ano de 2022, seguindo a mesma logica de 2018.

Arquivos:


- selecao_20221 (codigo Python)
        Script que faz o MERGE das bases e cria as features do ano de 2022.
        Combina as duas bases abaixo (base_2022_todas + bases_2022_dias_corrigidos).
  
  - base_2022_todas
        Base baixada/gerada a partir do codigo em R.
        Equivalente ao base_todas_long de 2018.

  - bases_2022_dias_corrigidos
        Features extraidas separadamente, com correcao dos dias.
        Equivalente ao "pesquisas features" de 2018.

  Saida da etapa: base de 2022 com features consolidadas.

Os PDFS podem ser encontrados aqui: https://drive.google.com/drive/folders/1vTA3sUMEUGDT92pwo8kJA6NTA7mu0Gmr?usp=sharing

Foram retirados todos os cenários a não ser o cenário 1 e também pesquisas que ficaram null (sem features), mas ficou pendente validar quais não bateram corretamente (Os que estavam na base do poder360 e não foram baixados, pois a base foi todos que estavam na wikipedia de 2022)


==============================================================================
VARIAVÉIS
------------------------------------------------------------------------------
VARIÁVEL DEPENDENTE 

abs_vies = |estimativa da pesquisa − resultado real TSE|
→ Erro absoluto da pesquisa em p.p. de votos válidos. É o que o modelo tenta explicar.

TEMPORAIS — janelas mutuamente exclusivas

dias_ate_eleicao = data da eleição − data final de campo
→ Distância temporal; captura intenção que ainda pode mudar. Núcleo do decaimento do erro.
is_vespera = 1 se 0–3 dias → categoria de referência do modelo de limiar.
is_ultima_semana = 1 se 4–7 dias.
is_15dias = 1 se 8–15 dias.
is_30dias = 1 se 16–30 dias.
is_60dias = 1 se 31–60 dias.
→ As cinco janelas testam se o erro cai de forma contínua ou se há um limiar de confiabilidade a partir de certa proximidade. São exclusivas por construção (cada pesquisa cai em uma só faixa), com >60 dias como referência na especificação padronizada.

PRECISÃO AMOSTRAL

log_amostra = ln(n)
→ Efeito do tamanho da amostra com retorno decrescente (dobrar n não dobra a precisão).
var_phat = p(1−p)/n
→ Variância teórica da estimativa — o "piso" de erro esperado. Alta em disputa equilibrada com amostra pequena. (Correlaciona com log_amostra; as duas competem como "mesma" informação de precisão.)

COMPOSIÇÃO DA DISPUTA / ELEITORADO

prop_indecisos = 100 − soma dos válidos
→ Fatia de branco/nulo/indeciso; mede incerteza sobre voto não declarado.
competitividade = pct 1º − pct 2º
→ Aperto da disputa; quanto menor, mais difícil acertar.
cand_grande = 1 se resultado real ≥ 20%
→ Candidato grande é estimado com mais precisão.

METODOLOGIA DE COLETA

telefonica = dummy (vs. presencial)
→ Modo de coleta; presencial/online entram como complemento (evita a armadilha das dummies).
final_de_semana = dummy
→ Campo cobriu fim de semana (perfil de respondente distinto).
dias_campo = duração da coleta.

DESENHO AMOSTRAL

conglomerados_2estagios → municípios → indivíduos.
conglomerados_3estagios → municípios → setores → indivíduos.
→ Capturam complexidade do desenho, que afeta o erro real vs. o declarado.

OBS: Retiradas até o momento, podem ser reconsideradas

QUALIDADE E CUSTO

auditoria_30 = 1 se auditou ≥30% dos questionários → proxy de controle de qualidade (referência = auditoria de 20%).
valor_pesquisa = custo em R$ → proxy de robustez do investimento.



