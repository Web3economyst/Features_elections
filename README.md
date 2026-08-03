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


