# Material da organização

Esta pasta recebe o material distribuído na aba *Data* do desafio. Esse material
não é versionado neste repositório.

| arquivo | uso |
|---|---|
| `txt/` | documentos de entrada (`<documento_id>.txt`) |
| `desafio1_bracis.db` | base canônica; usada para construir o índice |
| `kaggle_metric.py` | métrica oficial, usada sem modificação na avaliação |
| `goldenset.csv` | gabarito da amostra de desenvolvimento (opcional; apenas para avaliação) |

Para o conjunto de avaliação, basta substituir o conteúdo de `txt/`, ou indicar
outra pasta com `python gerar_submissao.py --txt <pasta>`.

Os resultados publicados usam a versão final do material: base com 1.014
registros e gabarito `goldenset_offsets.csv`, com 192 citações. Esse gabarito
deve ser copiado para esta pasta como `goldenset.csv`.

O conversor oficial já está no repositório, em `oficial/`. Para a avaliação
final, nada precisa ser copiado para esta pasta: `run.sh` recebe os caminhos do
`.db` e da pasta de documentos (ver o README).
