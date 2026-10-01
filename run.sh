#!/usr/bin/env bash
# Ponto de entrada único da solução.
#
#   bash run.sh <base.db> <pasta_txt> [pasta_saida]
#
# Constrói o índice a partir da base informada, processa todos os .txt da pasta
# e grava <pasta_saida>/submission.csv (padrão: ./saida), no formato das
# submissões do desafio. Não usa rede nem caminhos absolutos.
#
# A submissão final é a variante principal acrescida dos spans do rotulador
# BERTimbau (fusão estrita), quando os pesos estão em modelo_ner/ (ver
# baixar_modelo.py). Sem os pesos, ou com CACA_SEM_BERT=1, é a variante
# principal, inteiramente determinística.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USO="uso: bash run.sh <base.db> <pasta_txt> [pasta_saida]"
DB="${1:?$USO}"
TXT="${2:?$USO}"
SAIDA="${3:-saida}"
PY="${PYTHON:-python3}"
MODELO="${MODELO_NER:-$RAIZ/modelo_ner}"

[ -f "$DB" ]  || { echo "ERRO: base não encontrada: $DB" >&2; exit 1; }
[ -d "$TXT" ] || { echo "ERRO: pasta de documentos não encontrada: $TXT" >&2; exit 1; }

TRABALHO="$SAIDA/variantes"
ARGS=(--base "$DB" --txt "$TXT" --saida "$TRABALHO"
      --conversor "$RAIZ/oficial/json_to_submission.py")
if [ "${CACA_SEM_BERT:-0}" != "1" ] && [ -f "$MODELO/model.safetensors" ]; then
    ARGS+=(--modelo-ner "$MODELO")
else
    echo "AVISO: rotulador BERTimbau não utilizado (pesos ausentes em $MODELO ou CACA_SEM_BERT=1)"
fi

"$PY" "$RAIZ/gerar_submissao.py" "${ARGS[@]}"

if [ -f "$TRABALHO/submission_bert.csv" ]; then
    FINAL="$TRABALHO/submission_bert.csv"
else
    FINAL="$TRABALHO/submission.csv"
fi
cp "$FINAL" "$SAIDA/submission.csv"
echo
echo "submissão final: $SAIDA/submission.csv (variante: $(basename "$FINAL"))"
