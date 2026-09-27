# -*- coding: utf-8 -*-
"""
Calibração da confiança por tipo de evidência.

O bônus de calibração da métrica depende do Brier score sobre os pares
casados. Para um grupo homogêneo de predições, a confiança que minimiza o
Brier é a taxa de acerto do grupo. Este script mede, no gabarito aberto, a
taxa de acerto de cada tipo de evidência registrado pelo extrator (número CNJ
com UF, número curto sem UF, citação vaga, duplicata da base etc.) e sugere os
valores da tabela ``CONF`` de src/extrair.py, usada no modo ``calibrada``.

Uso:
    python src/calibrar.py <pasta_jsons> <goldenset.csv>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alinhamento import carregar_gold, casar  # noqa: E402


def main(pasta, gold_csv):
    gold, _ = carregar_gold(gold_csv)
    acc = defaultdict(lambda: [0, 0])       # evidência -> [acertos, total]
    brier_atual = []

    for doc_id, golds in gold.items():
        arq = Path(pasta) / f"{doc_id}.json"
        if not arq.exists():
            continue
        preds = []
        for c in json.loads(arq.read_text(encoding="utf-8"))["citacoes"]:
            r = c.get("resolucao") or {}
            preds.append({"span": (c["inicio"], c["fim"]),
                          "classe": c["classificacao"],
                          "id": str(r.get("id_canonico") or ""),
                          "conf": c.get("confianca"),
                          "evid": c.get("evidencia", "?")})
        # o Brier da métrica considera apenas os pares casados
        pares, _, _ = casar(preds, golds)
        for p, g in pares:
            ok = p["classe"] == g["classe"] and (
                g["classe"] != "real" or p["id"] in g["id"].replace(":", " ").split())
            acc[p["evid"]][0] += int(ok)
            acc[p["evid"]][1] += 1
            if p["conf"] is not None:
                brier_atual.append((p["conf"] - int(ok)) ** 2)

    print(f"{'evidência':12s} {'acertos':>9s} {'taxa':>8s}  {'CONF sugerida':>14s}")
    for ev, (ok, tot) in sorted(acc.items(), key=lambda kv: -kv[1][1]):
        taxa = ok / tot if tot else 0.0
        print(f"{ev:12s} {ok:4d}/{tot:<4d} {taxa:8.3f}  {taxa:14.3f}")
    if brier_atual:
        b = sum(brier_atual) / len(brier_atual)
        print(f"\nBrier atual = {b:.5f}  ->  bônus = {0.10 * (1 - b):.5f} "
              f"(teto 0,10000)")
    print("\nObservação: uma taxa de 1,000 numa amostra pequena não demonstra "
          "acerto total; no conjunto de avaliação, um teto como min(taxa, 0,97) "
          "é mais prudente.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
