# -*- coding: utf-8 -*-
"""
Relatório de erros do pipeline, agrupados por tipo.

Alinha predições e gabarito com o mesmo critério da métrica oficial e lista
trocas de classe (inclusive ``id_canonico`` divergente), falsos positivos e
citações não detectadas. É a ferramenta de diagnóstico para qualquer
alteração no extrator.

Uso:
    python src/analisar_erros.py <pasta_jsons> <goldenset.csv> <pasta_txt>
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alinhamento import carregar_gold, casar  # noqa: E402
from texto import ler_bruto  # noqa: E402


def main(pasta, gold_csv, pasta_txt):
    gold, nivel = carregar_gold(gold_csv)
    trocas, fps, fns = [], [], []

    for doc_id, golds in sorted(gold.items()):
        nv = nivel[doc_id]
        arq = Path(pasta) / f"{doc_id}.json"
        txt = ler_bruto(Path(pasta_txt) / f"{doc_id}.txt")
        preds = []
        if arq.exists():
            for c in json.loads(arq.read_text(encoding="utf-8"))["citacoes"]:
                r = c.get("resolucao") or {}
                preds.append({"span": (c["inicio"], c["fim"]),
                              "classe": c["classificacao"],
                              "id": str(r.get("id_canonico") or ""),
                              "conf": c.get("confianca"),
                              "trecho": c["trecho"]})
        pares, sobra_p, sobra_g = casar(preds, golds)
        for p, g in pares:
            if p["classe"] != g["classe"]:
                trocas.append((nv, doc_id, g["classe"], p["classe"], p["trecho"]))
            elif g["classe"] == "real" and p["id"] not in g["id"].replace(":", " ").split():
                trocas.append((nv, doc_id, "ID DIVERGENTE",
                               f"{p['id']} != {g['id']}", p["trecho"]))
        for p in sobra_p:
            fps.append((nv, doc_id, p["classe"], p["trecho"]))
        for g in sobra_g:
            fns.append((nv, doc_id, g["classe"],
                        txt[g["span"][0]:g["span"][1]]))

    def bloco(titulo, itens, cols):
        print(f"\n{'=' * 70}\n{titulo}  ({len(itens)})\n{'=' * 70}")
        for it in sorted(itens):
            print("  N" + it[0], " | ".join(str(x) for x in it[1:cols]),
                  "|", repr(it[cols]))

    bloco("TROCA DE CLASSE (span detectado, classe ou id incorretos)", trocas, 4)
    bloco("FALSO POSITIVO (span ausente do gabarito)", fps, 3)
    bloco("NÃO DETECTADO (citação do gabarito sem predição)", fns, 3)

    print(f"\n{'=' * 70}\nRESUMO")
    print("  trocas de classe :", Counter(f"{t[2]}->{t[3]}" for t in trocas))
    print("  falsos positivos :", Counter(f"N{f[0]}/{f[2]}" for f in fps))
    print("  não detectados   :", Counter(f"N{f[0]}/{f[2]}" for f in fns))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
