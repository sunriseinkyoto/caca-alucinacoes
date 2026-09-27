# -*- coding: utf-8 -*-
"""
Rotulador simulado, para analisar a fusão sem um modelo treinado.

Produz o mesmo formato de ml/inferir_ner.py a partir do gabarito: recupera a
fração ``--recall`` das citações, com bordas perturbadas como as de um modelo
real, e acrescenta a fração ``--ruido`` de spans espúrios. Foi o instrumento
que localizou o ponto de equilíbrio da fusão antes do treino, posteriormente
confirmado pelo BERTimbau treinado (ver ml/README.md).

Uso:
    python ml/ner_simulado.py <goldenset.csv> <pasta_txt> <saida.json> \
           [--recall 0.85] [--ruido 0.05] [--seed 3]
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("goldenset")
    ap.add_argument("pasta_txt")
    ap.add_argument("saida")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--ruido", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    gold = defaultdict(list)
    for r in csv.DictReader(open(a.goldenset, encoding="utf-8-sig")):
        gold[r["documento_id"]].append((int(r["inicio"]), int(r["fim"])))

    saida = {}
    for arq in sorted(Path(a.pasta_txt).glob("*.txt")):
        texto = arq.read_text(encoding="utf-8")
        spans = []
        for ini, fim in gold.get(arq.stem, []):
            if rng.random() > a.recall:
                continue                      # citação não recuperada
            # borda perturbada: um rotulador real raramente acerta o caractere exato
            di, df = rng.randint(-3, 2), rng.randint(-2, 3)
            i2, f2 = max(0, ini + di), min(len(texto), fim + df)
            if f2 - i2 < 4:
                i2, f2 = ini, fim
            spans.append({"inicio": i2, "fim": f2, "rotulo": "JUR",
                          "score": round(rng.uniform(0.88, 0.999), 4)})
        n_ruido = int(len(gold.get(arq.stem, [])) * a.ruido + rng.random())
        for _ in range(n_ruido):
            i = rng.randint(0, max(0, len(texto) - 40))
            spans.append({"inicio": i, "fim": i + rng.randint(8, 35),
                          "rotulo": "JUR",
                          "score": round(rng.uniform(0.85, 0.97), 4)})
        saida[arq.stem] = sorted(spans, key=lambda s: s["inicio"])

    Path(a.saida).write_text(json.dumps(saida, ensure_ascii=False),
                             encoding="utf-8")
    print(f"{sum(len(v) for v in saida.values())} spans simulados -> {a.saida}")


if __name__ == "__main__":
    main()
