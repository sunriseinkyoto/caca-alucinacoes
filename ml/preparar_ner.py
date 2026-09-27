# -*- coding: utf-8 -*-
"""
Conversão de (txt/ + goldenset.csv) em JSONL para treino do rotulador de spans.

Formato de saída, um objeto por janela de texto:
    {"id": "...", "texto": "...", "entidades": [{"inicio": 0, "fim": 9, "rotulo": "JUR"}]}

Decisões de projeto:

  * Os rótulos são JUR e LEI, e não as três classes do desafio. O modelo não
    tem como saber se um número existe na base canônica; quem decide entre
    real, inventada e incompleta é o resolvedor determinístico.
  * Janelas com sobreposição: os documentos ultrapassam o limite de subtokens
    do encoder, e a sobreposição evita que uma citação seja cortada na
    fronteira; na inferência, as janelas são remontadas.

Uso:
    python ml/preparar_ner.py <pasta_txt> <goldenset.csv> <saida.jsonl> \
           [--janela 1200] [--passo 900]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROTULO = {"jurisprudencia": "JUR", "lei": "LEI"}


def janelas(texto, tam, passo):
    if len(texto) <= tam:
        yield 0, len(texto)
        return
    i = 0
    while i < len(texto):
        yield i, min(i + tam, len(texto))
        if i + tam >= len(texto):
            break
        i += passo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta_txt")
    ap.add_argument("goldenset")
    ap.add_argument("saida")
    ap.add_argument("--janela", type=int, default=1200)
    ap.add_argument("--passo", type=int, default=900)
    a = ap.parse_args()

    gold = defaultdict(list)
    for r in csv.DictReader(open(a.goldenset, encoding="utf-8-sig")):
        gold[r["documento_id"]].append(
            (int(r["inicio"]), int(r["fim"]), ROTULO[r["tipo"]]))

    n_janelas = n_ent = 0
    with open(a.saida, "w", encoding="utf-8") as f:
        for arq in sorted(Path(a.pasta_txt).glob("*.txt")):
            texto = arq.read_text(encoding="utf-8")
            spans = sorted(gold.get(arq.stem, []))
            for k, (ini, fim) in enumerate(janelas(texto, a.janela, a.passo)):
                # só entram citações inteiramente contidas na janela: uma
                # citação cortada ensinaria o modelo a terminar no lugar errado
                ents = [{"inicio": s - ini, "fim": e - ini, "rotulo": r}
                        for s, e, r in spans if s >= ini and e <= fim]
                f.write(json.dumps({"id": f"{arq.stem}#{k}",
                                    "texto": texto[ini:fim],
                                    "entidades": ents},
                                   ensure_ascii=False) + "\n")
                n_janelas += 1
                n_ent += len(ents)
    print(f"{n_janelas} janelas, {n_ent} entidades -> {a.saida}")


if __name__ == "__main__":
    main()
