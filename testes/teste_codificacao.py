# -*- coding: utf-8 -*-
"""
Teste de robustez à codificação dos arquivos de entrada.

A métrica alinha spans por codepoints do texto exatamente como distribuído.
Este teste gera variantes da amostra de desenvolvimento com quebras de linha
CRLF, marca de ordem de bytes (BOM), acentos decompostos (NFD) e as três
combinadas, remapeia o gabarito para cada texto transformado e exige que o
score oficial permaneça no teto.

Uso:
    python testes/teste_codificacao.py <pasta_txt> <goldenset.csv> <indice.json> <kaggle_metric.py>
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import avaliar_oficial  # noqa: E402
import rodar  # noqa: E402
from texto import ler_bruto, normalizar  # noqa: E402

VARIANTES = {
    "LF (original)": lambda t: t,
    "CRLF": lambda t: t.replace("\n", "\r\n"),
    "BOM": lambda t: "﻿" + t,
    "NFD": lambda t: unicodedata.normalize("NFD", t),
    "CRLF + BOM + NFD": lambda t: "﻿" + unicodedata.normalize("NFD", t).replace("\n", "\r\n"),
}


def preparar(nome, transformar, pasta_txt, gold_csv, destino: Path):
    (destino / "txt").mkdir(parents=True)
    originais = {p.stem: ler_bruto(p) for p in Path(pasta_txt).glob("*.txt")}
    fontes = {}
    for doc, texto in originais.items():
        novo = transformar(texto)
        (destino / "txt" / f"{doc}.txt").write_bytes(novo.encode("utf-8"))
        fontes[doc] = normalizar(novo)
        assert fontes[doc].proc == texto, f"{nome}: normalização não inverte a variante"
    linhas = list(csv.DictReader(open(gold_csv, encoding="utf-8-sig")))
    for r in linhas:
        a, b = fontes[r["documento_id"]].mapear(int(r["inicio"]), int(r["fim"]))
        r["inicio"], r["fim"] = a, b
    with (destino / "gold.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def main(pasta_txt, gold_csv, indice, metrica) -> bool:
    km = avaliar_oficial.carregar_metrica(metrica)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for i, (nome, fn) in enumerate(VARIANTES.items()):
            d = Path(tmp) / str(i)
            preparar(nome, fn, pasta_txt, gold_csv, d)
            rodar.main(str(d / "txt"), indice, str(d / "pred"),
                       str(RAIZ / "templates.json"))
            sol = avaliar_oficial.solution_df(d / "gold.csv")
            sub = avaliar_oficial.submission_df(d / "pred", sol["documento_id"].tolist())
            score = km.avaliar(sol, sub, row_id="documento_id")["score_final"]
            print(f"{nome:18s} score = {score:.4f}")
            ok &= abs(score - 1.1) < 1e-9
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:5]) else 1)
