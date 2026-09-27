# -*- coding: utf-8 -*-
"""
Teste sobre corpora sintéticos com gabarito conhecido.

Gera corpora com ml/gerar_sinteticos.py — que combina os blocos de texto da
amostra de desenvolvimento, os registros reais da base canônica e o modelo de
ruído de OCR do nível 2 — e avalia o pipeline com a métrica oficial. As
sementes são fixas, e os corpora não são usados para ajustar o sistema.

Uso:
    python testes/teste_sintetico.py <desafio1_bracis.db> <indice.json> <kaggle_metric.py>
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import avaliar_oficial  # noqa: E402
import rodar  # noqa: E402

SEMENTES = [(7, 300), (424242, 400), (90210, 500)]


def main(base, indice, metrica) -> bool:
    km = avaliar_oficial.carregar_metrica(metrica)
    ok = True
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for semente, n in SEMENTES:
            d = Path(tmp) / str(semente)
            subprocess.run([sys.executable, str(RAIZ / "ml" / "gerar_sinteticos.py"),
                            base, str(RAIZ / "templates.json"), indice, str(d),
                            "--n", str(n), "--seed", str(semente)],
                           check=True, stdout=subprocess.DEVNULL)
            rodar.main(str(d / "txt"), indice, str(d / "pred"),
                       str(RAIZ / "templates.json"))
            gold = d / "goldenset.csv"
            n_cit = sum(1 for _ in csv.DictReader(gold.open(encoding="utf-8")))
            total += n_cit
            sol = avaliar_oficial.solution_df(gold)
            sub = avaliar_oficial.submission_df(d / "pred", sol["documento_id"].tolist())
            score = km.avaliar(sol, sub, row_id="documento_id")["score_final"]
            print(f"semente {semente:>6d}: {n} documentos, {n_cit} citações, score = {score:.4f}")
            ok &= abs(score - 1.1) < 1e-9
    print(f"total: {total} citações sintéticas")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:4]) else 1)
