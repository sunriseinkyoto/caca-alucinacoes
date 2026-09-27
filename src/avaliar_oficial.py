# -*- coding: utf-8 -*-
"""
Avaliação com a métrica oficial do desafio (kaggle_metric.py, sem modificação).

Converte o gabarito para o formato ``solution`` e as predições para o formato
``submission`` e executa a mesma função de score do leaderboard, imprimindo o
detalhamento por nível.

Uso:
    python src/avaliar_oficial.py <pasta_jsons> <goldenset.csv> <kaggle_metric.py>
"""
import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def carregar_metrica(caminho):
    spec = importlib.util.spec_from_file_location("kaggle_metric", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solution_df(gold_csv):
    """Gabarito (goldenset.csv) -> DataFrame no formato ``solution`` da métrica."""
    linhas = defaultdict(list)
    nivel = {}
    for r in csv.DictReader(open(gold_csv, encoding="utf-8-sig")):
        d = r["documento_id"]
        nivel[d] = int(r["nivel"])
        # o gabarito pode aceitar mais de um doc_id para a mesma citação
        # (duplicatas do acervo). No CSV eles vêm separados por espaço ou ":";
        # a métrica espera ":".
        ids = ":".join(re.split(r"[\s:;,]+", (r["id_canonico"] or "").strip())) or "-"
        linhas[d].append(f"{r['inicio']},{r['fim']},{r['classificacao']},{ids}")
    return pd.DataFrame([{"documento_id": d, "nivel": nivel[d],
                          "citacoes": "|".join(v)}
                         for d, v in sorted(linhas.items())])


def submission_df(pasta, docs):
    """Pasta de JSONs -> DataFrame no formato ``submission``, com a mesma
    codificação do conversor oficial json_to_submission.py."""
    out = []
    for d in docs:
        arq = Path(pasta) / f"{d}.json"
        partes = []
        if arq.exists():
            for c in json.loads(arq.read_text(encoding="utf-8"))["citacoes"]:
                resol = c.get("resolucao") or {}
                idc = str(resol.get("id_canonico", "") or "").strip() or "-"
                conf = c.get("confianca")
                conf_s = "-" if conf is None else f"{float(conf):.4f}"
                partes.append(f"{int(c['inicio'])},{int(c['fim'])},"
                              f"{c['classificacao']},{idc},{conf_s}")
        out.append({"documento_id": d, "citacoes": "|".join(partes) or "-"})
    return pd.DataFrame(out)


def main(pasta, gold_csv, metric_py):
    km = carregar_metrica(metric_py)
    sol = solution_df(gold_csv)
    sub = submission_df(pasta, sol["documento_id"].tolist())
    r = km.avaliar(sol, sub, row_id="documento_id")

    for nv, d in sorted(r["niveis"].items()):
        print(f"--- Nível {nv} (peso {km.PESOS_NIVEL[nv]:.0f}x) ---")
        for c, f1 in sorted(d["f1_por_classe"].items()):
            print(f"  F1 {c:11s} = {f1:.4f}")
        print(f"  macro_F1 = {d['macro_f1']:.4f}")
        print(f"  tau (inventada->real) = {d['tau']:.4f}  -> s = {d['s']:.4f}")
        print(f"  bônus de calibração   = {d['b']:.4f}")
        print(f"  SCORE DO NÍVEL        = {d['score']:.4f}\n")
    print(f"SCORE FINAL (oficial) = {r['score_final']:.4f}")
    return r


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
