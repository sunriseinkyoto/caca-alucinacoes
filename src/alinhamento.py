# -*- coding: utf-8 -*-
"""
Alinhamento local entre predições e gabarito, para ferramentas de diagnóstico.

Reproduz o critério da métrica oficial — sobreposição de spans com IoU >= 0,5,
pareamento guloso um para um pelo maior IoU — sem calcular o score, que é
sempre obtido com o script oficial (src/avaliar_oficial.py).
"""
import csv
from collections import defaultdict

IOU_MIN = 0.5


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    uni = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / uni if uni else 0.0


def carregar_gold(caminho):
    """Lê o gabarito e devolve ``(citações por documento, nível por documento)``."""
    gold = defaultdict(list)
    nivel = {}
    for r in csv.DictReader(open(caminho, encoding="utf-8-sig")):
        d = r["documento_id"]
        nivel[d] = r["nivel"]
        gold[d].append({"span": (int(r["inicio"]), int(r["fim"])),
                        "classe": r["classificacao"],
                        "id": (r["id_canonico"] or "").strip()})
    return gold, nivel


def casar(preds, golds):
    """Pareamento guloso por IoU.

    Returns:
        ``(pares, predições sem par, citações do gabarito sem par)``.
    """
    cands = sorted(((iou(p["span"], g["span"]), i, j)
                    for i, p in enumerate(preds) for j, g in enumerate(golds)
                    if iou(p["span"], g["span"]) >= IOU_MIN), reverse=True)
    up, ug, pares = set(), set(), []
    for _, i, j in cands:
        if i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        pares.append((preds[i], golds[j]))
    return (pares,
            [p for i, p in enumerate(preds) if i not in up],
            [g for j, g in enumerate(golds) if j not in ug])
