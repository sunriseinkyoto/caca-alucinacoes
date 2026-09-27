# -*- coding: utf-8 -*-
"""
Inferência do rotulador de spans: modelo treinado -> spans em codepoints.

Usa janela deslizante com sobreposição e remontagem: um span presente em duas
janelas entra uma única vez. A saída é um JSON
``{documento_id: [{inicio, fim, rotulo, score}]}``, consumido por ml/fundir.py.

Uso:
    python ml/inferir_ner.py <modelo/> <pasta_txt> <saida.json> \
           [--janela 900] [--passo 700] [--lote 8]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from texto import carregar as carregar_texto  # noqa: E402

ROTULOS = ["O", "B-JUR", "I-JUR", "B-LEI", "I-LEI"]


def janelas(n, tam, passo):
    if n <= tam:
        return [(0, n)]
    saida, i = [], 0
    while i < n:
        saida.append((i, min(i + tam, n)))
        if i + tam >= n:
            break
        i += passo
    return saida


def decodificar(offsets, ids, probs, desloc):
    """Converte rótulos BIO por token em spans por caractere."""
    spans, atual = [], None
    for (a, b), r, p in zip(offsets, ids, probs):
        if a == b:
            continue
        rot = ROTULOS[r]
        if rot == "O":
            if atual:
                spans.append(atual)
                atual = None
            continue
        pref, tipo = rot.split("-")
        if pref == "B" or atual is None or atual["rotulo"] != tipo:
            if atual:
                spans.append(atual)
            atual = {"inicio": a + desloc, "fim": b + desloc,
                     "rotulo": tipo, "scores": [p]}
        else:
            atual["fim"] = b + desloc
            atual["scores"].append(p)
    if atual:
        spans.append(atual)
    for s in spans:
        s["score"] = sum(s["scores"]) / len(s["scores"])
        del s["scores"]
    return spans


def iou(a, b):
    inter = max(0, min(a["fim"], b["fim"]) - max(a["inicio"], b["inicio"]))
    uni = (a["fim"] - a["inicio"]) + (b["fim"] - b["inicio"]) - inter
    return inter / uni if uni else 0.0


def dedup(spans):
    saida = []
    for s in sorted(spans, key=lambda x: -x["score"]):
        if all(iou(s, t) < 0.5 for t in saida):
            saida.append(s)
    return sorted(saida, key=lambda x: x["inicio"])


def carregar_modelo(caminho):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(caminho, use_fast=True, do_lower_case=False)
    modelo = AutoModelForTokenClassification.from_pretrained(caminho)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(disp).eval()
    return tok, modelo, disp


def rotular(texto, tok, modelo, disp, janela=900, passo=700, lote=8):
    """Spans ``[{inicio, fim, rotulo, score}]`` de um texto, em codepoints."""
    pedacos = janelas(len(texto), janela, passo)
    todos = []
    for i in range(0, len(pedacos), lote):
        grupo = pedacos[i:i + lote]
        cod = tok([texto[x:y] for x, y in grupo], truncation=True,
                  max_length=512, padding=True,
                  return_offsets_mapping=True, return_tensors="pt")
        offs = cod.pop("offset_mapping")
        with torch.no_grad():
            log = modelo(**{k: v.to(disp) for k, v in cod.items()}).logits
        prob = torch.softmax(log, -1)
        ids = prob.argmax(-1).cpu()
        top = prob.max(-1).values.cpu()
        for k, (x, _) in enumerate(grupo):
            todos += decodificar(offs[k].tolist(), ids[k].tolist(),
                                 top[k].tolist(), x)
    return dedup(todos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo")
    ap.add_argument("pasta_txt")
    ap.add_argument("saida")
    ap.add_argument("--janela", type=int, default=900)
    ap.add_argument("--passo", type=int, default=700)
    ap.add_argument("--lote", type=int, default=8)
    a = ap.parse_args()

    tok, modelo, disp = carregar_modelo(a.modelo)
    resultado = {}
    for arq in sorted(Path(a.pasta_txt).glob("*.txt")):
        # mesma leitura do pipeline: texto normalizado, offsets do arquivo original
        fonte = carregar_texto(arq)
        spans = rotular(fonte.proc, tok, modelo, disp, a.janela, a.passo, a.lote)
        for s in spans:
            s["inicio"], s["fim"] = fonte.mapear(s["inicio"], s["fim"])
        resultado[arq.stem] = spans

    Path(a.saida).write_text(json.dumps(resultado, ensure_ascii=False),
                             encoding="utf-8")
    n = sum(len(v) for v in resultado.values())
    print(f"{len(resultado)} documentos, {n} spans -> {a.saida}")


if __name__ == "__main__":
    main()
