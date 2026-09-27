# -*- coding: utf-8 -*-
"""
Calibração aprendida da confiança (componente experimental).

A tabela ``CONF`` de src/extrair.py calibra por grupo: todas as predições com
o mesmo tipo de evidência recebem a mesma confiança. Este script a substitui
por uma regressão logística sobre os sinais estruturais que o extrator
registra em cada citação — número de dígitos, presença de UF, uso de leitura
alternativa de OCR, camada do índice que resolveu a consulta, número de
candidatos, tokens de classe absorvidos, largura do span —, produzindo uma
probabilidade por citação.

Protocolo: treino no corpus sintético e teste no gabarito real, para que o
ganho medido não seja ajuste ao próprio conjunto de avaliação. Resultado
medido: Brier de 0,0102 para 0,0051 e +0,0005 no score final, limitado pelo
teto do bônus de calibração.

Uso:
    python ml/calibrar_ml.py --treino <pred_sint/> <sint/goldenset.csv> \
                             --teste  <pred_dev/>  <dev/goldenset.csv> \
                             --metrica <kaggle_metric.py> [--saida modelo.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from alinhamento import carregar_gold, casar      # noqa: E402
from avaliar_oficial import (carregar_metrica, solution_df,  # noqa: E402
                             submission_df)

EVIDENCIAS = ["vaga", "lei", "sumula", "tema", "cnj_uf", "cnj", "num_uf",
              "num", "ruidoso", "duplicata"]
NUMERICAS = ["n_digitos", "tem_uf", "n_variantes", "ruidoso", "cnj",
             "n_tokens_classe", "largura", "n_cand"]


def vetor(cit):
    ev = cit.get("evidencia", "").replace("ner:", "")
    s = cit.get("sinais") or {}
    v = [1.0 if ev == e else 0.0 for e in EVIDENCIAS]
    v += [float(s.get(k, 0) or 0) for k in NUMERICAS]
    v += [1.0 if s.get("camada") == "primaria" else 0.0,
          1.0 if s.get("camada") == "secundaria" else 0.0,
          1.0 if cit.get("evidencia", "").startswith("ner:") else 0.0,
          float(len(cit.get("trecho", ""))) / 50.0]
    return v


def montar(pasta, gold_csv):
    """Monta ``X, y`` sobre os pares casados, que são os considerados pelo Brier."""
    gold, _ = carregar_gold(gold_csv)
    X, y = [], []
    for doc_id, golds in gold.items():
        arq = Path(pasta) / f"{doc_id}.json"
        if not arq.exists():
            continue
        cits = json.loads(arq.read_text(encoding="utf-8"))["citacoes"]
        preds = [{"span": (c["inicio"], c["fim"]), "classe": c["classificacao"],
                  "id": str((c.get("resolucao") or {}).get("id_canonico") or ""),
                  "cit": c} for c in cits]
        pares, _, _ = casar(preds, golds)
        for p, g in pares:
            ok = p["classe"] == g["classe"] and (
                g["classe"] != "real" or p["id"] == g["id"])
            X.append(vetor(p["cit"]))
            y.append(int(ok))
    return np.array(X, dtype=float), np.array(y)


def brier(pasta, gold_csv, probs=None):
    """Brier sobre os pares casados; com ``probs``, usa-as no lugar da confiança."""
    gold, _ = carregar_gold(gold_csv)
    termos, k = [], 0
    for doc_id, golds in gold.items():
        arq = Path(pasta) / f"{doc_id}.json"
        if not arq.exists():
            continue
        cits = json.loads(arq.read_text(encoding="utf-8"))["citacoes"]
        preds = [{"span": (c["inicio"], c["fim"]), "classe": c["classificacao"],
                  "id": str((c.get("resolucao") or {}).get("id_canonico") or ""),
                  "cit": c} for c in cits]
        pares, _, _ = casar(preds, golds)
        for p, g in pares:
            ok = p["classe"] == g["classe"] and (
                g["classe"] != "real" or p["id"] == g["id"])
            c = p["cit"].get("confianca") if probs is None else probs[k]
            k += 1
            if c is not None:
                termos.append((float(c) - ok) ** 2)
    return float(np.mean(termos)) if termos else None


def aplicar(pasta_in, pasta_out, modelo):
    """Reescreve os JSONs substituindo ``confianca`` pela probabilidade do modelo."""
    coef = np.array(modelo["coef"])
    inter = modelo["intercepto"]
    Path(pasta_out).mkdir(parents=True, exist_ok=True)
    for arq in sorted(Path(pasta_in).glob("*.json")):
        doc = json.loads(arq.read_text(encoding="utf-8"))
        for c in doc["citacoes"]:
            z = float(np.dot(coef, vetor(c)) + inter)
            p = 1.0 / (1.0 + np.exp(-z))
            c["confianca"] = round(min(0.999, max(0.001, p)), 4)
        (Path(pasta_out) / arq.name).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", nargs=2, required=True,
                    metavar=("PRED", "GOLD"))
    ap.add_argument("--teste", nargs=2, required=True, metavar=("PRED", "GOLD"))
    ap.add_argument("--metrica", required=True)
    ap.add_argument("--saida", default="calibracao.json")
    a = ap.parse_args()

    Xtr, ytr = montar(a.treino[0], a.treino[1])
    print(f"treino: {len(ytr)} pares casados, {ytr.mean():.3f} de acerto")
    if ytr.min() == ytr.max():
        print("AVISO: o conjunto de treino tem uma única classe; a regressão não\n"
              "       tem o que aprender. Use um corpus sintético maior ou mais difícil.")
        return

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, ytr)
    modelo = {"coef": lr.coef_[0].tolist(), "intercepto": float(lr.intercept_[0]),
              "evidencias": EVIDENCIAS, "numericas": NUMERICAS}
    Path(a.saida).write_text(json.dumps(modelo, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    km = carregar_metrica(a.metrica)

    def score(pasta, gold):
        sol = solution_df(gold)
        sub = submission_df(pasta, sol["documento_id"].tolist())
        return km.avaliar(sol, sub, row_id="documento_id")["score_final"]

    pred_te, gold_te = a.teste
    b0, s0 = brier(pred_te, gold_te), score(pred_te, gold_te)
    saida_cal = "/tmp/_calibrado"
    aplicar(pred_te, saida_cal, modelo)
    b1, s1 = brier(saida_cal, gold_te), score(saida_cal, gold_te)

    print(f"\nTESTE (conjunto não visto no treino)")
    print(f"  tabela fixa CONF : Brier={b0:.5f}  bônus={0.10*(1-b0):.5f}  "
          f"score={s0:.4f}")
    print(f"  regressão        : Brier={b1:.5f}  bônus={0.10*(1-b1):.5f}  "
          f"score={s1:.4f}")
    print(f"  delta de score   : {s1 - s0:+.4f}")
    print(f"\nmodelo salvo em {a.saida}; "
          f"aplique com calibrar_ml.aplicar(pred/, pred_cal/, modelo)")


if __name__ == "__main__":
    main()
