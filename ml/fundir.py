# -*- coding: utf-8 -*-
"""
Fusão entre o sistema de regras e um rotulador de spans.

As regras prevalecem. O rotulador só acrescenta spans em trechos onde as
regras não encontraram nada, e todo span acrescentado passa pelo mesmo
resolvedor determinístico (``classificar_trecho``): o modelo propõe onde, o
índice decide o quê.

Dois critérios de fusão estão implementados:

  * estrito (padrão). Um span do rotulador é descartado se (1) sobrepuser, em
    qualquer medida, uma citação das regras, ou (2) não contiver nada
    verificável — nem identificador (número, lei, súmula) nem forma de citação
    vaga. O critério (1) impede que um span com borda diferente duplique uma
    citação das regras: se elas já encontraram algo ali, a decisão é delas. O critério (2) elimina trechos que
    o resolvedor só classificaria como ``incompleta`` por falta de alternativa.
  * original (``--fusao-antiga``). Descarta apenas sobreposições com IoU >= 0,5
    ou contenção. Mantido para comparação: nos 26 documentos, custa −0,0070
    com as regras completas; ver ml/README.md.

``--relatorio`` lista os spans que seriam acrescentados, sem gravar saída.

Uso:
    python ml/fundir.py <pred_regras/> <ner_spans.json> <pasta_txt> \
           <indice.json> <saida/> [--minimo 0.90] [--margem 12] [--relatorio]
           [--fusao-antiga]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import extrair  # noqa: E402
from extrair import Resolvedor, achar_vagas, classificar_trecho  # noqa: E402

# Fator aplicado à confiança das citações acrescentadas pelo rotulador no modo
# de confiança "calibrada". No modo "maxima" (padrão), elas recebem a mesma
# confiança do resolvedor, como as demais: nos corpora de validação, as
# citações acrescentadas pela fusão estrita tiveram acerto de 100%, e uma
# confiança menor só reduziria o bônus de calibração.
FATOR_CONF_NER = 0.85


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    uni = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / uni if uni else 0.0


def _fator_confianca():
    return FATOR_CONF_NER if extrair.MODO_CONFIANCA == "calibrada" else 1.0


def _sobrepoe(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def _e_vaga(trecho):
    """O trecho é, em sua maior parte, uma citação vaga reconhecida?"""
    n = len(trecho.strip())
    return any((b - a) >= 0.6 * n for a, b, _ in achar_vagas(trecho))


def fundir_documento(doc, spans_ner, texto, res, minimo, margem, estrita=True):
    regras = [(c["inicio"], c["fim"]) for c in doc["citacoes"]]
    existentes = list(regras)
    novos = []
    for s in spans_ner:
        if s.get("score", 1.0) < minimo:
            continue
        cand = (s["inicio"], s["fim"])
        if estrita:
            # qualquer sobreposição com as regras: as regras prevalecem
            if any(_sobrepoe(cand, e) for e in existentes):
                continue
        else:
            if any(iou(cand, e) >= 0.5 for e in existentes):
                continue
            if any(e[0] <= cand[0] and cand[1] <= e[1] for e in existentes):
                continue
        a = max(0, cand[0] - margem)
        b = min(len(texto), cand[1] + margem)
        tipo, classe, cid, conf, evid = classificar_trecho(texto[a:b], res)
        # o próprio span precisa conter algo verificável (identificador ou forma
        # de citação vaga); a janela ampliada não pode "emprestar" um número vizinho
        if estrita:
            proprio = texto[cand[0]:cand[1]]
            if classificar_trecho(proprio, res)[1] == "incompleta" and not _e_vaga(proprio):
                continue
        novos.append({"inicio": cand[0], "fim": cand[1],
                      "trecho": texto[cand[0]:cand[1]],
                      "tipo": tipo, "classificacao": classe,
                      "resolucao": ({"fonte": "jusbrasil", "id_canonico": cid}
                                    if cid else None),
                      "confianca": round(conf * _fator_confianca(), 4),
                      "evidencia": f"ner:{evid}"})
        existentes.append(cand)

    todas = sorted(doc["citacoes"] + novos, key=lambda c: c["inicio"])
    for i, c in enumerate(todas, 1):
        c["id"] = f"c{i}"
    doc["citacoes"] = todas
    return len(novos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_regex")
    ap.add_argument("ner_spans")
    ap.add_argument("pasta_txt")
    ap.add_argument("indice")
    ap.add_argument("saida")
    ap.add_argument("--minimo", type=float, default=0.90)
    ap.add_argument("--margem", type=int, default=12)
    ap.add_argument("--fusao-antiga", action="store_true",
                    help="critério original (IoU >= 0,5), para reproduzir a medição inicial")
    ap.add_argument("--relatorio", action="store_true",
                    help="não grava saída; lista os spans que o rotulador "
                         "acrescentaria. Permite avaliar, num conjunto sem "
                         "gabarito, se as regras deixaram de detectar citações: "
                         "spans legítimos indicam lacuna das regras; spans "
                         "espúrios indicam que a fusão deve permanecer desligada.")
    a = ap.parse_args()

    res = Resolvedor(json.loads(Path(a.indice).read_text(encoding="utf-8")))
    ner = json.loads(Path(a.ner_spans).read_text(encoding="utf-8"))
    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)

    total = antes = 0
    for arq in sorted(Path(a.pred_regex).glob("*.json")):
        doc = json.loads(arq.read_text(encoding="utf-8"))
        texto = Path(a.pasta_txt, f"{arq.stem}.txt").read_text(encoding="utf-8")
        antes += len(doc["citacoes"])
        n = fundir_documento(doc, ner.get(arq.stem, []), texto, res,
                             a.minimo, a.margem, estrita=not a.fusao_antiga)
        total += n
        if a.relatorio:
            for c in doc["citacoes"]:
                if str(c.get("evidencia", "")).startswith("ner:"):
                    print(f"  {arq.stem} [{c['inicio']}:{c['fim']}] "
                          f"{c['classificacao']:11s} {c['trecho']!r}")
        else:
            (out / arq.name).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    pct = 100 * total / max(antes, 1)
    print(f"\n{total} spans acrescentados pelo rotulador sobre {antes} das regras ({pct:.1f}%)")
    if a.relatorio:
        print("Os spans listados estão em trechos onde as regras não encontraram\n"
              "nada e contêm identificador ou forma de citação vaga. Citações\n"
              "legítimas indicam lacuna das regras, que a variante com BERT cobre;\n"
              "spans espúrios indicam que a variante principal deve prevalecer.")
    else:
        print(f"-> {out}")


if __name__ == "__main__":
    main()
