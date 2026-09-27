# -*- coding: utf-8 -*-
"""
Experimento: a fusão com o rotulador de spans eleva o score?

A pergunta não é se o rotulador é bom, e sim se acrescentar seus spans eleva
o score, dado o que o sistema de regras já acerta. Pela matriz da métrica, um
span espúrio é um falso positivo direto, e um span recuperado só se torna
verdadeiro positivo se o resolvedor também acertar a classe. Existe, portanto,
um ponto de equilíbrio, que depende de quanto as regras deixam de detectar
(o "buraco") e de quanto ruído o rotulador introduz.

O script degrada artificialmente a predição das regras — simulando templates
desconhecidos — e mede o score com e sem fusão, em duas variantes:

  * simulada (sem --ner): um rotulador artificial com recall e taxa de ruído
    configuráveis; varre o plano e localiza o ponto de equilíbrio;
  * real (com --ner spans.json): usa os spans produzidos pelo BERTimbau
    treinado; o ruído é o do próprio modelo.

Uso:
    python ml/experimento_fusao.py <pred_regras/> <goldenset.csv> <pasta_txt> \
           <indice.json> <kaggle_metric.py> [--ner spans.json] [--fusao-antiga]
"""
import argparse
import copy
import tempfile
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from avaliar_oficial import carregar_metrica, solution_df, submission_df  # noqa
from extrair import Resolvedor  # noqa: E402
from fundir import fundir_documento  # noqa: E402
from ner_simulado import main as _  # noqa: F401,E402


def score(km, pasta, gold_csv):
    sol = solution_df(gold_csv)
    sub = submission_df(pasta, sol["documento_id"].tolist())
    return km.avaliar(sol, sub, row_id="documento_id")["score_final"]


def main(pred_dir, gold_csv, txt_dir, indice_path, metric_py, ner_json=None,
         estrita=True):
    km = carregar_metrica(metric_py)
    res = Resolvedor(json.loads(Path(indice_path).read_text(encoding="utf-8")))
    docs = {a.stem: json.loads(a.read_text(encoding="utf-8"))
            for a in sorted(Path(pred_dir).glob("*.json"))}
    textos = {d: Path(txt_dir, f"{d}.txt").read_text(encoding="utf-8")
              for d in docs}

    import csv as _csv
    from collections import defaultdict
    gold = defaultdict(list)
    for r in _csv.DictReader(open(gold_csv, encoding="utf-8-sig")):
        gold[r["documento_id"]].append((int(r["inicio"]), int(r["fim"])))

    tmp = Path(tempfile.mkdtemp(prefix="fusao_"))

    def salvar(ds):
        for d, doc in ds.items():
            (tmp / f"{d}.json").write_text(json.dumps(doc, ensure_ascii=False),
                                           encoding="utf-8")
        return score(km, tmp, gold_csv)

    def degradar(ds, frac, rng):
        """Simula as regras deixando de detectar a fração ``frac`` das citações."""
        novo = copy.deepcopy(ds)
        for doc in novo.values():
            doc["citacoes"] = [c for c in doc["citacoes"] if rng.random() >= frac]
        return novo

    def ner_de(ds_degradado, recall, ruido, rng):
        """Rotulador simulado: lê o gabarito com o recall e o ruído informados."""
        saida = {}
        for d, texto in textos.items():
            spans = []
            for ini, fim in gold[d]:
                if rng.random() > recall:
                    continue
                di, df = rng.randint(-3, 2), rng.randint(-2, 3)
                i2, f2 = max(0, ini + di), min(len(texto), fim + df)
                spans.append({"inicio": i2, "fim": f2, "rotulo": "JUR",
                              "score": round(rng.uniform(0.88, 0.999), 4)})
            for _ in range(int(len(gold[d]) * ruido + rng.random())):
                i = rng.randint(0, max(0, len(texto) - 40))
                spans.append({"inicio": i, "fim": i + rng.randint(8, 35),
                              "rotulo": "JUR",
                              "score": round(rng.uniform(0.85, 0.97), 4)})
            saida[d] = spans
        return saida

    if ner_json:
        # ------------------------------------------------- rotulador treinado
        ner_real = json.loads(Path(ner_json).read_text(encoding="utf-8"))
        # precisão e recall de span do modelo neste conjunto, para referência
        tp = fp = 0
        for d, texto in textos.items():
            for s in ner_real.get(d, []):
                if s.get("score", 1.0) < 0.90:   # o mesmo limiar usado na fusão
                    continue
                casou = any(
                    max(0, min(s["fim"], f) - max(s["inicio"], i)) /
                    max(1e-9, (s["fim"] - s["inicio"]) + (f - i) -
                        max(0, min(s["fim"], f) - max(s["inicio"], i))) >= 0.5
                    for i, f in gold[d])
                tp += casou
                fp += not casou
        n_ouro = sum(len(v) for v in gold.values())
        bruto = sum(len(v) for v in ner_real.values())
        print(f"rotulador treinado ({ner_json}): {bruto} spans, {tp+fp} acima do "
              f"corte 0,90 -> P={tp/max(tp+fp,1):.4f} R={tp/max(n_ouro,1):.4f} "
              f"sobre {n_ouro} citações do gabarito")
        print(f"\n{'buraco':>7s} {'só regras':>10s} {'com modelo':>10s} {'delta':>8s}")
        print("-" * 40)
        for buraco in (0.0, 0.05, 0.10, 0.20, 0.50, 1.0):
            rng = random.Random(11)
            deg = degradar(docs, buraco, rng)
            base = salvar(deg)
            fund = copy.deepcopy(deg)
            for d, doc in fund.items():
                fundir_documento(doc, ner_real.get(d, []), textos[d], res,
                                 0.90, 12, estrita=estrita)
            com = salvar(fund)
            marca = "  <-- piora" if com < base - 1e-9 else ""
            print(f"{buraco:7.0%} {base:10.4f} {com:10.4f} "
                  f"{com - base:+8.4f}{marca}")
        return

    print(f"{'buraco':>7s} {'ruído':>10s} {'só regras':>10s} "
          f"{'com modelo':>10s} {'delta':>8s}")
    print("-" * 50)
    for buraco in (0.0, 0.05, 0.10, 0.20, 0.50, 1.0):
        for ruido in (0.0, 0.02, 0.05, 0.10):
            rng = random.Random(11)
            deg = degradar(docs, buraco, rng)
            base = salvar(deg)
            ner = ner_de(deg, 0.90, ruido, rng)
            fund = copy.deepcopy(deg)
            for d, doc in fund.items():
                fundir_documento(doc, ner.get(d, []), textos[d], res, 0.90, 12,
                                 estrita=estrita)
            com = salvar(fund)
            marca = "  <-- piora" if com < base - 1e-9 else ""
            print(f"{buraco:7.0%} {ruido:10.0%} {base:10.4f} {com:10.4f} "
                  f"{com - base:+8.4f}{marca}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for nome in ("pred", "gabarito", "pasta_txt", "indice", "metrica"):
        ap.add_argument(nome)
    ap.add_argument("--ner", help="spans do rotulador treinado (ml/inferir_ner.py)")
    ap.add_argument("--fusao-antiga", action="store_true",
                    help="critério de fusão original (IoU >= 0,5)")
    a = ap.parse_args()
    main(a.pred, a.gabarito, a.pasta_txt, a.indice, a.metrica,
         ner_json=a.ner, estrita=not a.fusao_antiga)
