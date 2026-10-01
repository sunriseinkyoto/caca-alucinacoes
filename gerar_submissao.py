# -*- coding: utf-8 -*-
"""
Gera as submissões a partir de uma pasta de documentos .txt.

Executa, em sequência:

  1. construção do índice a partir da base canônica (se disponível), com
     verificação de que coincide com o índice versionado no repositório;
  2. extração com a política de anotação vigente -> submission.csv;
  3. extração com a política anterior a 04/09/2026 (citações difusas)
     -> submission_difusas.csv;
  4. validação do contrato de saída de cada variante;
  5. conversão com o conversor oficial json_to_submission.py, sem modificação;
  6. avaliação com a métrica oficial, quando o gabarito é informado.

Com ``--modelo-ner <pasta>``, gera também submission_bert.csv: a variante
principal acrescida dos spans do rotulador BERTimbau, pelo critério de fusão
estrito de ml/fundir.py. Requer torch e transformers (requirements-ml.txt) e
os pesos treinados, anexados ao Release v1.0 do repositório (ver ml/README.md).

A variante principal é submission.csv. A variante difusas existe porque a
política de anotação mudou durante o desafio (ver README, "Política de
anotação"); a parte pública do leaderboard do conjunto de avaliação indica
qual das duas o gabarito segue.

Uso:
    python gerar_submissao.py                          # conjunto em dados/txt
    python gerar_submissao.py --txt <pasta_txt> --saida <pasta>
    python gerar_submissao.py --gabarito dados/goldenset.csv   # com avaliação
    python gerar_submissao.py --modelo-ner <pasta_do_modelo>   # + variante BERT
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import subprocess
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

import indexar  # noqa: E402
import rodar  # noqa: E402
import validar  # noqa: E402

DADOS = RAIZ / "dados"


def construir_indice(base: Path, saida: Path) -> Path:
    versionado = RAIZ / "indice.json"
    if not base.exists():
        raise SystemExit(f"base canônica não encontrada: {base}")
    destino = saida / "indice.json"
    indexar.main(str(base), str(destino))
    if versionado.exists():
        igual = filecmp.cmp(destino, versionado, shallow=False)
        print("índice idêntico ao versionado" if igual else
              "base diferente da amostra de desenvolvimento: será usado o índice "
              "construído a partir da base informada")
    return destino


def gerar_variante(nome: str, txt: Path, indice: Path, saida: Path,
                   conversor: Path, difusas: bool) -> Path:
    print(f"\n=== variante: {nome}")
    pasta_json = saida / f"json_{nome}"
    rodar.main(str(txt), str(indice), str(pasta_json),
               str(RAIZ / "templates.json"), "maxima", difusas)
    try:
        validar.main(str(pasta_json), str(txt))
    except SystemExit as e:
        if e.code:
            raise SystemExit(f"contrato inválido na variante {nome}; submissão não gerada")
    destino = saida / ("submission.csv" if nome == "principal"
                       else f"submission_{nome}.csv")
    subprocess.run([sys.executable, str(conversor), str(pasta_json), str(destino)],
                   check=True)
    return destino


def gerar_variante_bert(txt: Path, indice: Path, saida: Path, conversor: Path,
                        modelo_ner: Path) -> Path:
    """Variante principal + spans do rotulador, pelo critério de fusão estrito."""
    print("\n=== variante: bert")
    try:
        sys.path.insert(0, str(RAIZ / "ml"))
        from inferir_ner import carregar_modelo, rotular
        from fundir import fundir_documento
    except ModuleNotFoundError as e:
        raise SystemExit(f"a variante bert requer {e.name} "
                         "(pip install -r requirements-ml.txt)")
    import json
    import shutil
    from extrair import Resolvedor
    from texto import carregar

    origem, pasta_json = saida / "json_principal", saida / "json_bert"
    shutil.rmtree(pasta_json, ignore_errors=True)
    shutil.copytree(origem, pasta_json)
    res = Resolvedor(json.loads(indice.read_text(encoding="utf-8")))
    tok, modelo, disp = carregar_modelo(str(modelo_ner))
    acrescentados = 0
    for arq in sorted(txt.glob("*.txt")):
        fonte = carregar(arq)
        spans = rotular(fonte.proc, tok, modelo, disp)
        for s in spans:
            s["inicio"], s["fim"] = fonte.mapear(s["inicio"], s["fim"])
        destino = pasta_json / f"{arq.stem}.json"
        doc = json.loads(destino.read_text(encoding="utf-8"))
        acrescentados += fundir_documento(doc, spans, fonte.bruto, res, 0.90, 12)
        destino.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"spans acrescentados pelo rotulador: {acrescentados}")
    try:
        validar.main(str(pasta_json), str(txt))
    except SystemExit as e:
        if e.code:
            raise SystemExit("contrato inválido na variante bert; submissão não gerada")
    csv_ = saida / "submission_bert.csv"
    subprocess.run([sys.executable, str(conversor), str(pasta_json), str(csv_)], check=True)
    return csv_


def resumo(csv_path: Path) -> Counter:
    contagem: Counter = Counter()
    with csv_path.open(encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha["citacoes"] != "-":
                for cit in linha["citacoes"].split("|"):
                    contagem[cit.split(",")[2]] += 1
    return contagem


def avaliar(pasta_json: Path, gabarito: Path, metrica: Path) -> float:
    import avaliar_oficial
    km = avaliar_oficial.carregar_metrica(metrica)
    sol = avaliar_oficial.solution_df(gabarito)
    sub = avaliar_oficial.submission_df(pasta_json, sol["documento_id"].tolist())
    return km.avaliar(sol, sub, row_id="documento_id")["score_final"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--txt", type=Path, default=DADOS / "txt")
    ap.add_argument("--base", type=Path, default=DADOS / "desafio1_bracis.db")
    ap.add_argument("--conversor", type=Path,
                    default=RAIZ / "oficial" / "json_to_submission.py")
    ap.add_argument("--saida", type=Path, default=RAIZ / "saida")
    ap.add_argument("--gabarito", type=Path, help="goldenset.csv, para avaliar")
    ap.add_argument("--metrica", type=Path, default=DADOS / "kaggle_metric.py")
    ap.add_argument("--modelo-ner", type=Path,
                    help="pasta do BERTimbau treinado; gera também submission_bert.csv")
    a = ap.parse_args()

    if not a.txt.is_dir():
        raise SystemExit(f"pasta de documentos não encontrada: {a.txt}")
    if not a.conversor.exists():
        raise SystemExit(f"conversor oficial não encontrado: {a.conversor} "
                         "(o repositório inclui a cópia oficial em oficial/)")
    a.saida.mkdir(parents=True, exist_ok=True)

    indice = construir_indice(a.base, a.saida)
    variantes = {
        "principal": gerar_variante("principal", a.txt, indice, a.saida, a.conversor, False),
        "difusas": gerar_variante("difusas", a.txt, indice, a.saida, a.conversor, True),
    }

    if a.modelo_ner:
        variantes["bert"] = gerar_variante_bert(a.txt, indice, a.saida,
                                                a.conversor, a.modelo_ner)

    print("\n=== resumo")
    for nome, caminho in variantes.items():
        c = resumo(caminho)
        print(f"{caminho.name:26s} real={c['real']:4d}  inventada={c['inventada']:4d}  "
              f"incompleta={c['incompleta']:4d}  total={sum(c.values()):4d}")
    if a.gabarito:
        if not a.metrica.exists():
            raise SystemExit(f"métrica oficial não encontrada: {a.metrica}")
        for nome in variantes:
            s = avaliar(a.saida / f"json_{nome}", a.gabarito, a.metrica)
            print(f"score oficial ({nome}) = {s:.4f}")
    print(f"\nsubmissão principal: {variantes['principal']}")


if __name__ == "__main__":
    main()
