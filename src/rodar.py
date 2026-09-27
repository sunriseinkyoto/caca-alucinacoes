# -*- coding: utf-8 -*-
"""
Executa o pipeline sobre uma pasta de documentos e grava um JSON por documento,
no contrato de saída do desafio (schema 1.2).

Todo arquivo ``.txt`` da pasta de entrada gera um JSON, inclusive os que não
contêm citação: o conversor oficial precisa de uma linha por documento e
escreve ``-`` quando a lista de citações é vazia.

Os offsets são sempre referentes ao texto exatamente como distribuído
(ver ``src/texto.py``).

Uso:
    python src/rodar.py <pasta_txt> <indice.json> <pasta_saida>
                        [--templates templates.json]
                        [--confianca maxima|calibrada]
                        [--difusas]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extrair import (Resolvedor, extrair, carregar_neutras,  # noqa: E402
                     definir_modo_confianca, definir_modo_difusas)
from texto import carregar  # noqa: E402

SCHEMA = "1.2"


def processar_documento(caminho: Path, res: Resolvedor) -> dict:
    fonte = carregar(caminho)
    citacoes = []
    for i, c in enumerate(extrair(fonte.proc, res), 1):
        registro = c.to_json()
        ini, fim = fonte.mapear(registro["inicio"], registro["fim"])
        registro.update(inicio=ini, fim=fim, trecho=fonte.bruto[ini:fim])
        citacoes.append({"id": f"c{i}", **registro})
    return {"schema_version": SCHEMA, "documento_id": caminho.stem,
            "citacoes": citacoes}


def main(pasta_txt: str, indice_path: str, saida: str,
         templates: str | None = "templates.json",
         modo_confianca: str = "maxima", difusas: bool = False) -> int:
    definir_modo_confianca(modo_confianca)
    definir_modo_difusas(difusas)
    if difusas:
        print("modo difusas: ativo (política de anotação anterior a 04/09/2026)")

    indice = json.loads(Path(indice_path).read_text(encoding="utf-8"))
    if templates and Path(templates).exists():
        carregar_neutras(templates)
    res = Resolvedor(indice)

    out = Path(saida)
    out.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(Path(pasta_txt).glob("*.txt"))
    if not arquivos:
        raise SystemExit(f"nenhum .txt encontrado em {pasta_txt}")

    total = 0
    for arq in arquivos:
        doc = processar_documento(arq, res)
        total += len(doc["citacoes"])
        (out / f"{arq.stem}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(arquivos)} documentos, {total} citações -> {out}")
    return len(arquivos)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("pasta_txt")
    ap.add_argument("indice")
    ap.add_argument("saida")
    ap.add_argument("--templates", default="templates.json")
    ap.add_argument("--confianca", default="maxima",
                    choices=["maxima", "calibrada"])
    ap.add_argument("--difusas", action="store_true",
                    help="inclui as citações difusas da política de anotação "
                         "anterior a 04/09/2026")
    a = ap.parse_args()
    main(a.pasta_txt, a.indice, a.saida, a.templates, a.confianca, a.difusas)


if __name__ == "__main__":
    _cli()
