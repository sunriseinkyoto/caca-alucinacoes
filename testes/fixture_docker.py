# -*- coding: utf-8 -*-
"""
Dados mínimos, no formato do desafio, para testar a imagem Docker do zero.

Cria uma base com quatro registros inventados (acórdão, súmula, dispositivo de
um diploma que não aparece na amostra de desenvolvimento) e um documento que
cita cada um deles, mais duas citações inexistentes na base. Não usa nenhum
material da organização, por isso pode rodar em integração contínua.

Uso:
    python testes/fixture_docker.py criar <pasta>          # <pasta>/base.db e <pasta>/txt/
    python testes/fixture_docker.py conferir <submission.csv> [--exigir-bert]
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE documentos (
    documento_id TEXT PRIMARY KEY, id INTEGER NOT NULL UNIQUE, tribunal TEXT,
    ano INTEGER, relator TEXT,
    natureza TEXT NOT NULL CHECK (natureza IN ('acordao', 'sumula', 'dispositivo')),
    tipo TEXT NOT NULL CHECK (tipo IN ('jurisprudencia', 'lei')),
    texto TEXT NOT NULL, texto_len INTEGER NOT NULL)
"""
REGISTROS = [
    ("doc_9001", 7000000001, "STJ", 2021, "Ana Exemplo", "acordao", "jurisprudencia",
     "RECURSO ESPECIAL Nº 1.234.567 - SP (2020/0123456-7) RELATOR : MINISTRA ANA "
     "EXEMPLO RECORRENTE : FULANO DE TAL RECORRIDO : MINISTÉRIO PÚBLICO EMENTA "
     "PROCESSUAL PENAL. RECURSO ESPECIAL. TEXTO DE TESTE."),
    ("7000000002", 7000000002, "STJ", None, None, "sumula", "jurisprudencia",
     "Súmula n. 7 do STJ\nDIREITO PROCESSUAL CIVIL\nTexto de teste da súmula."),
    ("7000000003", 7000000003, None, None, None, "dispositivo", "lei",
     "Artigo 157 do Decreto-Lei nº 2.848, de 7 de dezembro de 1940\n"
     "Art. 157. Texto de teste do dispositivo."),
]
DOCUMENTO = (
    "PROCESSO DE TESTE\n\n"
    "Conforme decidido no REsp nº 1.234.567/SP, a tese foi acolhida. "
    "Incide, ainda, o art. 157 do Código Penal, e a Súmula 7 do STJ impede o "
    "reexame de provas. Em sentido contrário, o AREsp nº 9.876.543/RJ, que não "
    "se aplica ao caso, e o art. 999 do Código Penal.\n"
)
ESPERADO = [("real", "7000000001"), ("real", "7000000003"), ("real", "7000000002"),
            ("inventada", "-"), ("inventada", "-")]


def criar(pasta: Path) -> None:
    (pasta / "txt").mkdir(parents=True, exist_ok=True)
    base = pasta / "base.db"
    base.unlink(missing_ok=True)
    con = sqlite3.connect(base)
    con.execute(SCHEMA)
    con.executemany("INSERT INTO documentos VALUES (?,?,?,?,?,?,?,?,?)",
                    [r + (len(r[-1]),) for r in REGISTROS])
    con.commit()
    con.close()
    (pasta / "txt" / "teste_001.txt").write_text(DOCUMENTO, encoding="utf-8", newline="\n")
    print(f"dados mínimos criados em {pasta}")


def conferir(csv_path: Path, exigir_bert: bool) -> int:
    with csv_path.open(encoding="utf-8") as f:
        linhas = {r["documento_id"]: r["citacoes"] for r in csv.DictReader(f)}
    obtido = [(c.split(",")[2], c.split(",")[3])
              for c in linhas.get("teste_001", "-").split("|") if c != "-"]
    ok = obtido == ESPERADO
    print("obtido :", obtido)
    print("esperado:", ESPERADO)
    if exigir_bert and not (csv_path.parent / "variantes" / "submission_bert.csv").exists():
        print("ERRO: a variante com o rotulador BERTimbau não foi gerada (pesos ausentes?)")
        ok = False
    print("OK" if ok else "FALHA")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "criar":
        criar(Path(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "conferir":
        sys.exit(conferir(Path(sys.argv[2]), "--exigir-bert" in sys.argv))
    else:
        raise SystemExit(__doc__)
