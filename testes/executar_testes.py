# -*- coding: utf-8 -*-
"""
Executa a bateria completa de testes.

  1. amostra de desenvolvimento: score oficial e byte-identidade do
     submission.csv versionado;
  2. cobertura da base canônica (todos os acórdãos, em três escritas);
  3. cobertura de dispositivos legais e súmulas, com controles de segurança;
  4. robustez à codificação (CRLF, BOM, NFD);
  5. corpora sintéticos com três sementes (omitido com --rapido).

Espera o material da organização em dados/ (ver dados/LEIA-ME.md).

Uso:
    python testes/executar_testes.py [--rapido]
"""
from __future__ import annotations

import filecmp
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "testes"))


def exigir(*caminhos: Path) -> None:
    faltam = [str(c) for c in caminhos if not c.exists()]
    if faltam:
        raise SystemExit("material da organização ausente: " + ", ".join(faltam)
                         + " (ver dados/LEIA-ME.md)")


def teste_desenvolvimento() -> bool:
    import avaliar_oficial
    import rodar
    with tempfile.TemporaryDirectory() as tmp:
        pred, csv_ = Path(tmp) / "pred", Path(tmp) / "submission.csv"
        rodar.main(str(DADOS / "txt"), str(RAIZ / "indice.json"), str(pred),
                   str(RAIZ / "templates.json"))
        subprocess.run([sys.executable, str(RAIZ / "oficial" / "json_to_submission.py"),
                        str(pred), str(csv_)], check=True, stdout=subprocess.DEVNULL)
        km = avaliar_oficial.carregar_metrica(DADOS / "kaggle_metric.py")
        sol = avaliar_oficial.solution_df(DADOS / "goldenset.csv")
        sub = avaliar_oficial.submission_df(pred, sol["documento_id"].tolist())
        score = km.avaliar(sol, sub, row_id="documento_id")["score_final"]
        identico = filecmp.cmp(csv_, RAIZ / "submission.csv", shallow=False)
    print(f"score oficial = {score:.4f}; submission.csv "
          f"{'idêntico ao versionado' if identico else 'DIFERENTE do versionado'}")
    return abs(score - 1.1) < 1e-9 and identico


def teste_indice() -> bool:
    import indexar
    with tempfile.TemporaryDirectory() as tmp:
        destino = Path(tmp) / "indice.json"
        indexar.main(str(DADOS / "desafio1_bracis.db"), str(destino))
        igual = filecmp.cmp(destino, RAIZ / "indice.json", shallow=False)
    print(f"índice reconstruído {'idêntico' if igual else 'DIFERENTE'} ao versionado")
    return igual


def main() -> None:
    rapido = "--rapido" in sys.argv
    exigir(DADOS / "txt", DADOS / "desafio1_bracis.db", DADOS / "goldenset.csv",
           DADOS / "kaggle_metric.py", RAIZ / "oficial" / "json_to_submission.py")
    import teste_codificacao
    import teste_cobertura_base
    import teste_cobertura_normas
    import teste_sintetico

    base, indice = str(DADOS / "desafio1_bracis.db"), str(RAIZ / "indice.json")
    etapas = [
        ("índice reprodutível", teste_indice),
        ("amostra de desenvolvimento", teste_desenvolvimento),
        ("cobertura da base", lambda: teste_cobertura_base.main(base, indice)),
        ("cobertura de normas", lambda: teste_cobertura_normas.main(indice)),
        ("codificação", lambda: teste_codificacao.main(
            str(DADOS / "txt"), str(DADOS / "goldenset.csv"), indice,
            str(DADOS / "kaggle_metric.py"))),
    ]
    if not rapido:
        etapas.append(("corpora sintéticos", lambda: teste_sintetico.main(
            base, indice, str(DADOS / "kaggle_metric.py"))))

    resultados = []
    for nome, fn in etapas:
        print(f"\n### {nome}")
        t0 = time.time()
        ok = bool(fn())
        resultados.append((nome, ok, time.time() - t0))

    print("\n" + "=" * 52)
    for nome, ok, dt in resultados:
        print(f"{'OK   ' if ok else 'FALHA'}  {nome:30s} {dt:6.1f} s")
    sys.exit(0 if all(ok for _, ok, _ in resultados) else 1)


if __name__ == "__main__":
    main()
