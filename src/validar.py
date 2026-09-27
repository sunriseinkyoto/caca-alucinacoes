# -*- coding: utf-8 -*-
"""
Validação do contrato de saída, a executar antes de gerar a submissão.

Verifica as condições que a métrica oficial rejeita ou que invalidam a
submissão:

  * um JSON por documento de entrada, com o mesmo nome-base, e nenhum JSON
    sem documento correspondente;
  * ``schema_version``, ``documento_id`` e ``citacoes`` presentes;
  * ``0 <= inicio < fim <= len(texto)`` e ``trecho == texto[inicio:fim]``,
    sobre o texto exatamente como distribuído;
  * ``classificacao`` em {real, inventada, incompleta} e ``tipo`` em
    {lei, jurisprudencia};
  * ``classificacao = real`` exige ``id_canonico`` numérico;
  * ``confianca`` em [0, 1] ou ausente;
  * nenhum par de citações do mesmo documento com IoU >= 0,5.

Uso:
    python src/validar.py <pasta_jsons> <pasta_txt>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from texto import ler_bruto  # noqa: E402

CLASSES = {"real", "inventada", "incompleta"}
TIPOS = {"lei", "jurisprudencia"}


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    uniao = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / uniao if uniao else 0.0


def main(pasta_json, pasta_txt):
    erros, docs, n_cit = [], 0, 0
    for arq_txt in sorted(Path(pasta_txt).glob("*.txt")):
        docs += 1
        doc_id = arq_txt.stem
        texto = ler_bruto(arq_txt)   # exatamente como distribuído
        arq = Path(pasta_json) / f"{doc_id}.json"
        if not arq.exists():
            erros.append(f"{doc_id}: JSON ausente")
            continue
        try:
            doc = json.loads(arq.read_text(encoding="utf-8"))
        except Exception as e:
            erros.append(f"{doc_id}: JSON inválido ({e})")
            continue
        if doc.get("documento_id") != doc_id:
            erros.append(f"{doc_id}: documento_id divergente "
                         f"({doc.get('documento_id')!r})")
        if not doc.get("schema_version"):
            erros.append(f"{doc_id}: schema_version ausente")

        cits = doc.get("citacoes", [])
        n_cit += len(cits)
        spans = []
        for k, c in enumerate(cits, 1):
            pre = f"{doc_id} #{k}"
            try:
                ini, fim = int(c["inicio"]), int(c["fim"])
            except Exception:
                erros.append(f"{pre}: inicio/fim ausente ou não inteiro")
                continue
            if not (0 <= ini < fim <= len(texto)):
                erros.append(f"{pre}: span ({ini},{fim}) fora de "
                             f"[0,{len(texto)}] ou invertido")
                continue
            if c.get("trecho") != texto[ini:fim]:
                erros.append(f"{pre}: trecho != texto[{ini}:{fim}] "
                             f"({c.get('trecho')!r})")
            if c.get("tipo") not in TIPOS:
                erros.append(f"{pre}: tipo {c.get('tipo')!r} inválido")
            classe = c.get("classificacao")
            if classe not in CLASSES:
                erros.append(f"{pre}: classificacao {classe!r} inválida")
            resol = c.get("resolucao") or {}
            idc = str(resol.get("id_canonico") or "")
            if classe == "real":
                if not idc:
                    erros.append(f"{pre}: classe real sem id_canonico")
                elif not idc.isdigit():
                    erros.append(f"{pre}: id_canonico {idc!r} não numérico")
            conf = c.get("confianca")
            if conf is not None and not (0.0 <= float(conf) <= 1.0):
                erros.append(f"{pre}: confianca {conf} fora de [0, 1]")
            spans.append((ini, fim))
        for a in range(len(spans)):
            for b in range(a + 1, len(spans)):
                if iou(spans[a], spans[b]) >= 0.5:
                    erros.append(f"{doc_id}: citacoes #{a+1} e #{b+1} com "
                                 f"IoU >= 0,5 (a métrica rejeita duplicatas)")

    esperados = {p.stem for p in Path(pasta_txt).glob("*.txt")}
    for arq in sorted(Path(pasta_json).glob("*.json")):
        if arq.stem not in esperados:
            erros.append(f"{arq.stem}: JSON sem .txt correspondente")

    print(f"{docs} documentos, {n_cit} citações.")
    if erros:
        print(f"\n{len(erros)} problema(s) no contrato:")
        for e in erros[:60]:
            print("  -", e)
        sys.exit(1)
    print("Contrato válido.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
