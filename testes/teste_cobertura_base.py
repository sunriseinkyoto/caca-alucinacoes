# -*- coding: utf-8 -*-
"""
Teste de cobertura sobre a base canônica inteira.

Os 26 documentos de desenvolvimento citam apenas uma fração das classes
processuais presentes na base. Como toda citação ``real`` do conjunto de
avaliação resolve, por construção, para um registro da base, a população de
citações reais possíveis é conhecida de antemão: é a própria base.

Para cada acórdão, este teste reconstrói a citação a partir do cabeçalho do
registro — classe processual e número —, escreve-a em cada uma das formas de
superfície suportadas, insere o resultado numa frase neutra e verifica se o
extrator (1) detecta a citação com IoU >= 0,5 em relação ao span esperado,
(2) classifica-a como ``real`` e (3) devolve o ``id_canonico`` do registro.

Uso:
    python testes/teste_cobertura_base.py <desafio1_bracis.db> <indice.json> [--detalhe]
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from extrair import Resolvedor, extrair  # noqa: E402
from indexar import ESTADOS, sem_acento  # noqa: E402

CONECTIVOS = {"NO", "NA", "NOS", "NAS", "EM", "DE", "DO", "DA", "DOS", "DAS",
              "COM", "E", "CONTRA"}
# grafia das siglas do cabeçalho quando escritas por extenso no corpo do texto
EXTENSO = {
    "AG.REG.": "Agravo Regimental", "AGRG": "Agravo Regimental",
    "AGINT": "Agravo Interno", "EDCL": "Embargos de Declaração",
    "EMB.DECL.": "Embargos de Declaração", "EMB.DIV.": "Embargos de Divergência",
    "EDV": "Embargos de Divergência", "QO": "Questão de Ordem",
    "PEXT": "Pedido de Extensão", "RESP": "Recurso Especial",
    "ORD.": "Ordinário", "NOSEMBARGOS": "nos Embargos",
}
RE_CNJ = re.compile(r"\d{1,7}-\s?\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")


def classe_do_cabecalho(trib: str, h: str) -> str | None:
    if trib == "TST":
        m = re.search(r"discutidos\s+estes\s+autos\s+de\s+(.{5,160}?)\s+"
                      r"(?:n[ºo°.]|TST-)", h, re.I | re.S)
    elif trib == "STF":
        m = re.search(r"(?:TURMA|PLEN[ÁA]RIO|PLENO)\s+(.{3,120}?)\s+\d[\d.]*\s", h)
    elif trib == "TSE":
        m = re.search(r"AC[ÓO]RD[ÃA]O\s+(.{3,160}?)\s+N[ºo°.]", h)
    elif trib == "STM":
        m = re.search(r"(?:/\d{4}\s+|Pleno\s+)([A-ZÇÃÁÉÍÓÚÊÔ][A-ZÇÃÁÉÍÓÚÊÔ\s]{3,120}?)"
                      r"\s+N[ºo°.]", h)
    else:
        m = re.match(r"\s*(?:SUPERIOR TRIBUNAL DE JUSTIÇA\s+)?(.{3,160}?)\s+N[ºo°.]", h)
    if not m:
        return None
    k = " ".join(m.group(1).split())
    if not re.fullmatch(r"[A-Za-zÀ-ÿ.\s]+", k) or len(k.split()) > 16:
        return None                       # cabeçalho fora do padrão: fica de fora
    return re.sub(r"^A\s+", "", k)


def numero_do_cabecalho(trib: str, h: str) -> str | None:
    if trib == "STJ":
        m = re.search(r"N[ºo°.]\s*(\d{1,3}(?:\.\d{3})+|\d+)\s*-\s*([A-Z]{2})\b", h)
        return f"{m.group(1)}/{m.group(2)}" if m else None
    if trib == "STF":
        hs = sem_acento(h).upper()
        m = re.search(r"\b(\d{1,3}(?:\.\d{3})+|\d{3,7})\s+(" + "|".join(ESTADOS) + r")\b", hs)
        return f"{m.group(1)}/{ESTADOS[m.group(2)]}" if m else None
    if trib == "TST":
        m = re.search(r"discutidos\s+estes\s+autos.{0,200}?(" + RE_CNJ.pattern + ")",
                      h, re.I | re.S)
        return m.group(1).replace(" ", "") if m else None
    m = RE_CNJ.search(h)
    return m.group(0).replace(" ", "") if m else None


def por_extenso(classe: str) -> str:
    """'AGINT NO RECURSO EM MANDADO DE SEGURANÇA' -> 'Agravo Interno no Recurso
    em Mandado de Segurança'."""
    saida = []
    for tok in classe.split():
        up = tok.upper()
        if up in EXTENSO:
            saida.append(EXTENSO[up])
        elif up in CONECTIVOS:
            saida.append(up.lower())
        else:
            saida.append(tok[:1].upper() + tok[1:].lower())
    return " ".join(saida)


# Siglas por tribunal, na grafia usual de cada corte. Aplicadas da locução mais
# longa para a mais curta; o que não tem sigla consagrada fica por extenso.
SIGLAS_POR_TRIBUNAL = {
    "STJ": [("agravo interno", "AgInt"), ("agravo regimental", "AgRg"),
            ("embargos de declaração", "EDcl"),
            ("agravo em recurso especial", "AREsp"), ("recurso especial", "REsp"),
            ("recurso em habeas corpus", "RHC"), ("habeas corpus", "HC"),
            ("recurso em mandado de segurança", "RMS"), ("mandado de segurança", "MS"),
            ("conflito de competência", "CC"), ("reclamação", "Rcl"),
            ("ação rescisória", "AR"), ("petição", "Pet")],
    "STF": [("agravo regimental", "AgR"), ("embargos de declaração", "ED"),
            ("recurso extraordinário com agravo", "ARE"),
            ("recurso extraordinário", "RE"), ("reclamação", "Rcl"),
            ("habeas corpus", "HC"), ("ação rescisória", "AR"),
            ("mandado de segurança", "MS"), ("ação direta de inconstitucionalidade", "ADI")],
    "TSE": [("agravo regimental", "AgR"), ("embargos de declaração", "ED"),
            ("agravo em recurso especial eleitoral", "AREspe"),
            ("recurso especial eleitoral", "REspe"), ("recurso ordinário eleitoral", "ROE"),
            ("recurso ordinário", "RO"), ("agravo de instrumento", "AI"),
            ("mandado de segurança", "MS"), ("habeas corpus", "HC")],
    "TST": [("embargos de declaração", "ED"),
            ("agravo de instrumento em recurso de revista", "AIRR"),
            ("recurso de revista com agravo", "ARR"), ("recurso de revista", "RR")],
    "STM": [("recurso em sentido estrito", "RSE"), ("habeas corpus", "HC"),
            ("embargos de declaração", "ED")],
}


def por_sigla(extenso: str, trib: str) -> str | None:
    s = extenso
    for frase, sigla in sorted(SIGLAS_POR_TRIBUNAL.get(trib, []), key=lambda x: -len(x[0])):
        s = re.sub(rf"(?i)\b{re.escape(frase)}\b", sigla, s)
    return s if s != extenso else None


def com_ocr(s: str) -> str:
    """Aplica as confusões de OCR descritas pela organização à classe processual:
    m -> rn em todas as ocorrências e o -> 0, l -> 1 na primeira palavra longa."""
    s = s.replace("m", "rn")
    palavras = s.split(" ")
    for i, w in enumerate(palavras):
        if len(w) >= 6:
            palavras[i] = w.replace("o", "0", 1).replace("l", "1", 1)
            break
    return " ".join(palavras)


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    uni = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / uni if uni else 0.0


RUIDO_CABECALHO = re.compile(r"(?i)superior tribunal|revista eletr|recorrido|s[uú]mula"
                             r"|classe|exporta|\d")


def chaves_do_numero(numero: str) -> list[str]:
    """As mesmas chaves que o índice usa: dígitos sem zeros à esquerda, com e sem UF."""
    num, _, uf = numero.partition("/")
    d = re.sub(r"\D", "", num).lstrip("0") or "0"
    return ([f"{d}|{uf}"] if uf else []) + [d]


def main(db, indice_path, detalhe=False):
    indice = json.loads(Path(indice_path).read_text(encoding="utf-8"))
    res = Resolvedor(indice)
    con = sqlite3.connect(db)
    falhas = defaultdict(list)
    total = Counter()
    ok = Counter()
    fora = Counter()
    for cid, trib, texto in con.execute(
            "SELECT id, tribunal, texto FROM documentos WHERE natureza='acordao'"):
        cid = str(cid)
        h = texto[:1500]
        classe, numero = classe_do_cabecalho(trib, h), numero_do_cabecalho(trib, h)
        if not classe or not numero or RUIDO_CABECALHO.search(classe):
            fora["cabeçalho fora do padrão"] += 1
            continue
        # o número reconstruído precisa ser o número PRÓPRIO do registro no índice
        donos = next((indice["acordaos"][k] for k in chaves_do_numero(numero)
                      if k in indice["acordaos"]), None)
        if not donos or cid not in donos:
            fora["número do cabeçalho não é o do registro"] += 1
            continue
        if len(donos) > 1:
            # duplicata no acervo: a organização garante que nenhuma citação
            # aponta para duplicatas, então o caso não é testável
            fora["número compartilhado por mais de um registro"] += 1
            continue
        extenso = por_extenso(classe)
        formas = [("extenso", extenso)]
        sigla = por_sigla(extenso, trib)
        if sigla:
            formas.append(("sigla", sigla))
        formas.append(("ocr", com_ocr(extenso)))
        for forma, rotulo in formas:
            cit = f"{rotulo} nº {numero}"
            frase = f"Conforme decidido no {cit}, a tese deve prevalecer no caso concreto."
            alvo = (frase.index(cit), frase.index(cit) + len(cit))
            chave = f"{trib}/{forma}"
            total[chave] += 1
            achou = [c for c in extrair(frase, res)
                     if iou((c.inicio, c.fim), alvo) >= 0.5]
            if not achou:
                motivo = "não detectada"
            elif achou[0].classificacao != "real":
                motivo = f"classe {achou[0].classificacao}"
            elif str(achou[0].id_canonico) != cid:
                motivo = "id errado"
            else:
                ok[chave] += 1
                continue
            falhas[(chave, motivo)].append(
                (cit, frase[achou[0].inicio:achou[0].fim] if achou else ""))

    print(f"{'tribunal/forma':14s} {'testadas':>8s} {'corretas':>8s}")
    for t in sorted(total):
        print(f"{t:14s} {total[t]:8d} {ok[t]:8d}   {100*ok[t]/total[t]:5.1f}%")
    print(f"{'TOTAL':14s} {sum(total.values()):8d} {sum(ok.values()):8d}   "
          f"{100*sum(ok.values())/max(1,sum(total.values())):5.1f}%")
    for motivo, n in fora.items():
        print(f"(fora do teste: {n} — {motivo})")
    for (t, motivo), casos in sorted(falhas.items()):
        print(f"\n[{t}] {motivo}: {len(casos)}")
        for cit, visto in casos[: (50 if detalhe else 4)]:
            print(f"    {cit!r}\n       -> {visto!r}")
    return sum(ok.values()) == sum(total.values())


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2], "--detalhe" in sys.argv) else 1)
