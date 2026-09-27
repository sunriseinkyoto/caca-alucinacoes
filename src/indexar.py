# -*- coding: utf-8 -*-
"""
Construção do índice da base canônica (desafio1_bracis.db).

O índice de acórdãos tem duas camadas:

  acordaos       camada primária: o número do próprio processo, lido do
                 cabeçalho do registro segundo a regra de cada tribunal;
  acordaos_sec   camada secundária: os demais números do cabeçalho, usados
                 apenas quando a primária não resolve.

A separação é o que garante precisão. Uma busca no texto integral (FTS)
devolve todo acórdão que menciona um número, e acórdãos citam uns aos outros
com frequência; só o cabeçalho identifica o processo. Consultar a camada
primária primeiro evita múltiplos candidatos espúrios.

Saída: indice.json com as chaves acordaos, acordaos_sec, sumulas, leis e meta.

Uso:
    python src/indexar.py <desafio1_bracis.db> [indice.json]
"""
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

UFS = ("AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR "
       "SC SP SE TO").split()

ESTADOS = {
    "RIO GRANDE DO SUL": "RS", "RIO GRANDE DO NORTE": "RN",
    "RIO DE JANEIRO": "RJ", "SAO PAULO": "SP", "MINAS GERAIS": "MG",
    "DISTRITO FEDERAL": "DF", "SANTA CATARINA": "SC",
    "MATO GROSSO DO SUL": "MS", "MATO GROSSO": "MT", "ESPIRITO SANTO": "ES",
    "BAHIA": "BA", "PARANA": "PR", "PERNAMBUCO": "PE", "CEARA": "CE",
    "GOIAS": "GO", "PARAIBA": "PB", "PIAUI": "PI", "MARANHAO": "MA",
    "ALAGOAS": "AL", "SERGIPE": "SE", "AMAZONAS": "AM", "AMAPA": "AP",
    "ACRE": "AC", "RONDONIA": "RO", "RORAIMA": "RR", "TOCANTINS": "TO",
    "PARA": "PA",
}


def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def so_digitos(s: str) -> str:
    return re.sub(r"\D", "", s)


def chave(numero: str, uf: str | None) -> str:
    d = so_digitos(numero).lstrip("0") or "0"
    return f"{d}|{uf.upper()}" if uf else d


# ------------------------------------------------------------------ acórdãos
RE_CNJ = re.compile(r"\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")
# O TST não traz o número do processo no topo do documento: ele aparece na
# fórmula "Vistos, relatados e discutidos estes autos de <classe> nº
# TST-<CNJ>". Essa âncora identifica o processo em 199 dos 201 acórdãos do
# TST e evita indexar processos que o acórdão apenas cita.
RE_TST_ANCORA = re.compile(
    r"discutidos\s+estes\s+autos[^\n]{0,200}?"
    r"(\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})", re.I)
RE_TST_FALLBACK = re.compile(
    r"PROCESSO\s+N[ºo°.]?\s*TST-[A-Za-z\-]{1,25}?-"
    r"(\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})")
RE_NUM_UF = re.compile(
    r"N[ºo°.\s]{0,3}\s*(\d{1,3}(?:\.\d{3})+|\d{3,9})\s*[-–—/(]?\s*("
    + "|".join(UFS) + r")\b", re.I)
RE_NUM_ESTADO = re.compile(
    r"\b(\d{1,3}(?:\.\d{3})+|\d{3,7})\s+(" + "|".join(ESTADOS) + r")\b")
RE_NUM_QUALQUER = re.compile(r"N[ºo°.\s]{0,3}\s*(\d{1,3}(?:\.\d{3})+|\d{4,9})", re.I)


def classe_do_cabecalho(texto: str, n: int = 220) -> str:
    """Trecho do cabeçalho que contém a classe processual.

    Usado apenas para desempatar registros com o mesmo número de processo.
    """
    cab = sem_acento(texto[:n]).upper()
    cab = re.sub(r"A\s*C\s*O\s*R\s*D\s*A\s*O", " ", cab)
    # a classe processual é o que precede o número
    m = re.search(r"N[ºO°.]{0,2}\s*\d|\b\d{3,}", cab)
    if m:
        cab = cab[:m.start()]
    return re.sub(r"[^A-Z0-9]+", " ", cab).strip()


def chaves_do_acordao(texto: str, janela: int = 1300):
    """Devolve as chaves ``(primárias, secundárias)`` de um acórdão."""
    cab = texto[:janela]
    cab_sa = sem_acento(cab).upper()
    prim, sec = set(), set()

    # 1) Número próprio do TST, na fórmula "Vistos, relatados e discutidos
    #    estes autos de <classe> nº TST-<CNJ>". "PROCESSO Nº TST-<CNJ>" é
    #    usado apenas como alternativa quando a fórmula está ausente: o número
    #    pertence ao processo que o traz nessa fórmula, não a um processo
    #    transcrito ou citado no corpo.
    m = RE_TST_ANCORA.search(texto) or RE_TST_FALLBACK.search(texto[:4000])
    if m:
        prim.add(chave(m.group(1), None))

    # 2) "Nº 1.522.200 - SC" (STJ, STM, TSE)
    for m in RE_NUM_UF.finditer(cab):
        num, uf = m.group(1), m.group(2).upper()
        prim.add(chave(num, uf))
        prim.add(chave(num, None))

    # 3) "RECLAMAÇÃO 76.532 RIO DE JANEIRO" (STF)
    for m in RE_NUM_ESTADO.finditer(cab_sa):
        num, uf = m.group(1), ESTADOS[m.group(2)]
        prim.add(chave(num, uf))
        prim.add(chave(num, None))

    # 4) número CNJ no cabeçalho
    for m in RE_CNJ.finditer(cab):
        prim.add(chave(m.group(0), None))

    # 5) alternativa: qualquer "Nº <número>" do cabeçalho, na camada secundária
    for m in RE_NUM_QUALQUER.finditer(cab):
        k = chave(m.group(1), None)
        if k not in prim:
            sec.add(k)

    prim.discard("0")
    sec.discard("0")
    return prim, sec


# -------------------------------------------------------------------- súmulas
# Cada súmula traz o identificador no cabeçalho ("Súmula n. 83 do STJ",
# "Súmula Vinculante n. 10 do STF"), de onde saem as chaves de busca.
RE_SUMULA_CAB = re.compile(
    r"S[uú]mula\s+(?P<vinc>Vinculante\s+)?n?[ºo°.]{0,2}\s*(?P<num>\d{1,4})"
    r"(?:\s+d[oe]\s+(?P<trib>STF|STJ|STM|TSE|TST))?", re.I)


def chaves_da_sumula(cid, texto, tribunal):
    chaves = set()
    trib = (tribunal or "").upper()
    m = RE_SUMULA_CAB.search(texto[:120])
    if m:
        num = m.group("num")
        t = (m.group("trib") or trib).upper()
        if m.group("vinc"):
            chaves.add(f"{t}|SV|{num}")
        chaves.add(f"{t}|{num}")
    return chaves


# -------------------------------------------------- dispositivos legais
# Cada dispositivo identifica o diploma no cabeçalho ("Artigo 276 da Lei
# nº 4.737, de 15 de julho de 1965"). O número do diploma é um dado
# legislativo estável e determina o apelido usado na busca.
LEI_POR_NUMERO = {
    "13105/2015": "CPC",
    "3689/1941": "CPP",
    "1001/1969": "CODIGO PENAL MILITAR",
    "8078/1990": "CDC",
    "4737/1965": "CODIGO ELEITORAL",
    "10406/2002": "CODIGO CIVIL",
    "5452/1943": "CLT",
    "13467/2017": "LEI 13467/2017",
    "9504/1997": "LEI 9504/1997",
}
RE_DISPOSITIVO_CAB = re.compile(
    r"Artigo\s+(?P<art>\d{1,4})[ºo°]?\s+d[ao]\s+(?P<lei>.+?)(?:\n|$)", re.I)
RE_NUM_LEI = re.compile(r"n?[ºo°.]{0,2}\s*(\d{1,3}(?:\.\d{3})*)"
                        r"(?:,\s*de\s+.*?\b(\d{4}))?", re.I)


def ler_dispositivo(texto):
    """Extrai ``(artigo, apelido)`` do cabeçalho de um dispositivo.

    Exemplo: 'Artigo 373 da Lei nº 13.105, de 16 de março de 2015' -> ('373', 'CPC').
    """
    m = RE_DISPOSITIVO_CAB.search(texto[:200])
    if not m:
        return None, None
    art, lei = m.group("art"), m.group("lei")
    lei_sa = sem_acento(lei).lower()
    if "constitui" in lei_sa:
        return art, "CF"
    if "complementar" in lei_sa:
        mn = RE_NUM_LEI.search(lei)
        if mn:
            n = mn.group(1).replace(".", "")
            return art, f"LC {n}/{mn.group(2) or ''}".rstrip("/")
    mn = RE_NUM_LEI.search(lei)
    if mn and mn.group(2):
        chave = f"{mn.group(1).replace('.', '')}/{mn.group(2)}"
        if chave in LEI_POR_NUMERO:
            return art, LEI_POR_NUMERO[chave]
        return art, f"LEI {chave}"
    return art, None


# ----------------------------------------------------------------------- main

def main(db_path, saida="indice.json"):
    con = sqlite3.connect(db_path)
    prim_idx = defaultdict(list)
    sec_idx = defaultdict(list)
    sumulas, leis, meta = {}, {}, {}

    q = ("SELECT documento_id, id, tribunal, ano, relator, natureza, tipo, texto "
         "FROM documentos")
    for doc_id, cid, trib, ano, rel, nat, tipo, texto in con.execute(q):
        cid = str(cid)
        meta[cid] = {"documento_id": doc_id, "tribunal": trib, "ano": ano,
                     "relator": rel, "natureza": nat, "tipo": tipo,
                     "classe_cab": (classe_do_cabecalho(texto)
                                    if nat == "acordao" else "")}
        if nat == "acordao":
            prim, sec = chaves_do_acordao(texto)
            for k in prim:
                prim_idx[k].append(cid)
            for k in sec:
                sec_idx[k].append(cid)
        elif nat == "sumula":
            for k in chaves_da_sumula(cid, texto, trib):
                sumulas[k] = cid
        elif nat == "dispositivo":
            art, ap = ler_dispositivo(texto)
            if art and ap:
                leis[f"{ap}|{art}"] = cid
            else:
                print(f"[aviso] dispositivo {cid} não identificado: "
                      f"{texto[:60]!r}", file=sys.stderr)

    # uma chave que já é primária de algum registro não serve de alternativa
    sec_idx = {k: v for k, v in sec_idx.items() if k not in prim_idx}

    # sort_keys: o índice é um artefato versionado e conferido na verificação
    # de reprodutibilidade, e precisa ser idêntico byte a byte a cada execução.
    Path(saida).write_text(json.dumps(
        {"acordaos": prim_idx, "acordaos_sec": sec_idx,
         "sumulas": sumulas, "leis": leis, "meta": meta},
        ensure_ascii=False, sort_keys=True), encoding="utf-8")

    amb = sum(1 for v in prim_idx.values() if len(v) > 1)
    print(f"indice: {len(prim_idx)} chaves primarias ({amb} ambiguas), "
          f"{len(sec_idx)} secundarias, {len(sumulas)} sumulas, "
          f"{len(leis)} leis -> {saida}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "desafio1_bracis.db",
         sys.argv[2] if len(sys.argv) > 2 else "indice.json")
