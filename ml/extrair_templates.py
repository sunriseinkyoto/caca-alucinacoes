# -*- coding: utf-8 -*-
"""
Extração dos blocos de texto que compõem os documentos do desafio.

Os 26 documentos de desenvolvimento são sintéticos, montados a partir de um
conjunto pequeno de blocos: cabeçalho de autuação, títulos de seção, frases
de enchimento e frases-veículo (as que hospedam uma citação). Este script
separa esses blocos e os grava em templates.json, que alimenta o gerador de
corpus sintético (ml/gerar_sinteticos.py) e fornece ao extrator a lista de
frases neutras.

Uso:
    python ml/extrair_templates.py <pasta_txt> <goldenset.csv> [templates.json]
"""
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

MARCA = "{CIT}"
RE_SECAO = re.compile(r"^[IVXLC]+\s*[—–-]\s*.+$", re.M)
# abreviações que terminam em ponto sem encerrar a frase. A fronteira à
# esquerda é necessária: sem ela, "[A-Z]\." casaria o fim de qualquer
# palavra ("sustenta." -> "a.") e o texto deixaria de ser fatiado.
ABREV = re.compile(
    r"(?<![A-Za-zÀ-ÿ])"
    r"(?:[Ff][Ll][Ss]?|[Aa]rts?|[Nn]os?|[Nn]|[Mm]in|[Rr]el|[Pp]roc|[Pp]p?"
    r"|[Ii]nc|[Pp]ar|[Ss]úm|[A-ZÀ-Ý])\.\s*$")


def colapsar(s: str) -> str:
    """Colapsa espaços internos, preservando a presença de espaço nas bordas."""
    esq = " " if s[:1].isspace() else ""
    dir_ = " " if s[-1:].isspace() else ""
    miolo = " ".join(s.split())
    return f"{esq}{miolo}{dir_}" if miolo else (esq or dir_)


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def fatiar_frases(texto, spans):
    """Divide o parágrafo em frases sem cortar o interior de uma citação."""
    proibido = set()
    for ini, fim in spans:
        proibido.update(range(ini, fim))
    cortes = [0]
    for m in re.finditer(r"(?<=[.!?])\s+", texto):
        if m.start() - 1 in proibido:
            continue
        if ABREV.search(texto[max(0, m.start() - 8):m.start()]):
            continue                       # "fls. 187/850", "art. 5º", "Min. X"
        cortes.append(m.end())
    cortes.append(len(texto))
    return [(cortes[i], cortes[i + 1]) for i in range(len(cortes) - 1)]


def main(pasta_txt, gold_csv, saida="templates.json"):
    gold = defaultdict(list)
    for r in csv.DictReader(open(gold_csv, encoding="utf-8-sig")):
        gold[r["documento_id"]].append(
            (int(r["inicio"]), int(r["fim"]), r["tipo"], r["classificacao"]))

    cabecalhos, secoes, enchimento, neutras = [], [], set(), set()
    veiculos = defaultdict(set)     # (tipo, classe) -> {frase com {CIT}}
    fechos = set()

    for doc_id, spans in sorted(gold.items()):
        texto = Path(pasta_txt, f"{doc_id}.txt").read_text(encoding="utf-8")
        pares = sorted((a, b) for a, b, _, _ in spans)
        info = {(a, b): (t, c) for a, b, t, c in spans}

        # --- cabeçalho: tudo o que precede o parágrafo da primeira citação.
        # Cortar no primeiro título de seção incluiria citações não anotadas,
        # que se tornariam falsos negativos no treino do rotulador.
        primeira = pares[0][0] if pares else len(texto)
        corte = texto.rfind("\n\n", 0, primeira)
        corte = corte if corte > 0 else min(primeira, 400)
        cabecalhos.append(texto[:corte].rstrip() + "\n\n")
        if "_n1_" in doc_id:
            secoes.extend(RE_SECAO.findall(texto))

        # --- frases
        for ini_f, fim_f in fatiar_frases(texto, pares):
            frase = texto[ini_f:fim_f]
            dentro = [(a, b) for a, b in pares if ini_f <= a and b <= fim_f]
            if not dentro:
                limpa = " ".join(frase.split())
                # apenas do nível 1: as frases do nível 2 já contêm ruído de
                # OCR, que o gerador aplica depois, de forma controlada
                if 40 < len(limpa) < 240 and not RE_SECAO.match(limpa):
                    # `neutras` guarda todas as frases sem citação, dos dois
                    # níveis: são elas que mostram que uma expressão com forma de
                    # citação vaga pode ocorrer sem ser citação
                    neutras.add(limpa)
                    if ("_n1_" in doc_id and len(limpa) < 220
                            and not re.match(r"^[\d/]", limpa)):
                        enchimento.add(limpa)
                continue
            if len(dentro) > 1:
                continue                       # frase com duas citações: rara, descartada
            a, b = dentro[0]
            tpl = colapsar(frase[:a - ini_f]) + MARCA + colapsar(frase[b - ini_f:])
            tipo, classe = info[(a, b)]
            if 15 < len(tpl) < 240:
                veiculos[f"{tipo}|{classe}"].add(tpl)

        # --- fecho: último parágrafo
        if "_n1_" in doc_id:
            fechos.add(texto[texto.rfind("\n\n") + 2:].strip())

    dados = {
        "cabecalhos": cabecalhos,
        "secoes": sorted(set(secoes)),
        "enchimento": sorted(enchimento),
        "veiculos": {k: sorted(v) for k, v in veiculos.items()},
        "fechos": sorted(fechos),
        "neutras": sorted(neutras),
    }
    Path(saida).write_text(json.dumps(dados, ensure_ascii=False, indent=1, sort_keys=True),
                           encoding="utf-8")
    print(f"{len(cabecalhos)} cabecalhos, {len(dados['secoes'])} secoes, "
          f"{len(dados['enchimento'])} frases de enchimento, "
          f"{len(dados['neutras'])} frases neutras, "
          f"{sum(len(v) for v in dados['veiculos'].values())} frases-veiculo "
          f"em {len(dados['veiculos'])} combinacoes tipo|classe -> {saida}")
    for k, v in sorted(dados["veiculos"].items()):
        print(f"   {k:28s} {len(v):3d}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "templates.json")
