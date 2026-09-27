# -*- coding: utf-8 -*-
"""
Gerador de corpus sintético com gabarito.

Combina três fontes:
  1. templates.json — blocos de texto extraídos dos 26 documentos do desafio;
  2. a base canônica — acórdãos, súmulas e dispositivos com número, tribunal,
     ano, relator e classe processual verdadeiros;
  3. ml/ruido_ocr.py — o modelo de ruído do nível 2.

A saída (pasta txt/ e goldenset.csv) segue o formato do desafio e pode ser
usada diretamente por src/rodar.py, src/avaliar_oficial.py e
src/analisar_erros.py. Serve a dois propósitos: dado de treino para o
rotulador de spans e conjunto de teste adversarial, com gabarito conhecido,
para o sistema de regras.

Uso:
    python ml/gerar_sinteticos.py <db> <templates.json> <indice.json> <saida/> \
           [--n 400] [--nivel 0|1|2] [--seed 7]
"""
import argparse
import csv
import json
import random
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ruido_ocr import (ruido_texto, ruido_identificador,  # noqa: E402
                       preserva_identidade, digitos)

UFS = ("AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR "
       "SC SP SE TO").split()


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# ==================================================== formas de citação
# Variantes de superfície da mesma classe processual; o nível 2 combina todas.
ABREVS = {
    "recurso especial": ["REsp", "R.Esp.", "Rec. Esp.", "RESP",
                         "Recurso Especial", "REsp."],
    "agravo em recurso especial": ["AREsp", "AREsp.", "ARESP", "A.REsp.",
                                   "Agravo em Recurso Especial"],
    "reclamacao": ["Rcl", "RCL", "Recl.", "Reclamação", "Rcl."],
    "habeas corpus": ["HC", "H.C.", "Habeas Corpus"],
    "recurso em habeas corpus": ["RHC", "R.H.C.", "Recurso em Habeas Corpus"],
    "recurso extraordinario": ["RE", "R.E.", "Recurso Extraordinário"],
    "apelacao": ["APL", "Apelação", "Apel.", "APL."],
    "recurso sentido estrito": ["RSE", "Recurso em Sentido Estrito", "R.S.E."],
    "agravo interno": ["AgInt", "Ag. Int.", "AGINT", "Agravo Interno"],
    "agravo regimental": ["AgRg", "Ag.Rg.", "AGRG", "Agravo Regimental"],
    "embargos declaracao": ["EDcl", "EDs", "ED", "Embargos de Declaração"],
    "recurso especial eleitoral": ["REspe", "RESPE", "R.Espe.",
                                   "Recurso Especial Eleitoral"],
    "recurso revista": ["RR", "R.R.", "Recurso de Revista"],
    "acao rescisoria": ["AR", "A.R.", "Ação Rescisória"],
    "mandado seguranca": ["MS", "M.S.", "Mandado de Segurança"],
}
PREFIXOS = ["", "AgRg no ", "AgInt no ", "EDcl no ", "EDcl no AgInt no ",
            "AgInt nos EDcl no ", "Terceiro AG.REG na "]
MARCAS_N = ["nº ", "n. ", "n° ", "N° ", "Nº ", "No ", "", "n.º "]
SEP_UF = ["/{uf}", " - {uf}", " ({uf})", "-{uf}", " – {uf}", "/ {uf}"]


def formatar_numero(num: str, rng: random.Random) -> str:
    """Varia a forma de escrita do número sem alterar seus dígitos.

    '1.522.200' -> '1522200' | '1.522. 200' | '1 522 200'; números CNJ também
    com a pontuação removida ou espaçada.
    """
    d = digitos(num)
    if "-" in num:                                    # número CNJ: preserva a forma
        return rng.choice([num, num, num.replace(".", " "),
                           num.replace(".", ""), num.replace(".", "", 1),
                           num.replace("-", " - ")])
    formas = [num, d]
    if len(d) > 3:
        partes, resto = [], d
        while len(resto) > 3:
            partes.insert(0, resto[-3:])
            resto = resto[:-3]
        pontuado = ".".join(([resto] if resto else []) + partes)
        formas += [pontuado, pontuado, pontuado.replace(".", " "),
                   pontuado.replace(".", ". ")]
    return rng.choice(formas)


def render_juris(num, uf, classe_semantica, rng, ruidoso):
    """Monta a forma de superfície de uma citação de jurisprudência."""
    fam = ABREVS.get(classe_semantica) or rng.choice(list(ABREVS.values()))
    corpo = (rng.choice(PREFIXOS) + rng.choice(fam) + " "
             + rng.choice(MARCAS_N) + formatar_numero(num, rng))
    if uf and rng.random() < 0.85:
        corpo += rng.choice(SEP_UF).format(uf=uf)
    if ruidoso:
        for _ in range(6):
            cand = ruido_identificador(corpo, rng)
            if preserva_identidade(corpo, cand):
                return cand
    return corpo


# =========================================================== registros da base
RE_CNJ = re.compile(r"\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")
RE_NUM_UF = re.compile(r"N[ºo°.\s]{0,3}\s*(\d{1,3}(?:\.\d{3})+|\d{3,9})"
                       r"\s*[-–—/(]?\s*(" + "|".join(UFS) + r")\b", re.I)
RE_TST_ANCORA = re.compile(r"discutidos\s+estes\s+autos[^\n]{0,200}?"
                           r"(\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})", re.I)


def superficie_canonica(texto, tribunal):
    """Devolve ``(número como escrito, UF)`` do próprio processo, lido do cabeçalho.

    A chave do índice não serve para isso: contém apenas dígitos, sem zeros à
    esquerda, e reconstruir um número CNJ a partir dela produziria um número
    inválido ('0600216-46.2020.6.14.0022' -> '6002164620206140022').
    """
    if tribunal == "TST":
        m = RE_TST_ANCORA.search(texto)
        if m:
            return m.group(1), None
    cab = texto[:1300]
    m = RE_NUM_UF.search(cab)
    if m:
        return m.group(1), m.group(2).upper()
    m = RE_CNJ.search(cab)
    if m:
        return m.group(0), None
    return None, None


def carregar_pools(db_path, indice):
    """Seleciona apenas acórdãos cuja forma canônica resolve para um único id no
    índice, de modo que o gabarito sintético seja correto por construção.
    """
    con = sqlite3.connect(db_path)
    prim, meta = indice["acordaos"], indice["meta"]
    acordaos, relatores = [], {}

    q = "SELECT id, tribunal, ano, relator, natureza, texto FROM documentos"
    for cid, trib, ano, rel, nat, texto in con.execute(q):
        if nat != "acordao":
            continue
        cid = str(cid)
        num, uf = superficie_canonica(texto, trib)
        if not num:
            continue
        chave = digitos(num).lstrip("0") or "0"
        if prim.get(chave) != [cid]:
            continue                       # ambíguo ou não indexado: excluído
        acordaos.append({"id": cid, "num": num, "dig": digitos(num), "uf": uf,
                         "tribunal": trib, "ano": ano, "relator": rel,
                         "classe": meta.get(cid, {}).get("classe_cab", "")})
        if rel and trib:
            relatores.setdefault(trib, set()).add((rel, ano))

    sumulas = [(k, v) for k, v in indice["sumulas"].items() if "SV" not in k]
    leis = list(indice["leis"].items())
    return acordaos, {k: sorted(v) for k, v in relatores.items()}, sumulas, leis


NOMES_LEI = {
    "CPC": ["do CPC", "do Código de Processo Civil", "da Lei nº 13.105/2015"],
    "CPP": ["do CPP", "do Código de Processo Penal"],
    "CF": ["da Constituição Federal", "da Constituição da República", "da CF"],
    "CLT": ["da CLT", "da Consolidação das Leis do Trabalho"],
    "CDC": ["do CDC", "do Código de Defesa do Consumidor"],
    "CODIGO CIVIL": ["do Código Civil"],
    "CODIGO ELEITORAL": ["do Código Eleitoral"],
    "CODIGO PENAL MILITAR": ["do Código Penal Militar", "do CPM"],
    "LC 64/1990": ["da Lei Complementar nº 64/1990", "da LC 64/1990"],
}
INCISOS = ["", ", I,", ", II,", ", IX,", ", LV,", ", XXIX,", ", § 1º-A,"]


def render_lei(apelido, artigo, rng, ruidoso):
    art = rng.choice(["art. ", "artigo ", "art ", "Art. "])
    corpo = f"{art}{artigo}{rng.choice(INCISOS)} {rng.choice(NOMES_LEI[apelido])}"
    return ruido_texto(corpo, rng, 0.02) if ruidoso else corpo


VAGAS_JURIS = [
    "julgado do {t} proferido em {a} pela relatoria de {r}",
    "acórdão do {t} julgado em {a} sob relatoria de {r}",
    "precedente do {t} de {a}, da relatoria de {r}",
    "{c} do {t}, de {a}, Rel. Min. {r}",
    "{c} de {a}, Rel. Min. {r}",
]
TRIB_EXTENSO = {"STF": "Supremo Tribunal Federal",
                "STJ": "Superior Tribunal de Justiça",
                "STM": "Superior Tribunal Militar",
                "TSE": "Tribunal Superior Eleitoral",
                "TST": "Tribunal Superior do Trabalho"}
CLASSES_VAGAS = ["Reclamação", "Rcl", "Recurso Especial", "APL", "AREsp",
                 "Agravo em Recurso Especial", "Recurso em Habeas Corpus"]
# Segue a política de anotação vigente: é `incompleta` apenas a referência
# que nomeia tribunal, ano e relator. Um gerador com outra política
# produziria gabarito sintético incompatível com o oficial.
VAGAS_LEI = []


def render_vaga_juris(relatores, rng, ruidoso):
    trib = rng.choice(list(relatores)) if relatores else "STF"
    rel, ano = rng.choice(relatores[trib]) if relatores.get(trib) else ("Fulano", 2020)
    corpo = rng.choice(VAGAS_JURIS).format(
        t=trib, tx=TRIB_EXTENSO.get(trib, trib), a=ano or 2020,
        r=re.sub(r"^(Min\.|Ministr[ao])\s*", "", rel or "Fulano"),
        c=rng.choice(CLASSES_VAGAS))
    return ruido_texto(corpo, rng, 0.02) if ruidoso else corpo


# ================================================================== montagem
class Documento:
    """Acumula segmentos de texto e mantém os offsets das citações em codepoints."""

    def __init__(self):
        self.partes = []
        self.spans = []

    def texto(self, s):
        self.partes.append(s)

    def citacao(self, s, tipo, classe, id_canonico):
        pos = sum(len(p) for p in self.partes)
        self.partes.append(s)
        self.spans.append({"inicio": pos, "fim": pos + len(s), "trecho": s,
                           "tipo": tipo, "classificacao": classe,
                           "id_canonico": id_canonico or ""})

    def render(self):
        return "".join(self.partes), self.spans


def cnj_falso(rng):
    return (f"{rng.randint(1000000, 9999999)}-{rng.randint(10, 99)}."
            f"{rng.randint(2015, 2025)}.{rng.randint(1, 8)}."
            f"{rng.randint(1, 20):02d}.{rng.randint(1000, 9999)}")


def numero_inexistente(rng, indice):
    """Sorteia um número comprovadamente ausente da base, para que a citação
    rotulada como ``inventada`` não seja, na verdade, real.
    """
    for _ in range(50):
        cand = (cnj_falso(rng) if rng.random() < 0.4
                else str(rng.randint(10000, 9999999)))
        chave = digitos(cand).lstrip("0") or "0"
        if chave not in indice["acordaos"] and chave not in indice["acordaos_sec"]:
            return cand
    return cnj_falso(rng)


def gerar_documento(doc_id, tpl, pools, rng, nivel, indice):
    acordaos, relatores, sumulas, leis = pools
    ruidoso = nivel == 2
    d = Documento()

    # --- cabeçalho, com os distratores (Processo nº, fls., Protocolo)
    cab = rng.choice(tpl["cabecalhos"])
    cab = re.sub(r"\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}",
                 lambda _m: cnj_falso(rng), cab)
    d.texto(ruido_texto(cab, rng, 0.006) if ruidoso else cab)

    usados = set()
    n_secoes = rng.randint(3, 4)
    for s in range(n_secoes):
        if s < len(tpl["secoes"]):
            titulo = tpl["secoes"][s]
            d.texto("\n" + (ruido_texto(titulo, rng, 0.01) if ruidoso else titulo)
                    + "\n\n")
        for _ in range(rng.randint(3, 6)):
            if rng.random() < 0.45:                       # frase de enchimento
                f = rng.choice(tpl["enchimento"])
                d.texto((ruido_texto(f, rng) if ruidoso else f) + " ")
                continue

            tipo = "lei" if rng.random() < 0.18 else "jurisprudencia"
            classe = rng.choices(["real", "inventada", "incompleta"],
                                 weights=[48, 28, 24])[0]
            chave = f"{tipo}|{classe}"
            if chave not in tpl["veiculos"]:
                continue
            molde = rng.choice(tpl["veiculos"][chave])
            antes, depois = molde.split("{CIT}")
            if ruidoso:
                antes, depois = ruido_texto(antes, rng), ruido_texto(depois, rng)
            d.texto(antes)

            idc = ""
            if tipo == "lei":
                if classe == "real":
                    k, idc = rng.choice(leis)
                    ap, art = k.split("|")
                    sup = render_lei(ap, art, rng, ruidoso)
                elif classe == "inventada":
                    ap = rng.choice(list(NOMES_LEI))
                    art = rng.randint(100, 1200)
                    while f"{ap}|{art}" in indice["leis"]:
                        art = rng.randint(100, 1200)
                    sup = render_lei(ap, str(art), rng, ruidoso)
                else:
                    continue          # a política vigente não tem citação vaga de lei
            else:
                if classe == "real":
                    if rng.random() < 0.08 and sumulas:
                        k, idc = rng.choice(sumulas)
                        trib, num = k.split("|")
                        sup = f"Súmula {num} do {trib}"
                        sup = ruido_texto(sup, rng, 0.03) if ruidoso else sup
                    else:
                        for _ in range(12):
                            ac = rng.choice(acordaos)
                            if ac["id"] not in usados:
                                break
                        usados.add(ac["id"])
                        idc = ac["id"]
                        sup = render_juris(ac["num"], ac["uf"],
                                           classe_semantica(ac["classe"]),
                                           rng, ruidoso)
                elif classe == "inventada":
                    if rng.random() < 0.10:
                        trib = rng.choice(["STF", "STJ", "TSE", "TST"])
                        n = rng.randint(500, 999)
                        while f"{trib}|{n}" in indice["sumulas"]:
                            n = rng.randint(500, 999)
                        sup = f"Súmula {n} do {trib}"
                    else:
                        falso = numero_inexistente(rng, indice)
                        sup = render_juris(falso, rng.choice(UFS),
                                           rng.choice(list(ABREVS)), rng, ruidoso)
                else:
                    sup = render_vaga_juris(relatores, rng, ruidoso)

            d.citacao(sup, tipo, classe, idc)
            d.texto(depois)
        d.texto("\n\n")

    fecho = rng.choice(tpl["fechos"])
    d.texto("\n" + (ruido_texto(fecho, rng, 0.006) if ruidoso else fecho) + "\n")
    return d.render()


def classe_semantica(classe_cab: str) -> str:
    """'AGINT NO RECURSO ESPECIAL' -> chave correspondente em ABREVS."""
    c = sem_acento(classe_cab).lower()
    for chave in ABREVS:
        if all(p in c for p in chave.split()):
            return chave
    return "recurso especial"


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("templates")
    ap.add_argument("indice")
    ap.add_argument("saida")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--nivel", type=int, default=0, help="0 = metade N1, metade N2")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    tpl = json.loads(Path(a.templates).read_text(encoding="utf-8"))
    indice = json.loads(Path(a.indice).read_text(encoding="utf-8"))
    pools = carregar_pools(a.db, indice)
    print(f"pools: {len(pools[0])} acordaos resolviveis, "
          f"{sum(len(v) for v in pools[1].values())} pares relator/ano, "
          f"{len(pools[2])} sumulas, {len(pools[3])} leis")

    out = Path(a.saida)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    linhas = []
    for i in range(a.n):
        nivel = a.nivel or (1 if i % 2 == 0 else 2)
        doc_id = f"syn_n{nivel}_{i:05d}"
        texto, spans = gerar_documento(doc_id, tpl, pools, rng, nivel, indice)
        texto = unicodedata.normalize("NFC", texto)
        (out / "txt" / f"{doc_id}.txt").write_text(texto, encoding="utf-8",
                                                   newline="\n")
        for k, s in enumerate(spans, 1):
            assert texto[s["inicio"]:s["fim"]] == s["trecho"], doc_id
            linhas.append({"nivel": nivel, "documento_id": doc_id,
                           "citacao_id": f"g{k}", "inicio": s["inicio"],
                           "fim": s["fim"], "trecho": s["trecho"],
                           "tipo": s["tipo"],
                           "classificacao": s["classificacao"],
                           "id_canonico": s["id_canonico"]})

    with (out / "goldenset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nivel", "documento_id", "citacao_id",
                                          "inicio", "fim", "trecho", "tipo",
                                          "classificacao", "id_canonico"])
        w.writeheader()
        w.writerows(linhas)
    print(f"{a.n} documentos, {len(linhas)} citações -> {out}")


if __name__ == "__main__":
    main()
