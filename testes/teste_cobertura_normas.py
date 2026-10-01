# -*- coding: utf-8 -*-
"""
Teste de cobertura sobre os 13 dispositivos legais e as súmulas da base.

A base contém poucos registros normativos, e todos podem ser citados de várias
formas equivalentes — sigla ou nome por extenso, com ou sem inciso ou
parágrafo, com quebra de linha ou ruído de OCR. A organização exemplifica:
"art. 373, I, do CPC" e "artigo 373, inciso I, do Código de Processo Civil"
apontam para o mesmo registro.

Para cada registro, o teste gera cada forma de superfície, insere-a numa frase
neutra e exige classificação ``real`` com o ``id_canonico`` correto. Como
controle, o mesmo diploma com um artigo inexistente na base deve resultar em
``inventada``, e diplomas fora da base com nome parecido (Constituição
Estadual, Código de Processo Penal Militar) nunca podem resultar em ``real``.

Uso:
    python testes/teste_cobertura_normas.py <indice.json> [--detalhe]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from extrair import Resolvedor, extrair  # noqa: E402

# apelido no índice -> (formas por sigla, formas por extenso), já com a preposição
DIPLOMAS = {
    "CPC": (["do CPC", "do CPC/2015"],
            ["do Código de Processo Civil", "da Lei nº 13.105/2015"]),
    "CPP": (["do CPP"], ["do Código de Processo Penal", "do Decreto-Lei nº 3.689/1941"]),
    "CODIGO PENAL MILITAR": (["do CPM"], ["do Código Penal Militar",
                                          "do Decreto-Lei nº 1.001/1969"]),
    "CDC": (["do CDC"], ["do Código de Defesa do Consumidor", "da Lei nº 8.078/1990"]),
    "CODIGO ELEITORAL": ([], ["do Código Eleitoral", "da Lei nº 4.737/1965"]),
    "CODIGO CIVIL": (["do CC", "do CC/2002"], ["do Código Civil", "da Lei nº 10.406/2002"]),
    "CLT": (["da CLT"], ["da Consolidação das Leis do Trabalho",
                         "do Decreto-Lei nº 5.452/1943"]),
    "CF": (["da CF", "da CF/88", "da CRFB/88"],
           ["da Constituição Federal", "da Constituição da República",
            "da Constituição da República Federativa do Brasil"]),
    "LC 64/1990": (["da LC 64/1990", "da LC nº 64/90"],
                   ["da Lei Complementar nº 64/1990", "da Lei Complementar nº 64/90"]),
}


def com_ocr(s: str) -> str:
    """Confusões descritas pela organização: m -> rn, o -> 0 numa palavra longa,
    s -> 5 num conectivo."""
    s = s.replace("m", "rn").replace(" das ", " da5 ")
    palavras = s.split(" ")
    for i, w in enumerate(palavras):
        if len(w) >= 7 and "o" in w:
            palavras[i] = w.replace("o", "0", 1)
            break
    return " ".join(palavras)


def formas(artigo: str, sigla: list[str], extenso: list[str]):
    for d in sigla:
        yield "sigla", f"art. {artigo} {d}"
        yield "sigla+inciso", f"art. {artigo}, I, {d}"
    for d in extenso:
        yield "extenso", f"artigo {artigo} {d}"
        yield "extenso+inciso", f"artigo {artigo}, inciso I, {d}"
        yield "paragrafo", f"art. {artigo}, § 1º, {d}"
        yield "quebra de linha", f"art.\n{artigo} {d}"
    if extenso:
        yield "ocr", f"arl. {artigo} {com_ocr(extenso[0])}"
        yield "ocr", f"artigõ {artigo} {com_ocr(extenso[0])}"


FORMAS_SUMULA = [
    ("padrão", "Súmula {n} do {t}"), ("com nº", "Súmula nº {n} do {t}"),
    ("barra", "Súmula {n}/{t}"), ("abreviada", "Súm. {n} do {t}"),
    ("maiúsculas", "SÚMULA {n} DO {t}"), ("ocr", "5úrnula {n} do {t}"),
]


# Diplomas FORA da base cujo nome começa como o de um diploma da base. Nenhum
# pode resultar em ``real``: seria uma citação sem registro carimbada como
# verificada — o erro penalizado por τ na métrica oficial.
FORA_DA_BASE = [
    "art. 5º da Constituição Estadual",
    "art. 5º da Constituição do Estado de São Paulo",
    "art. 312 do Código de Processo Penal Militar",
    "art. 186 do Código Civil de 1916",
    "art. 373 da Lei nº 13.106/2015",
    "art. 14 da Lei nº 8.079/1990",
    "art. 1º da Lei Complementar nº 65/1990",
    "art. 93 da CF/89",
]


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    uni = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / uni if uni else 0.0


def formas_genericas(apelido: str):
    """Formas numéricas de um diploma fora da tabela acima ('LEI 2848/1940'),
    para que o teste rode sobre qualquer base no formato do desafio."""
    esp, resto = apelido.split(" ", 1)
    num, ano = resto.split("/")
    milhar = f"{int(num):,}".replace(",", ".")
    nome = "Lei Complementar" if esp == "LC" else "Lei"
    return [], [f"da {nome} nº {milhar}/{ano}", f"da {nome} {num}/{ano}"]


def verificar(cit: str, res: Resolvedor, classe: str, cid: str | None):
    frase = f"Cumpre observar o disposto no {cit}, de aplicação cogente ao caso."
    alvo = (frase.index(cit), frase.index(cit) + len(cit))
    achou = [c for c in extrair(frase, res) if iou((c.inicio, c.fim), alvo) >= 0.5]
    if not achou:
        return "não detectada", ""
    c = achou[0]
    visto = frase[c.inicio:c.fim]
    if c.classificacao != classe:
        return f"classe {c.classificacao}", visto
    if cid and str(c.id_canonico) != str(cid):
        return "id errado", visto
    return None, visto


def main(indice_path: str, detalhe: bool = False) -> bool:
    indice = json.loads(Path(indice_path).read_text(encoding="utf-8"))
    res = Resolvedor(indice)
    total, ok, falhas = Counter(), Counter(), defaultdict(list)

    def conta(grupo, cit, classe, cid):
        total[grupo] += 1
        motivo, visto = verificar(cit, res, classe, cid)
        if motivo is None:
            ok[grupo] += 1
        else:
            falhas[(grupo, motivo)].append((cit, visto))

    for chave, cid in sorted(indice["leis"].items()):
        apelido, artigo = chave.rsplit("|", 1)
        sigla, extenso = DIPLOMAS.get(apelido) or formas_genericas(apelido)
        for forma, cit in formas(artigo, sigla, extenso):
            conta(f"lei/{forma}", cit, "real", cid)
        # controle: artigo inexistente no mesmo diploma
        inexistente = str(int(artigo) + 1000)
        for forma, cit in formas(inexistente, sigla[:1], extenso[:1]):
            if forma in ("sigla", "extenso"):
                conta("lei/controle inventada", cit, "inventada", None)

    for chave, cid in sorted(indice["sumulas"].items()):
        partes = chave.split("|")
        if len(partes) == 3:                       # vinculante
            for cit in (f"Súmula Vinculante {partes[2]}",
                        f"Súmula Vinculante nº {partes[2]} do STF"):
                conta("súmula/vinculante", cit, "real", cid)
            continue
        trib, num = partes
        for forma, molde in FORMAS_SUMULA:
            conta(f"súmula/{forma}", molde.format(n=num, t=trib), "real", cid)
        conta("súmula/controle inventada", f"Súmula {int(num) + 900} do {trib}",
              "inventada", None)

    for cit in FORA_DA_BASE:
        total["controle fora da base"] += 1
        frase = f"Cumpre observar o disposto no {cit}, de aplicação cogente ao caso."
        if any(c.classificacao == "real" for c in extrair(frase, res)):
            falhas[("controle fora da base", "classificada como real")].append((cit, ""))
        else:
            ok["controle fora da base"] += 1

    print(f"{'grupo':30s} {'testadas':>8s} {'corretas':>8s}")
    for g in sorted(total):
        print(f"{g:30s} {total[g]:8d} {ok[g]:8d}   {100*ok[g]/total[g]:5.1f}%")
    t, o = sum(total.values()), sum(ok.values())
    print(f"{'TOTAL':30s} {t:8d} {o:8d}   {100*o/t:5.1f}%")
    for (g, motivo), casos in sorted(falhas.items()):
        print(f"\n[{g}] {motivo}: {len(casos)}")
        for cit, visto in casos[: (50 if detalhe else 6)]:
            print(f"    {cit!r}\n       -> {visto!r}")
    return o == t


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], "--detalhe" in sys.argv) else 1)
