# -*- coding: utf-8 -*-
"""
Modelo de ruído de OCR, derivado do que se observa nos documentos do nível 2.

Restrição da especificação do desafio: um dígito nunca é substituído por
outro dígito, pois isso mudaria a identidade da citação e transformaria uma
citação real ruidosa numa inventada. Trocas entre letra e dígito são
permitidas (e recuperáveis); trocas entre dígitos, nunca.

Confusões observadas nos documentos do desafio:
  e->c   (profcrido, dc, entcndimento, recentc, obscrvância)
  c->e   (eomporta, jurisprudêneia)
  i->l   (materla, dellto, Júnlor)
  m->rn  (assirn)
  a->ã   (tribunãis, Magãlhães, Kukinã)
  S<->5  (5úmula, DO5)      O<->0 (C0NTROVÉRSIA, 170076O)
  6->G   (6G.838)           9->g  (1.45g.779)
"""
import random
import re
import unicodedata

# --- confusões de letra (fora de identificadores) ---------------------------
CONFUSOES_TEXTO = [
    ("e", "c"), ("c", "e"), ("i", "l"), ("l", "i"), ("m", "rn"), ("rn", "m"),
    ("a", "ã"), ("o", "õ"), ("t", "l"), ("s", "5"), ("O", "0"), ("S", "5"),
    ("u", "ü"), ("n", "ñ"), ("d", "cl"),
]
# --- confusões dentro de identificadores: apenas letra↔dígito, nunca dígito↔dígito
CONFUSOES_ID = [("0", "O"), ("1", "l"), ("5", "S"), ("6", "G"), ("9", "g"),
                ("2", "Z"), ("8", "B"), ("7", "T")]

NBSP = " "


def _troca_uma(texto: str, rng: random.Random, pares) -> str:
    """Aplica uma confusão numa ocorrência sorteada."""
    rng.shuffle(pares := list(pares))
    for de, para in pares:
        pos = [m.start() for m in re.finditer(re.escape(de), texto)]
        if pos:
            i = rng.choice(pos)
            return texto[:i] + para + texto[i + len(de):]
    return texto


def ruido_texto(texto: str, rng: random.Random, taxa: float = 0.012) -> str:
    """Ruído em prosa: confusão de letra e, ocasionalmente, espaço duplo."""
    n = max(0, int(len(texto) * taxa))
    for _ in range(n):
        texto = _troca_uma(texto, rng, CONFUSOES_TEXTO)
    if rng.random() < 0.15:
        pos = [m.start() for m in re.finditer(r" ", texto)]
        if pos:
            i = rng.choice(pos)
            texto = texto[:i] + "  " + texto[i + 1:]
    return texto


def ruido_identificador(texto: str, rng: random.Random,
                        forca: float = 1.0) -> str:
    """Ruído dentro de uma citação, preservando a identidade do número: um
    dígito pode ser trocado por letra (recuperável), nunca por outro dígito.
    """
    op = []
    if rng.random() < 0.35 * forca:
        op.append("letra")
    if rng.random() < 0.40 * forca:
        op.append("quebra")
    if rng.random() < 0.35 * forca:
        op.append("espaco")
    if rng.random() < 0.30 * forca:
        op.append("pontuacao")
    if rng.random() < 0.25 * forca:
        op.append("nbsp")

    for o in op:
        if o == "letra":
            # só em dígito cercado por dígitos, ou no último dígito do bloco
            alvos = [m.start() for m in re.finditer(r"(?<=\d)\d(?=\d)", texto)]
            if alvos:
                i = rng.choice(alvos)
                mapa = {d: l for d, l in CONFUSOES_ID}
                if texto[i] in mapa:
                    texto = texto[:i] + mapa[texto[i]] + texto[i + 1:]
        elif o == "quebra":
            alvos = [m.start() for m in re.finditer(r"(?<=[\d.\-])(?=[\d.])", texto)]
            if alvos:
                i = rng.choice(alvos)
                texto = texto[:i] + "\n" + texto[i:]
        elif o == "espaco":
            alvos = [m.start() for m in re.finditer(r"(?<=[.\-])(?=\d)", texto)]
            if alvos:
                i = rng.choice(alvos)
                texto = texto[:i] + " " + texto[i:]
        elif o == "pontuacao":
            if rng.random() < 0.5:
                texto = texto.replace(".", "", 1)          # 1.741.784 -> 1741.784
            else:
                alvos = [m.start() for m in re.finditer(r"(?<=\d)-(?=\d)", texto)]
                if alvos:
                    i = rng.choice(alvos)
                    texto = texto[:i] + "--" + texto[i + 1:]
        elif o == "nbsp":
            alvos = [m.start() for m in re.finditer(r" ", texto)]
            if alvos:
                i = rng.choice(alvos)
                texto = texto[:i] + NBSP + texto[i + 1:]
    return texto


def digitos(s: str) -> str:
    return re.sub(r"\D", "", s)


def preserva_identidade(original: str, ruidoso: str) -> bool:
    """Verificação de segurança: a sequência de dígitos restante não pode ter se
    tornado outro número. Dígito removido (convertido em letra) é aceitável;
    dígito substituído por dígito, não.
    """
    a, b = digitos(original), digitos(ruidoso)
    if a == b:
        return True
    # b precisa ser subsequência de a (apenas remoções, nunca substituições)
    it = iter(a)
    return all(c in it for c in b)


def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")
