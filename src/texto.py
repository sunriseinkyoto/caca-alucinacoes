# -*- coding: utf-8 -*-
"""
Leitura de documentos com preservação exata de offsets.

A métrica oficial alinha spans por IoU sobre codepoints Unicode do texto
*exatamente como distribuído*. Três variações de codificação, invisíveis a olho
nu, deslocam esses codepoints:

  * quebras de linha CRLF (``\\r\\n``) — ``Path.read_text`` as converte para
    ``\\n`` sem aviso, e cada linha anterior a uma citação desloca o span em um
    codepoint. Num documento de 3.000 caracteres o deslocamento acumulado
    ultrapassa o comprimento das citações e o score cai de 1,1000 para 0,0946;
  * marca de ordem de bytes (BOM) no início do arquivo;
  * acentos em forma decomposta (NFD), em que ``ú`` ocupa dois codepoints e
    os padrões escritos em forma composta deixam de casar.

A estratégia é separar as duas representações. O extrator trabalha sobre um
texto normalizado (LF, sem BOM, NFC), e cada codepoint normalizado guarda o
intervalo que ocupa no texto original. Os spans produzidos são então mapeados
de volta. Para arquivos já em LF, sem BOM e em NFC — o caso da amostra de
desenvolvimento — o mapeamento é a identidade e a saída não se altera.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

BOM = "﻿"


@dataclass
class TextoFonte:
    bruto: str              # texto exatamente como distribuído
    proc: str               # texto normalizado, entregue ao extrator
    _ini: list[int]         # _ini[i]: início, no bruto, do codepoint i de proc
    _fim: list[int]         # _fim[i]: fim (exclusivo), no bruto, do codepoint i

    @property
    def identidade(self) -> bool:
        return self.bruto == self.proc

    def mapear(self, ini: int, fim: int) -> tuple[int, int]:
        """Converte um span [ini, fim) de ``proc`` para ``bruto``."""
        if self.identidade:
            return ini, fim
        if fim <= ini:
            p = self._ini[ini] if ini < len(self._ini) else len(self.bruto)
            return p, p
        return self._ini[ini], self._fim[fim - 1]


def normalizar(bruto: str) -> TextoFonte:
    proc: list[str] = []
    ini: list[int] = []
    fim: list[int] = []

    i, n = 0, len(bruto)
    if bruto.startswith(BOM):
        i = 1
    while i < n:
        c = bruto[i]
        # quebras de linha: CRLF e CR isolado viram LF
        if c == "\r":
            j = i + 2 if i + 1 < n and bruto[i + 1] == "\n" else i + 1
            proc.append("\n")
            ini.append(i)
            fim.append(j)
            i = j
            continue
        # segmento = caractere-base seguido de suas marcas combinantes;
        # a composição NFC nunca atravessa a fronteira de um segmento
        j = i + 1
        while j < n and unicodedata.combining(bruto[j]):
            j += 1
        seg = unicodedata.normalize("NFC", bruto[i:j])
        for ch in seg:
            proc.append(ch)
            ini.append(i)
            fim.append(j)
        i = j
    return TextoFonte(bruto, "".join(proc), ini, fim)


def ler_bruto(caminho: str | Path) -> str:
    """Decodifica o arquivo sem nenhuma tradução de quebra de linha."""
    return Path(caminho).read_bytes().decode("utf-8")


def carregar(caminho: str | Path) -> TextoFonte:
    return normalizar(ler_bruto(caminho))
