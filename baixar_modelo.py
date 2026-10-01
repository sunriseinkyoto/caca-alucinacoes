# -*- coding: utf-8 -*-
"""
Baixa os pesos do rotulador BERTimbau do Release v1.0, confere o SHA-256 e os
extrai em ``modelo_ner/``, na raiz do repositório.

É executado durante a construção da imagem Docker, antes de qualquer
execução: a avaliação em si roda sem acesso à rede. Também pode ser usado
fora do Docker:

    python baixar_modelo.py

Se o download não for possível (sem rede, por exemplo), o script avisa e
termina sem erro; ``run.sh`` gera então a submissão sem o rotulador. Um
arquivo baixado com SHA-256 diferente do publicado é sempre um erro.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = os.environ.get(
    "MODELO_URL",
    "https://github.com/sunriseinkyoto/caca-alucinacoes/releases/download/v1.0/modelo_ner.zip")
SHA256_ZIP = os.environ.get(
    "MODELO_SHA256", "b6be93ccacbe73f0fdc0e521521bba371f8424d36cfd08809cb3bcf81c15b2ab")
SHA256_PESOS = "d24fc1868312d1faeebbab5b4b2b337ff1ab7ccbf5284824eeb3ed70a3cc2dba"

RAIZ = Path(__file__).resolve().parent
DESTINO = RAIZ / "modelo_ner"


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> int:
    pesos = DESTINO / "model.safetensors"
    if pesos.exists() and sha256(pesos) == SHA256_PESOS:
        print(f"pesos já presentes e conferidos: {pesos}")
        return 0
    zip_ = RAIZ / "modelo_ner.zip"
    if not zip_.exists():
        print(f"baixando {URL}")
        try:
            urllib.request.urlretrieve(URL, zip_)
        except Exception as e:  # noqa: BLE001 - qualquer falha de rede
            zip_.unlink(missing_ok=True)
            print(f"AVISO: não foi possível baixar os pesos ({e}); "
                  "a submissão será gerada sem o rotulador BERTimbau")
            return 0
    obtido = sha256(zip_)
    if obtido != SHA256_ZIP:
        print(f"ERRO: SHA-256 de {zip_.name} = {obtido}, esperado {SHA256_ZIP}")
        return 1
    with zipfile.ZipFile(zip_) as z:
        z.extractall(RAIZ)
    zip_.unlink()
    if sha256(pesos) != SHA256_PESOS:
        print("ERRO: SHA-256 de model.safetensors diverge do publicado")
        return 1
    print(f"pesos conferidos e extraídos em {DESTINO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
