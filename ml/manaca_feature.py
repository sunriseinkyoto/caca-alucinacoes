# -*- coding: utf-8 -*-
"""
Perplexidade do Manacá-1B como variável de calibração (componente experimental,
não executado).

Hipótese: o texto ao redor de uma citação inventada teria estatística diferente
do texto ao redor de uma citação real, e a perplexidade de um modelo de
linguagem sobre essa vizinhança seria informativa para a confiança.

O componente não foi executado porque sua contribuição máxima ao score é nula
por construção: o bônus de calibração tem teto de 0,10, e o pipeline
determinístico, com confiança 1,0 e acurácia total, já o atinge. A
investigação da hipótese permanece possível — o ablation de ml/calibrar_ml.py
compara o modelo com e sem estas variáveis —, mas não altera a submissão.

Escolha do modelo: Manacá-1B (licença CC BY 4.0, pré-treino com parcela
jurídica), que em fp16 ocupa cerca de 3,4 GB.

Uso:
    python ml/manaca_feature.py <pasta_pred/> <pasta_txt> <saida.json> \
           [--modelo menezesbruno/manaca-1b-base] [--janela 220]
"""
import argparse
import json
from pathlib import Path

import torch


def perplexidade(modelo, tok, texto, disp):
    """Perplexidade do trecho sob o modelo (valores maiores indicam texto menos provável).
    """
    ids = tok(texto, return_tensors="pt", truncation=True,
              max_length=512).input_ids.to(disp)
    if ids.shape[1] < 2:
        return None
    with torch.no_grad():
        perda = modelo(ids, labels=ids).loss
    return float(torch.exp(perda))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("pasta_txt")
    ap.add_argument("saida")
    ap.add_argument("--modelo", default="menezesbruno/manaca-1b-base")
    ap.add_argument("--janela", type=int, default=220)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        a.modelo, torch_dtype=torch.float16 if disp == "cuda" else torch.float32)
    modelo.to(disp).eval()

    saida = {}
    for arq in sorted(Path(a.pred).glob("*.json")):
        texto = Path(a.pasta_txt, f"{arq.stem}.txt").read_text(encoding="utf-8")
        doc = json.loads(arq.read_text(encoding="utf-8"))
        linhas = []
        for c in doc["citacoes"]:
            i, f = c["inicio"], c["fim"]
            ctx = texto[max(0, i - a.janela):min(len(texto), f + a.janela)]
            # perplexidade do contexto com e sem a citação: a diferença
            # isola o quanto a própria citação destoa da vizinhança
            pp_com = perplexidade(modelo, tok, ctx, disp)
            sem = texto[max(0, i - a.janela):i] + texto[f:min(len(texto), f + a.janela)]
            pp_sem = perplexidade(modelo, tok, sem, disp)
            pp_cit = perplexidade(modelo, tok, texto[i:f], disp)
            linhas.append({"inicio": i, "fim": f, "pp_contexto": pp_com,
                           "pp_sem_citacao": pp_sem, "pp_citacao": pp_cit,
                           "delta": (None if (pp_com is None or pp_sem is None)
                                     else pp_com - pp_sem)})
        saida[arq.stem] = linhas
        print(f"  {arq.stem}: {len(linhas)} citacoes")

    Path(a.saida).write_text(json.dumps(saida, ensure_ascii=False),
                             encoding="utf-8")
    print(f"-> {a.saida}\n\nAgora rode o ablation: acrescente estas colunas em "
          f"ml/calibrar_ml.py (vetor()) e compare o Brier no conjunto de teste "
          f"com e sem elas.")


if __name__ == "__main__":
    main()
