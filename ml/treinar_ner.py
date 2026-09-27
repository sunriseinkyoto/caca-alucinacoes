# -*- coding: utf-8 -*-
"""
Ajuste fino de um encoder (BERTimbau) para rotulação de spans de citação (BIO).

Um encoder foi preferido a um modelo generativo porque a métrica avalia spans
(IoU >= 0,5 em codepoints): o encoder produz offsets nativamente, pelo
``offset_mapping`` do tokenizador rápido, enquanto um modelo generativo
reescreve o texto e exige recuperar a posição a posteriori.

Executa em GPU de 6 GB com ``--lote 16`` em fp16, ou em CPU; ``--limite_min``
e ``--congelar`` permitem ajustar o treino a um orçamento de CPU. O laço de
treino é explícito, sem Trainer ou accelerate, para simplificar a verificação
de reprodutibilidade.

A validação mede F1 de span, e não acurácia por token: mais de 90% dos tokens
têm rótulo O, e um modelo que nunca marca nada já teria acurácia alta. Os
rótulos BIO são decodificados com a mesma função da inferência
(ml/inferir_ner.decodificar), e é esse F1 que seleciona o checkpoint.

Configuração usada nos resultados publicados: 3 épocas, lote 16, lr 3e-5,
max_len 256, semente 13 (ver ml/resultados/treino_ner.json).

Uso:
    python ml/treinar_ner.py treino.jsonl valid.jsonl saida_modelo/ \
           [--modelo neuralmind/bert-base-portuguese-cased] [--epocas 3] [--lote 16]

Teste do laço sem baixar pesos:
    python ml/treinar_ner.py t.jsonl v.jsonl /tmp/m --modelo __teste__ --epocas 1
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from inferir_ner import decodificar, iou  # noqa: E402

ROTULOS = ["O", "B-JUR", "I-JUR", "B-LEI", "I-LEI"]
R2I = {r: i for i, r in enumerate(ROTULOS)}


def carregar(caminho):
    return [json.loads(l) for l in Path(caminho).read_text(encoding="utf-8").splitlines() if l.strip()]


class DadosNER(Dataset):
    def __init__(self, exemplos, tok, max_len=512):
        self.ex, self.tok, self.max_len = exemplos, tok, max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ex = self.ex[i]
        cod = self.tok(ex["texto"], truncation=True, max_length=self.max_len,
                       return_offsets_mapping=True)
        offs = cod["offset_mapping"]
        rot = [-100] * len(offs)
        for j, (a, b) in enumerate(offs):
            if a == b:                       # token especial ([CLS], [SEP])
                continue
            rot[j] = R2I["O"]
        for ent in ex["entidades"]:
            primeiro = True
            for j, (a, b) in enumerate(offs):
                if a == b or rot[j] == -100:
                    continue
                # sobreposição entre o token e a entidade
                if a < ent["fim"] and b > ent["inicio"]:
                    pref = "B-" if primeiro else "I-"
                    rot[j] = R2I[pref + ent["rotulo"]]
                    primeiro = False
        return {"input_ids": cod["input_ids"],
                "attention_mask": cod["attention_mask"],
                "labels": rot}


def colar(lote, pad_id):
    n = max(len(x["input_ids"]) for x in lote)
    def pad(seq, val):
        return seq + [val] * (n - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in lote]),
        "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in lote]),
        "labels": torch.tensor([pad(x["labels"], -100) for x in lote]),
    }


def modelo_de_teste():
    """BERT mínimo com pesos aleatórios, apenas para verificar o laço de treino
    sem depender de download de pesos.
    """
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast
    import tempfile
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + \
            [chr(c) for c in range(32, 127)] + [f"##{chr(c)}" for c in range(97, 123)]
    d = Path(tempfile.mkdtemp())
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=False)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=64,
                     num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, num_labels=len(ROTULOS))
    return tok, BertForTokenClassification(cfg)


def avaliar_spans(modelo, exemplos, tok, disp, lote=8, max_len=512,
                  minimo=0.0):
    """Precisão, recall e F1 de span (IoU >= 0,5, pareamento guloso um para um),
    com a mesma decodificação BIO usada na inferência.
    """
    modelo.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(0, len(exemplos), lote):
            grupo = exemplos[i:i + lote]
            cod = tok([e["texto"] for e in grupo], truncation=True,
                      max_length=max_len, padding=True,
                      return_offsets_mapping=True, return_tensors="pt")
            offs = cod.pop("offset_mapping")
            prob = torch.softmax(
                modelo(**{k: v.to(disp) for k, v in cod.items()}).logits, -1)
            ids = prob.argmax(-1).cpu()
            top = prob.max(-1).values.cpu()
            for k, ex in enumerate(grupo):
                pred = [s for s in decodificar(offs[k].tolist(), ids[k].tolist(),
                                               top[k].tolist(), 0)
                        if s["score"] >= minimo]
                ouro = [dict(e) for e in ex["entidades"]]
                usados = set()
                for p in sorted(pred, key=lambda s: -s["score"]):
                    melhor, jm = None, 0.5
                    for j, g in enumerate(ouro):
                        if j in usados:
                            continue
                        v = iou(p, g)
                        if v >= jm:
                            melhor, jm = j, v
                    if melhor is None:
                        fp += 1
                    else:
                        usados.add(melhor)
                        tp += 1
                fn += len(ouro) - len(usados)
    modelo.train()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("treino")
    ap.add_argument("valid")
    ap.add_argument("saida")
    ap.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--congelar", type=int, default=0,
                    help="congela o embedding e as N camadas de baixo. Em CPU "
                         "corta ~1/3 do custo do backward; as camadas baixas do "
                         "BERT codificam forma de superficie, que e o que menos "
                         "precisa mudar para esta tarefa.")
    ap.add_argument("--limite_min", type=float, default=0.0,
                    help="orçamento de tempo em minutos; 0 = sem limite. Ao "
                         "esgotar, encerra a época corrente e salva.")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.modelo == "__teste__":
        tok, modelo = modelo_de_teste()
    else:
        from transformers import (AutoTokenizer,
                                  AutoModelForTokenClassification)
        tok = AutoTokenizer.from_pretrained(a.modelo, use_fast=True,
                                        do_lower_case=False)
        modelo = AutoModelForTokenClassification.from_pretrained(
            a.modelo, num_labels=len(ROTULOS),
            id2label={i: r for r, i in R2I.items()}, label2id=R2I)

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(disp).train()
    pad_id = tok.pad_token_id

    if a.congelar:
        base = modelo.base_model
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for camada in base.encoder.layer[:a.congelar]:
            for p in camada.parameters():
                p.requires_grad = False

    ex_tr, ex_va = carregar(a.treino), carregar(a.valid)
    dtr = DadosNER(ex_tr, tok, a.max_len)
    ctr = DataLoader(dtr, batch_size=a.lote, shuffle=True,
                     collate_fn=lambda b: colar(b, pad_id))

    otim = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad],
                             lr=a.lr, weight_decay=0.01)
    passos = max(1, len(ctr) * a.epocas)
    sched = torch.optim.lr_scheduler.OneCycleLR(otim, max_lr=a.lr,
                                                total_steps=passos,
                                                pct_start=0.1)
    escala = torch.amp.GradScaler(disp) if disp == "cuda" else None

    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"dispositivo={disp} | treino={len(dtr)} janelas | valid={len(ex_va)} "
          f"| {len(ctr)} passos/época | {treinaveis/1e6:.1f}M parâmetros "
          f"treináveis de {sum(p.numel() for p in modelo.parameters())/1e6:.1f}M",
          flush=True)
    t0 = time.time()
    melhor = -1.0
    Path(a.saida).mkdir(parents=True, exist_ok=True)
    for ep in range(a.epocas):
        soma = n = 0
        for lote in ctr:
            lote = {k: v.to(disp) for k, v in lote.items()}
            otim.zero_grad(set_to_none=True)
            if escala:
                with torch.autocast("cuda", dtype=torch.float16):
                    perda = modelo(**lote).loss
                escala.scale(perda).backward()
                escala.unscale_(otim)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                escala.step(otim)
                escala.update()
            else:
                perda = modelo(**lote).loss
                perda.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                otim.step()
            sched.step()
            soma += perda.item()
            n += 1
            if n % 25 == 0:
                print(f"  ep{ep+1} passo {n}/{len(ctr)}  perda={soma/n:.4f}  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
        p, r, f1 = avaliar_spans(modelo, ex_va, tok, disp, a.lote, a.max_len)
        print(f"época {ep+1}/{a.epocas}  perda={soma/max(n,1):.4f}  "
              f"span P={p:.4f} R={r:.4f} F1={f1:.4f}  "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)
        if f1 > melhor:                      # apenas o melhor checkpoint é mantido
            melhor = f1
            modelo.save_pretrained(a.saida)
            tok.save_pretrained(a.saida)
            print(f"  -> checkpoint salvo (F1={f1:.4f})", flush=True)
        if a.limite_min and (time.time() - t0) / 60 >= a.limite_min:
            print(f"orçamento de {a.limite_min} min esgotado; treino encerrado.")
            break

    if melhor < 0:                           # nenhuma época foi concluída
        modelo.save_pretrained(a.saida)
        tok.save_pretrained(a.saida)
    (Path(a.saida) / "treino.json").write_text(json.dumps(
        {"modelo_base": a.modelo, "epocas": a.epocas, "lote": a.lote,
         "lr": a.lr, "seed": a.seed, "max_len": a.max_len,
         "congelar": a.congelar, "dispositivo": disp,
         "janelas_treino": len(dtr), "melhor_f1_span_valid": round(melhor, 4),
         "minutos": round((time.time() - t0) / 60, 1)},
        ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"modelo salvo em {a.saida}  (melhor F1 de span na validação: "
          f"{melhor:.4f})")


if __name__ == "__main__":
    main()
