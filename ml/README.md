# Componente de aprendizado de máquina

Este diretório reúne os experimentos com modelos. A submissão principal é
inteiramente determinística. O rotulador BERTimbau está disponível como
variante opcional (`python gerar_submissao.py --modelo-ner <pasta>`), com o
critério de fusão estrito descrito na seção 2. Todos os componentes foram
avaliados com a métrica oficial.

| componente | estado | resultado medido |
|---|---|---|
| corpus sintético | executado | revelou oito defeitos reais do extrator; é o conjunto de treino do rotulador e a base de `testes/teste_sintetico.py` |
| rotulador de spans BERTimbau | treinado; variante opcional | P = 0,9948 e R = 1,0000 nos 26 documentos reais, sem nenhum documento real no treino; com a fusão estrita, 1,1000 em qualquer fração de citações perdidas pelas regras, de 0% a 100%, no dev e em 5.717 citações sintéticas inéditas |
| calibração por regressão logística | executado | Brier de 0,0102 para 0,0051; +0,0005 no score |
| perplexidade do Manacá-1B | implementado, não executado | contribuição máxima nula: o bônus de calibração já está no teto |

Dependências adicionais: `pip install -r requirements-ml.txt`.

## 1. Corpus sintético

Os 26 documentos de desenvolvimento são montados a partir de um conjunto pequeno
de blocos: cabeçalho de autuação, títulos de seção, frases de enchimento e
frases-veículo (as que hospedam uma citação).

- `extrair_templates.py` separa esses blocos e os grava em `templates.json`.
- `gerar_sinteticos.py` recombina os blocos com registros reais da base
  canônica (número, tribunal, ano, relator e classe processual verdadeiros) e
  aplica o modelo de ruído de `ruido_ocr.py`.

A saída segue o formato do desafio e alimenta diretamente o pipeline e a métrica
oficial. Três propriedades tornam o gabarito sintético correto por construção:

1. entram apenas acórdãos cuja forma canônica resolve para um único registro;
2. todo número de citação `inventada` é verificado como ausente da base;
3. o ruído nunca troca um dígito por outro, apenas letras por dígitos.

Executar o extrator sobre corpora sintéticos revelou oito defeitos que também
existiam no caminho do conjunto de avaliação. Entre eles: a abreviação `Apel.`,
nomes de lei com OCR (`Trabaiho`, `Consurnidor`), conectivos corrompidos
(`dãs`, `cla`), `arl.` por `art.` e rótulos de autuação com acentos
corrompidos. Corrigi-los levou o score de desenvolvimento de 1,0894 para 1,1000.

**Limite.** O corpus herda o vocabulário dos 26 documentos. Classes processuais
que eles não citam ficam fora dele; por isso a bateria de testes inclui a
cobertura derivada da base canônica (`testes/teste_cobertura_base.py`).

## 2. Rotulador de spans (BERTimbau)

Rotulação de sequência em esquema BIO, com dois rótulos (`JUR` e `LEI`). O
modelo prevê apenas o span e o tipo, nunca a classe: não tem como saber se um
número existe na base. Os spans propostos passam pelo mesmo resolvedor
determinístico (`classificar_trecho`). O modelo propõe *onde*; o índice decide
*o quê*.

### Treino

| item | valor |
|---|---|
| modelo base | `neuralmind/bert-base-portuguese-cased` |
| treino | 520 documentos sintéticos (sementes 1001 e 1002, níveis 1 e 2), 4.298 citações, 1.736 janelas de 900 caracteres |
| validação | 60 documentos sintéticos (semente 2001), 495 citações, 203 janelas |
| documentos reais no treino | nenhum |
| hiperparâmetros | 3 épocas, lote 16, lr 3e-5 (OneCycle), AdamW, 256 subtokens, semente 13 |
| tempo | 70,5 minutos em CPU |
| critério de checkpoint | F1 de span (IoU ≥ 0,5) na validação |

A acurácia por token não serve como critério: mais de 90% dos tokens têm rótulo
`O`, e um modelo que nunca marca nada já teria acurácia alta. A validação
decodifica os rótulos BIO com a mesma função da inferência e mede F1 de span.

### Resultados de detecção

- F1 de span na validação sintética, medido por janela durante o treino:
  **0,9283**. A métrica por janela é mais baixa que a por documento, porque
  citações cortadas na borda de uma janela contam como erro; na inferência, as
  janelas se sobrepõem e são remontadas.
- Nos 26 documentos reais, que não participam do treino: 195 spans propostos,
  193 acima do limiar de 0,90, com **precisão 0,9948 e recall 1,0000** sobre as
  192 citações do gabarito. O único falso positivo acima do limiar é
  `'acórdão da Segunda Turma'`, uma expressão sem identificador que a política
  de anotação vigente não considera citação.
- Nos corpora sintéticos de validação (sementes 7 e 424242, 700 documentos,
  5.717 citações): precisão 0,9948 e 0,9936; recall 1,0000 nos dois.

**Correção.** Uma medição anterior dos documentos reais registrou precisão de
0,9689 e recall de 0,9740. Ela usou, por engano, o texto da versão v2 em três
documentos que a organização alterou na versão final. Em `gen_n1_003` e
`gen_n1_010` foi acrescentado "AgInt no "; em `gen_n1_006`, "ED no AgR no ".
Com isso, todos os spans desses documentos ficaram deslocados em 9 a 13
caracteres. Os números acima vêm da inferência refeita sobre o texto final
(`ml/resultados/ner_dev.json`).

### Fusão com o sistema de regras

`experimento_fusao.py` degrada artificialmente o recall das regras, simulando
templates desconhecidos, e mede o score com e sem os spans do rotulador. Com
100% de perda, as regras não detectam nada, e a detecção fica inteiramente a
cargo do rotulador; o resolvedor determinístico continua decidindo a classe e o
`id_canonico`.

Dois critérios de fusão:

- **original** (`--fusao-antiga`): descarta apenas spans com IoU ≥ 0,5 em
  relação a uma citação das regras, ou contidos nela;
- **estrito** (padrão): descarta spans que sobreponham, em qualquer medida,
  uma citação das regras, e spans sem nada verificável (nem identificador nem
  forma de citação vaga). No modo de confiança `maxima`, as citações
  acrescentadas recebem a confiança do resolvedor, como as demais.

Nos 26 documentos reais:

| citações perdidas pelas regras | só regras | fusão original | fusão estrita |
|---:|---:|---:|---:|
| 0% | 1,1000 | 1,0930 | **1,1000** |
| 5% | 1,0797 | 1,0930 | **1,1000** |
| 10% | 1,0450 | 1,0930 | **1,1000** |
| 20% | 0,9742 | 1,0930 | **1,1000** |
| 50% | 0,7474 | 1,0930 | **1,1000** |
| 100% | 0,0000 | 1,0930 | **1,1000** |

Nos corpora sintéticos de validação, que não participaram do treino nem do
desenho do critério estrito, a fusão estrita também dá 1,1000 em todas as
linhas, nas duas sementes. A fusão original perde apenas no falso positivo
descrito acima (dev) ou em spans equivalentes (semente 7: 1,0995).

Duas conclusões:

1. **O rotulador sozinho atinge o teto.** Com a detecção inteiramente a cargo
   do BERTimbau e a classificação a cargo do resolvedor, o score é 1,1000 no
   dev real e nos sintéticos. Treinado apenas em texto sintético, ele detecta
   todas as citações dos documentos reais.
2. **Ele não eleva o score quando as regras estão completas**, porque o teto já
   foi atingido. O valor dele é de redundância: se as regras falharem num
   formato de citação desconhecido, a fusão estrita recupera o que elas
   perderam, sem custo quando elas não falham.

A submissão principal permanece determinística, porque a variante com o
BERTimbau acrescenta dependências (torch, transformers) e os pesos treinados,
que não são versionados, à verificação de reprodutibilidade, sem ganho
esperado enquanto as regras estiverem completas. Ela está disponível como
`submission_bert.csv` (`gerar_submissao.py --modelo-ner`).

### Reprodução

```bash
python ml/gerar_sinteticos.py dados/desafio1_bracis.db templates.json indice.json ml/dados/tr1 --n 260 --nivel 1 --seed 1001
python ml/gerar_sinteticos.py dados/desafio1_bracis.db templates.json indice.json ml/dados/tr2 --n 260 --nivel 2 --seed 1002
python ml/gerar_sinteticos.py dados/desafio1_bracis.db templates.json indice.json ml/dados/va  --n 60  --nivel 2 --seed 2001
# concatenar tr1 e tr2 em ml/dados/tr (txt/ e goldenset.csv)

python ml/preparar_ner.py ml/dados/tr/txt ml/dados/tr/goldenset.csv ml/dados/tr.jsonl --janela 900 --passo 700
python ml/preparar_ner.py ml/dados/va/txt ml/dados/va/goldenset.csv ml/dados/va.jsonl --janela 900 --passo 700

python ml/treinar_ner.py ml/dados/tr.jsonl ml/dados/va.jsonl modelo_ner/ \
       --modelo <pasta do bert-base-portuguese-cased> \
       --epocas 3 --lote 16 --max_len 256 --lr 3e-5 --seed 13

python ml/inferir_ner.py modelo_ner/ dados/txt ml/resultados/ner_dev.json
python src/rodar.py dados/txt indice.json pred
python ml/experimento_fusao.py pred dados/goldenset.csv dados/txt indice.json dados/kaggle_metric.py \
       --ner ml/resultados/ner_dev.json             # estrito; --fusao-antiga para o original
```

Os spans do modelo nos corpora de validação estão em `ml/resultados/ner_s7.json`
e `ner_s424242.json`, e as tabelas, em `ml/resultados/fusao_estrita.json`.

Os pesos treinados (433 MB) não são versionados. As métricas do treino e os
spans produzidos estão em `ml/resultados/`.

## 3. Calibração aprendida

`calibrar_ml.py` substitui a tabela de confiança por tipo de evidência
(`CONF`, em `src/extrair.py`) por uma regressão logística sobre os sinais que o
extrator registra em cada citação: número de dígitos, presença de UF, uso de
leitura alternativa de OCR, camada do índice, número de candidatos, tokens de
classe absorvidos e largura do span. O protocolo treina no corpus sintético e
testa no gabarito real.

Medição feita numa versão anterior do pipeline, com score de desenvolvimento de
1,0894:

```
tabela fixa CONF : Brier = 0,01019   bônus = 0,09898   score = 1,0894
regressão        : Brier = 0,00512   bônus = 0,09949   score = 1,0899
```

O Brier caiu pela metade, com ganho de +0,0005. O limite é aritmético: o bônus
tem teto de 0,10, e o sistema já estava em 0,099. Com o pipeline atual, a
acurácia é total, e a confiança constante de 1,0 atinge o teto.

## 4. Perplexidade do Manacá-1B

`manaca_feature.py` calcula, para cada citação, a perplexidade da vizinhança sob
o Manacá-1B, com e sem a citação, como variável adicional da calibração. A
hipótese é que o texto ao redor de uma citação inventada tenha estatística
diferente.

O componente não foi executado, porque sua contribuição máxima ao score é nula.
O bônus de calibração tem teto de 0,10, e o pipeline já o atinge. A hipótese
pode ser investigada com o ablation de `calibrar_ml.py`, sem efeito sobre a
submissão. O Manacá-1B foi escolhido pela licença (CC BY 4.0) e pelo
pré-treino com parcela jurídica.

## 5. Arquivos

```
extrair_templates.py   documentos -> templates.json
ruido_ocr.py           modelo de ruído do nível 2, com garantia de identidade dos números
gerar_sinteticos.py    templates + base canônica + ruído -> corpus com gabarito
preparar_ner.py        spans -> JSONL em janelas deslizantes
treinar_ner.py         ajuste fino BIO do encoder
inferir_ner.py         modelo -> spans, com remontagem das janelas
fundir.py              fusão regras + rotulador (critério estrito); --relatorio para diagnóstico
ner_simulado.py        rotulador simulado (análise de sensibilidade da fusão)
experimento_fusao.py   medição da fusão (simulada ou com o modelo treinado)
calibrar_ml.py         calibração por regressão logística
manaca_feature.py      perplexidade do Manacá-1B como variável de calibração
resultados/            métricas do treino, spans do modelo e tabela de fusão
```
