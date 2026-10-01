<p align="center">
  <img src="assets/banner.svg" alt="Caça-Alucinações — verificador determinístico de citações jurídicas" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/score-1%2C1000%20(teto)-2ea44f?style=for-the-badge" alt="score 1,1000">
  <a href="https://github.com/sunriseinkyoto/caca-alucinacoes/actions/workflows/docker.yml"><img src="https://github.com/sunriseinkyoto/caca-alucinacoes/actions/workflows/docker.yml/badge.svg" alt="docker"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/testes-1.959%20casos-2ea44f?style=for-the-badge" alt="1.959 casos de teste">
  <img src="https://img.shields.io/badge/GPU-n%C3%A3o%20necess%C3%A1ria-555555?style=for-the-badge" alt="GPU não necessária">
  <img src="https://img.shields.io/badge/licen%C3%A7a-MIT-blue?style=for-the-badge" alt="Licença MIT">
</p>

# Caça-Alucinações — verificador determinístico de citações jurídicas

Solução para o desafio **Caça-Alucinações** (BRACIS 2026 × Jusbrasil): localizar
todas as citações de jurisprudência e de lei em documentos jurídicos e classificar
cada uma como `real`, `inventada` ou `incompleta` contra a base canônica do desafio.

## Resultados

Todos os scores foram obtidos com a métrica oficial (`kaggle_metric.py`), sem
modificação. O teto da métrica é **1,1000**.

| conjunto | documentos | citações | score |
|---|---:|---:|---:|
| amostra de desenvolvimento (gabarito final) | 26 | 192 | **1,1000** |
| corpus sintético, semente 7 | 300 | 2.467 | **1,1000** |
| corpus sintético, semente 424242 | 400 | 3.250 | **1,1000** |
| corpus sintético, semente 90210 | 500 | 4.115 | **1,1000** |

Nos dois níveis: macro-F1 = 1,0000, τ = 0 (nenhuma citação inventada classificada
como real) e bônus de calibração de 0,1000.

Além disso, o sistema passa em testes derivados da própria base canônica, que não
dependem da amostra de desenvolvimento:

| teste | casos | resultado |
|---|---:|---:|
| acórdãos da base, com a classe por extenso, por sigla e com ruído de OCR | 1.705 | 100% |
| dispositivos e súmulas em todas as formas de superfície, com controles | 254 | 100% |
| arquivos com CRLF, BOM e acentos decompostos (NFD) | 5 variantes | 1,1000 |

O sistema de regras processa os 26 documentos em cerca de 6 s, em CPU, e depende
de um único pacote externo (`regex`). A execução completa, com o rotulador
BERTimbau, leva cerca de 1,5 min em CPU, sem GPU e sem rede.

## Execução na avaliação final

O ponto de entrada único é `run.sh`. Ele recebe o caminho do `.db` e da pasta
com os `.txt` e grava `submission.csv` no formato das submissões:

```bash
bash run.sh <base.db> <pasta_txt> [pasta_saida]     # saída padrão: ./saida
```

O ambiente está declarado no `Dockerfile`:

```bash
docker build -t caca-alucinacoes .
docker run --rm \
    -v /caminho/dos/dados:/dados:ro \
    -v /caminho/da/saida:/saida \
    caca-alucinacoes /dados/desafio1_bracis.db /dados/txt /saida
# resultado: /caminho/da/saida/submission.csv
```

O que `run.sh` executa, nesta ordem:

1. **índice da base informada.** `src/indexar.py` reconstrói o índice a partir do
   `.db` no formato original, sem nenhum dado da amostra de desenvolvimento. É
   o único enriquecimento da base, refeito a cada execução;
2. **extração por regras**, resolvida contra esse índice;
3. **rotulador BERTimbau**, com a fusão estrita (`ml/fundir.py`): acrescenta
   apenas citações em trechos onde as regras não encontraram nada e que tenham
   identificador verificável. A classe e o `id_canonico` vêm sempre do
   resolvedor determinístico;
4. **validação do contrato** e conversão com o conversor oficial
   (`oficial/json_to_submission.py`, sem modificação).

| regra | como é atendida |
|---|---|
| ambiente declarado | `Dockerfile`, com versões fixadas em `requirements-docker.txt` e `torch==2.8.0` |
| pesos em revisão fixa | `modelo_ner.zip` no [Release v1.0](https://github.com/sunriseinkyoto/caca-alucinacoes/releases/tag/v1.0), baixado e conferido por SHA-256 na construção da imagem (`baixar_modelo.py`) |
| sem rede na execução | a imagem roda com `HF_HUB_OFFLINE=1`; nenhuma etapa de `run.sh` acessa a rede |
| máquina limpa | sem caminhos absolutos nem passos manuais; o conversor oficial está em `oficial/` |
| hardware | CPU, cerca de 2 GB de RAM; a GPU é opcional (`--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124` e `--gpus all`) |
| determinismo | não há amostragem: as regras são determinísticas, e o rotulador roda em modo de inferência. Em CPU, o resultado é o mesmo em qualquer máquina |
| enriquecimento da base | gerado pelo próprio `run.sh` a partir do `.db` informado |

A cada push, a integração contínua (`.github/workflows/docker.yml`) constrói a
imagem do zero, executa `run.sh` com `--network none` sobre uma base e um
documento mínimos (`testes/fixture_docker.py`, sem material da organização) e
confere a saída, incluindo o uso do rotulador.

Se os pesos não estiverem disponíveis, `run.sh` avisa e entrega a variante só
com regras, que também atinge 1,1000 na amostra de desenvolvimento.
`CACA_SEM_BERT=1` força essa variante.

Na amostra de desenvolvimento, `bash run.sh` reproduz byte a byte o
`submission.csv` versionado (score 1,1000): o rotulador não acrescenta nenhuma
citação, porque as regras já encontram todas. O valor dele está em documentos
com formatos que as regras não reconhecem.

## Reprodução

Requisitos: Python 3.10 ou superior.

```bash
pip install -r requirements.txt
```

Copie o material distribuído pela organização para `dados/` (detalhes em
[`dados/LEIA-ME.md`](dados/LEIA-ME.md)):

```
dados/
├── txt/                    documentos de entrada
├── desafio1_bracis.db      base canônica
├── kaggle_metric.py        métrica oficial
└── goldenset.csv           gabarito (apenas para avaliação)
```

Gere a submissão:

```bash
python gerar_submissao.py                              # documentos em dados/txt
python gerar_submissao.py --txt <pasta_do_conjunto>    # outro conjunto
python gerar_submissao.py --gabarito dados/goldenset.csv   # gera e avalia
python gerar_submissao.py --modelo-ner <pasta_do_modelo>   # + variante com BERTimbau
```

O comando constrói o índice a partir da base (e confirma que coincide byte a byte
com o `indice.json` versionado), executa o extrator, valida o contrato de saída e
converte os JSONs com o conversor oficial. As saídas ficam em `saida/`:

```
saida/
├── submission.csv            submissão principal
├── submission_difusas.csv    variante para a política de anotação anterior
├── submission_bert.csv       variante opcional com o rotulador BERTimbau
├── json_principal/           um JSON por documento (contrato schema 1.2)
└── json_difusas/
```

A variante `submission_bert.csv` só é gerada com `--modelo-ner` e requer as
dependências de `requirements-ml.txt` e os pesos treinados do rotulador. Os
pesos (433 MB) não cabem no repositório e são distribuídos como anexo do
[Release v1.0](https://github.com/sunriseinkyoto/caca-alucinacoes/releases/tag/v1.0), no arquivo `modelo_ner.zip`:

```bash
pip install -r requirements-ml.txt
# extrair modelo_ner.zip na raiz do repositório, criando modelo_ner/
python gerar_submissao.py --modelo-ner modelo_ner
```

O SHA-256 de `modelo_ner/model.safetensors` está nas notas do Release e permite
confirmar que os pesos são os mesmos usados nos resultados publicados. Detalhes
do treino em [`ml/README.md`](ml/README.md).

Para executar a bateria completa de testes (cerca de 4 minutos):

```bash
python testes/executar_testes.py            # completa
python testes/executar_testes.py --rapido   # sem os corpora sintéticos
```

## Submissão

Na avaliação final, a organização executa `run.sh` sobre a nova base e os novos
documentos (ver [Execução na avaliação final](#execução-na-avaliação-final)).
Para a amostra de desenvolvimento, `saida/submission.csv` coincide byte a byte
com o `submission.csv` versionado na raiz do repositório.

### Política de anotação

Durante o desafio, a política de anotação mudou. A política vigente, comunicada
pela organização em 04/09/2026 e adotada no gabarito final, considera
`incompleta` a referência que identifica uma decisão sem identificador único
(tribunal, ano e relator). Ela não inclui expressões difusas como "jurisprudência
pacífica desta Corte". A política anterior as anotava como `incompleta`. A página
do desafio ainda descreve essa política: 225 citações, 65 delas incompletas.

As duas políticas estão implementadas:

| variante | política | amostra (gabarito final) | amostra (gabarito original) |
|---|---|---:|---:|
| `submission.csv` | vigente | **1,1000** | 0,8746 |
| `submission_difusas.csv` | anterior | 0,9776 | 0,9972 |

A variante difusas reproduz, texto a texto, as 65 incompletas do gabarito
original. Contra aquele gabarito, as duas variantes ficam abaixo do teto por
uma razão independente da política. A versão original da base não trazia
cabeçalho explícito em súmulas e dispositivos, que por isso não podem ser
indexados. Para a decisão, o que importa é a diferença entre as variantes em
cada gabarito: cerca de 0,12 a favor da política correta.

Apostar na política errada custa cerca de 0,12 nos dois sentidos, e a confiança
não reduz esse custo: uma predição sem par conta como falso positivo, qualquer
que seja sua confiança. A avaliação final segue a política vigente, e por isso
`run.sh` entrega a variante principal (acrescida do rotulador). A variante
difusas continua sendo gerada em `saida/variantes/`, apenas como referência.

### Variante com BERTimbau

A variante `submission_bert.csv` acrescenta à principal os spans do rotulador
BERTimbau em trechos onde as regras não encontraram nada. Com as regras
completas, ela coincide com a principal: na amostra de desenvolvimento, a saída
é idêntica byte a byte. O valor dela está no conjunto de avaliação, se ele
trouxer formatos de citação que as regras não reconhecem: nos testes, a fusão
mantém 1,1000 mesmo quando as regras perdem todas as citações. É a variante que
`run.sh` entrega como `submission.csv`. O log informa quantos spans o rotulador
acrescentou.

## Abordagem

A base canônica é um conjunto **fechado** de 1.014 registros (996 acórdãos,
5 súmulas e 13 dispositivos). Isso transforma a tarefa: em vez de recuperação
semântica, ela passa a ser *parsing*, normalização e casamento exato de
identificadores. O pipeline é determinístico:

1. **Índice em duas camadas** (`src/indexar.py`). A camada primária associa cada
   registro ao número do **próprio** processo, lido do cabeçalho com uma regra por
   tribunal. No TST, por exemplo, o número aparece na fórmula "Vistos, relatados
   e discutidos estes autos de … nº TST-…". A camada secundária guarda os demais
   números do cabeçalho. Uma busca no texto integral devolveria todo acórdão que
   *menciona* o número; só o cabeçalho identifica o processo.
2. **Varredura de números tolerante a OCR** (`src/extrair.py`). A detecção ancora
   no número — o componente que o ruído preserva, dada a garantia de que um
   dígito nunca é trocado por outro — e expande à esquerda sobre a classe
   processual. Letras de OCR cercadas por dígitos são convertidas
   (`1.45g.779` → 1459779), e números fragmentados por espaço, ponto ou quebra de
   linha são reconstruídos.
3. **Dispositivos e súmulas.** O nome do diploma é comparado por distância de
   edição, o que tolera `Consurnidor` e `Fedcral`, mas os dígitos do diploma
   exigem igualdade exata. Nomes que prosseguem além do apelido reconhecido
   ("Constituição Estadual") são tratados como outro diploma. As duas regras
   impedem que uma citação sem registro seja classificada como `real`.
4. **Citações vagas.** Padrões com tolerância a erros de edição, ancorados em
   literais curtos para manter o custo baixo.
5. **Distratores.** Linhas de autuação do cabeçalho (número dos autos, protocolo,
   OAB, folhas) são descartadas, inclusive com rótulos corrompidos por OCR
   ("Rcferência", "Auto5").
6. **Offsets exatos** (`src/texto.py`). O texto é processado em forma normalizada
   (LF, sem BOM, NFC), e os spans são mapeados de volta para os codepoints do
   arquivo como distribuído.
7. **Confiança 1,0.** O teto de 1,1000 exige Brier = 0. Com acurácia *a*, a perda
   máxima dessa escolha em relação à melhor confiança constante é 0,10·(1−a)²:
   0,00025 a 95% de acurácia. O modo `--confianca calibrada` está disponível.

## Garantias de robustez

O conjunto de avaliação é cego. As garantias de robustez vêm de testes que não
dependem da amostra de desenvolvimento:

- **Cobertura da base** (`testes/teste_cobertura_base.py`). Toda citação `real`
  do conjunto cego resolve, por construção, para um registro da base. Por isso, o
  teste reconstrói a citação de **cada acórdão** a partir do seu cabeçalho e a
  escreve por extenso, por sigla e com ruído de OCR. Esse teste revelou lacunas
  de vocabulário: "Recurso Extraordinário **com** Agravo", "Embargos Infringentes
  e de Nulidade", `RO`, `AREspe`. Revelou também a perda de detecção quando a
  classe sofre OCR ("Reclarnação": de 37% para 100% de cobertura nessa escrita).
- **Cobertura de normas** (`testes/teste_cobertura_normas.py`). Cobre os 13
  dispositivos e as súmulas em todas as formas ("art. 373, I, do CPC", "artigo
  373, inciso I, do Código de Processo Civil", "Lei nº 13.105/2015", "CRFB/88",
  "Súmula 83/STJ", "5úrnula"). Inclui controles: artigo inexistente deve resultar
  em `inventada`; diploma fora da base nunca pode resultar em `real`.
- **Base diferente.** O índice é sempre reconstruído a partir do `.db`
  informado. Diplomas que não aparecem na amostra são reconhecidos pelo número
  ("Lei nº 11.343/2006", "Decreto-Lei 2.848/40", "LC 135/2010") e, para 32
  diplomas frequentes, também pelo nome ("Código Penal", "CTN", "ECA", "Lei
  Maria da Penha"). Se a base tiver o dispositivo, a citação é `real`; senão,
  `inventada`. Isso foi verificado com uma cópia da base acrescida de
  dispositivos desses diplomas.
- **Codificação** (`testes/teste_codificacao.py`). Com quebras de linha CRLF, a
  leitura padrão de texto do Python deslocava todos os offsets, e o score caía
  para 0,0946 sem aviso.
- **Corpora sintéticos** (`testes/teste_sintetico.py`). Três sementes fixas, que
  não são usadas para ajustar o sistema.

## Estrutura do repositório

```
run.sh                    ponto de entrada único: base + documentos -> submission.csv
Dockerfile                ambiente de execução
requirements-docker.txt   versões fixadas da imagem
baixar_modelo.py          download e conferência dos pesos do Release v1.0
gerar_submissao.py        geração das variantes, validação e avaliação
oficial/                  conversor oficial, sem modificação
indice.json               índice da base canônica (reprodutível a partir da base)
templates.json            blocos de texto extraídos da amostra de desenvolvimento
submission.csv            submissão da amostra de desenvolvimento
src/
  indexar.py              construção do índice
  extrair.py              extração, classificação e resolução
  texto.py                leitura com preservação exata de offsets
  rodar.py                execução do pipeline sobre uma pasta
  validar.py              validação do contrato de saída
  avaliar_oficial.py      avaliação com a métrica oficial
  analisar_erros.py       relatório de erros por tipo
  alinhamento.py          alinhamento por IoU para diagnóstico
  calibrar.py             calibração da confiança por tipo de evidência
testes/                   bateria de testes (executar_testes.py) e dados mínimos do Docker
.github/workflows/        construção e execução da imagem a cada push
ml/                       rotulador BERTimbau (integra run.sh) e experimentos
assets/                   imagens do README
dados/                    material da organização (não versionado)
```

## Componente de aprendizado de máquina

O diretório `ml/` reúne os experimentos com modelos. O rotulador BERTimbau
integra a execução final (`run.sh`), pela fusão estrita; os demais componentes
são experimentais. Os resultados completos estão em [`ml/README.md`](ml/README.md):

| componente | estado | resultado medido |
|---|---|---|
| corpus sintético | executado | revelou oito defeitos reais; é a base do teste sintético |
| rotulador de spans BERTimbau | treinado (3 épocas, 70,5 min de CPU); integra `run.sh` | P = 0,9948 e R = 1,0000 nos 26 documentos reais, sem documento real no treino. Com o critério de fusão estrito, 1,1000 em qualquer fração de citações perdidas pelas regras, de 0% a 100%, no dev e em 5.717 citações sintéticas inéditas. Sozinho, com o resolvedor, também atinge o teto |
| calibração por regressão logística | executado | Brier de 0,0102 para 0,0051; +0,0005 no score |
| perplexidade do Manacá-1B | implementado, não executado | contribuição máxima nula: o bônus de calibração já está no teto |

## Conformidade com as regras

- Apenas ferramentas e modelos de código aberto. A execução não usa API externa
  nem acesso à rede; os pesos são baixados na construção da imagem.
- O conversor oficial (`json_to_submission.py`, incluído em `oficial/`) e a
  métrica oficial (`kaggle_metric.py`) são usados sem modificação.
- As saídas são determinísticas e reprodutíveis a partir de um clone limpo. O
  índice reconstruído coincide byte a byte com o versionado, e o `submission.csv`
  da amostra de desenvolvimento é reproduzido byte a byte pela bateria de testes.

## Licença

Distribuído sob a licença MIT (ver [`LICENSE`](LICENSE)). O material da
organização (documentos, base canônica, gabarito e métrica) não faz parte deste
repositório e segue os termos do desafio; a cópia do conversor oficial em
`oficial/` é incluída apenas para a execução em máquina limpa.
