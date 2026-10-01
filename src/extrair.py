# -*- coding: utf-8 -*-
"""
Extração, classificação e resolução de citações jurídicas.

O extrator é determinístico e aplica quatro famílias de padrões, nesta ordem
de prioridade:

  A) Citações vagas -> ``incompleta``. Referências que identificam uma decisão
     sem identificador único (tribunal, ano e relator), casadas por expressões
     regulares com tolerância a erros de edição (módulo ``regex``), o que
     absorve o ruído de OCR do nível 2.
  B) Dispositivos legais -> ``art. N [inciso | parágrafo] do <diploma>``,
     resolvidos contra os dispositivos da base canônica.
  C) Súmulas e temas -> resolvidos contra as súmulas da base.
  D) Jurisprudência com número -> varredura tolerante a ruído. O número pode
     estar fragmentado por espaço, quebra de linha, hífen ou letra de OCR
     ("1.45g.779" = 1.459.779). A detecção ancora no número e expande à
     esquerda sobre a classe processual: o número é o componente que o ruído
     preserva; a classe, o que ele mais degrada.

Regra de classificação das citações com identificador:
  exatamente um registro na base         -> real (com ``id_canonico``)
  nenhum registro                        -> inventada
  mais de um registro com o mesmo número -> desempate pela classe processual
                                            (o gabarito aceita qualquer um
                                            dos registros duplicados)
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import product

import regex as re


# ---------------------------------------------------------------- utilitários

def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


UFS = ("AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR"
       "|SC|SP|SE|TO")
E = r"[\s ]"          # espaço, espaço não separável, quebra de linha

# Letras que o OCR troca por dígito. Só são convertidas quando cercadas
# por dígitos (ver variantes_digitos).
OCR_LETRA_DIGITO = {
    "o": "0", "O": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "i": "1", "|": "1",
    "z": "2", "Z": "2",
    "s": "5", "S": "5",
    "b": "6", "G": "6",
    "t": "7", "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}
SEPARADORES_INTERNOS = set(" . \n\t-–—")


@dataclass
class Citacao:
    inicio: int
    fim: int
    trecho: str
    tipo: str            # jurisprudencia | lei
    classificacao: str   # real | inventada | incompleta
    id_canonico: str | None
    confianca: float
    evidencia: str = ""
    sinais: dict | None = None

    def to_json(self) -> dict:
        return {
            "inicio": self.inicio, "fim": self.fim, "trecho": self.trecho,
            "tipo": self.tipo, "classificacao": self.classificacao,
            "resolucao": ({"fonte": "jusbrasil", "id_canonico": self.id_canonico}
                          if self.id_canonico else None),
            "confianca": round(self.confianca, 4),
            # campo auxiliar, ignorado pelo conversor e pela métrica; registra o
            # caminho de resolução para a calibração da confiança (src/calibrar.py)
            "evidencia": self.evidencia,
            "sinais": self.sinais or {},
        }


# ========================================================== A) CITAÇÕES VAGAS
TRIB = (r"(?:STF|STJ|STM|TSE|TST|Supremo\s+Tribunal\s+Federal"
        r"|Superior\s+Tribunal\s+de\s+Justi\S{1,4}a|Superior\s+Tribunal\s+Militar"
        r"|Tribunal\s+Superior\s+Eleitoral|Tribunal\s+Superior\s+do\s+Trabalho)")
# Os padrões vagos são compilados com re.I, o que faria [A-ZÁ...Ç] aceitar
# também minúsculas e permitiria que o nome do relator avançasse sobre a
# frase seguinte ("Francisco Falcão. Cumpre lembrar o art. 14 do CDC").
# (?-i:...) restaura a exigência de maiúscula apenas nesta letra; um
# lookahead negativo não serve, porque sob re.I também rejeitaria a
# maiúscula. Pelo mesmo motivo, um token de nome não pode terminar em
# ponto, salvo quando é inicial de uma letra.
_M1 = r"(?-i:[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ])"
_MAI = rf"(?:{_M1}\.|{_M1}[\wÀ-ÿ\-]*)"
NOME = (_MAI + r"(?:" + E + r"+(?:de|da|do|dos|das|e|" + _MAI + r")){0,6}")
# classes processuais que aparecem em citação vaga ("Rcl de 2021, Rel. Min. X")
CLASSE_VAGA = (r"(?:Reclama\S{1,4}o|Rcl|Recl\.?|Recurso\s+Especial"
               r"|Agravo\s+em\s+Recurso\s+Especial|REsp|AREsp|R\.?Esp\.?"
               r"|Recurso\s+em\s+Habeas\s+Corpus|Habeas\s+Corpus|RHC|HC"
               r"|Recurso\s+Extraordin\S{1,4}rio|RE|Apela\S{1,4}o|APL|RSE"
               r"|AgRg|AgInt|EDcl|ac\S{1,4}rd\S{1,4}o|julgado|precedente)")

# Cada padrão é envolvido em (?e)(?:...){e<=N}: tolera até N erros de edição.
_VAGAS_JURIS = [
    # "julgado do STF proferido em 2024 pela relatoria de Dias Toffoli"
    # "acórdão do STJ julgado em 2021 sob relatoria de Assusete Magalhães"
    (rf"(?:julgado|ac\S{{1,4}}rd\S{{1,4}}o|precedente){E}+d[oe]{E}+{TRIB}"
     rf"{E}+(?:proferido|julgado){E}+em{E}+\d{{4}}{E}+"
     rf"(?:pela|sob){E}+relatoria{E}+d[ec]{E}+{NOME}", 3),
    # "precedente do STF de 2026, da relatoria de CRISTIANO ZANIN"
    (rf"(?:julgado|ac\S{{1,4}}rd\S{{1,4}}o|precedente){E}+d[oe]{E}*{TRIB}"
     rf"{E}*,?{E}*de{E}+\d{{4}}{E}*,?{E}*d[ae]{E}+relatoria{E}+d[ec]{E}+{NOME}", 3),
    # "Reclamação do STF, de 2025, Rel. Min. CRISTIANO ZANIN"
    # "Recurso em Habeas Corpus do STJ, de 2019, Rel. Min. Sebastião Reis Júnlor"
    (rf"{CLASSE_VAGA}{E}*,?{E}*d[oe]{E}+{TRIB}{E}*,?{E}*de{E}+\d{{4}}{E}*,?"
     rf"{E}*Rel\.?{E}*(?:Min\.?)?{E}*{NOME}", 3),
    # "Rcl de 2021, Rel. Min. Rosa Weber" / "APL de 2023, Rel. Min. LEONARDO PUNTEL"
    (rf"(?<![\w])(?:{CLASSE_VAGA}){E}+de{E}+\d{{4}}{E}*,{E}*Rel\.{E}*"
     rf"(?:Min\.?{E}*)?{NOME}", 2),
]
# A política de anotação vigente não inclui citações vagas de lei ("artigo
# correspondente do Código de Processo Civil") nem apelos genéricos à
# jurisprudência ("reiterados precedentes do STJ"). É `incompleta` a
# referência que nomeia tribunal, ano e relator — uma decisão concreta sem
# identificador único. As 32 incompletas do gabarito final seguem essa forma.
# As demais expressões permanecem disponíveis no modo difusas (abaixo).
_VAGAS_LEI = []

# O casamento difuso sobre o documento inteiro é custoso (cerca de 0,7 s por
# documento para o padrão mais caro). Cada padrão tem, por isso, uma âncora:
# um literal curto que obrigatoriamente ocorre nele. O casamento difuso roda
# apenas na janela ao redor de cada âncora, com o mesmo resultado e cerca de
# 15 vezes menos tempo.
# Usa-se ENHANCEMATCH, e não BESTMATCH: com BESTMATCH, finditer omite
# ocorrências difusas anteriores a uma ocorrência exata ("relaloria" antes
# de "relatoria"), o que suprimiria a citação.
_ANCORAS = [
    (r"(?e)(?:relatoria){e<=2}", 130, 150),   # padrão 0
    (r"(?e)(?:relatoria){e<=2}", 130, 150),   # padrão 1
    (r"(?e)(?:Rel\.){e<=1}", 170, 150),       # padrão 2
    (r"(?e)(?:Rel\.){e<=1}", 170, 150),       # padrão 3
]
_ANCORA_LEI = []

VAGAS = ([(re.compile(rf"(?e)(?:{p}){{e<={n}}}", re.I | re.ENHANCEMATCH),
           re.compile(a, re.I | re.ENHANCEMATCH), esq, dir_, "jurisprudencia")
          for (p, n), (a, esq, dir_) in zip(_VAGAS_JURIS, _ANCORAS)]
         + [(re.compile(rf"(?e)(?:{p}){{e<={n}}}", re.I | re.ENHANCEMATCH),
             re.compile(a, re.I | re.ENHANCEMATCH), esq, dir_, "lei")
            for (p, n), (a, esq, dir_) in zip(_VAGAS_LEI, _ANCORA_LEI)])


# ---------------------------------------------------------- frases neutras
# Os documentos intercalam frases de enchimento, e algumas contêm uma
# expressão com a forma de citação vaga sem constituir citação. O que
# distingue os dois casos é a frase que hospeda a expressão, não a
# expressão em si. As frases neutras conhecidas (extraídas da amostra de
# desenvolvimento para templates.json) são usadas para descartar
# candidatos que caem dentro de uma delas.
FRASES_NEUTRAS: list[str] = []
TOL_NEUTRA = 0.16


def _norm_frase(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ",
                           sem_acento(s).lower()).split())


def carregar_neutras(caminho) -> int:
    """Carrega de templates.json as frases neutras que disparariam algum
    padrão vago; as demais nunca chegariam a produzir candidatos.
    """
    import json as _json
    from pathlib import Path as _Path
    dados = _json.loads(_Path(caminho).read_text(encoding="utf-8"))
    FRASES_NEUTRAS.clear()
    for frase in dados.get("neutras", []):
        if any(True for _ in achar_vagas(frase)):
            FRASES_NEUTRAS.append(_norm_frase(frase))
    return len(FRASES_NEUTRAS)


def _frase_em_volta(texto: str, a: int, b: int) -> str:
    ini = max(texto.rfind(". ", 0, a), texto.rfind("\n", 0, a)) + 1
    fim = texto.find(".", b)
    return texto[ini:fim + 1 if fim > 0 else len(texto)].strip()


def e_neutra(texto: str, a: int, b: int) -> bool:
    if not FRASES_NEUTRAS:
        return False
    alvo = _norm_frase(_frase_em_volta(texto, a, b))
    if len(alvo) < 25:
        return False
    return any(_distancia(alvo, n) <= TOL_NEUTRA for n in FRASES_NEUTRAS)


# O casamento difuso minimiza o número de erros e, por isso, tende a
# terminar cedo: em "Rel. Min. MARIA ELIZABETH GUIMARÃES TEIXEIRA ROCHA" ele
# pode parar em "MARIA E". Após o casamento, o span é estendido de forma
# determinística sobre o restante do nome. Um token de nome não pode
# terminar em ponto (salvo inicial de uma letra), para que a extensão não
# avance sobre a frase seguinte.
_NOME_TOK = r"(?:[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ]\.|[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ][\wÀ-ÿ\-]*)"
RE_RESTO_NOME = re.compile(rf"(?:[\s ]+(?:d[aeo]s?|e|{_NOME_TOK})){{0,8}}")


def _abreviacao_antes(texto: str, pos: int) -> bool:
    """O token imediatamente anterior a ``pos`` é uma abreviação curta ('Mln.')?

    Só nesse caso o ponto pode ser atravessado; em 'Falcão.' ele encerra a frase.
    """
    k = pos
    while k > 0 and (texto[k - 1].isalnum() or texto[k - 1] in "-'"):
        k -= 1
    return 0 < pos - k <= 4


def _estender_nome(texto: str, fim: int) -> int:
    # o casamento difuso pode terminar no meio de uma palavra ("MARIA E"):
    # primeiro completa-se a palavra, depois os nomes seguintes
    while fim < len(texto) and (texto[fim].isalnum() or texto[fim] in "-'"):
        fim += 1
    # "Rel. Mln. MARCELO ...": o casamento pode parar em "Mln", e o ponto
    # bloquearia a extensão. O ponto é atravessado quando fecha abreviação curta.
    if (fim < len(texto) and texto[fim] == "." and _abreviacao_antes(texto, fim)
            and fim + 1 < len(texto) and texto[fim + 1] in " \t \n"):
        fim += 1
    m = RE_RESTO_NOME.match(texto, fim)
    if not m:
        return fim
    novo = m.end()
    # não termina em conectivo isolado ("... de")
    while novo > fim and re.search(r"[\s ](?:d[aeo]s?|e)$", texto[fim:novo]):
        novo = fim + re.search(r"^(.*)[\s ](?:d[aeo]s?|e)$",
                               texto[fim:novo]).end(1)
    return novo



# ------------------------------------------------------- MODO DIFUSAS
# Referências que não apontam para fonte identificável ("jurisprudência
# pacífica desta Corte"). A política de anotação em vigor desde 04/09/2026
# as exclui, e o gabarito final da amostra de desenvolvimento não as
# contém; a política anterior as anotava como `incompleta`. O modo difusas
# reproduz a política anterior e é ativado com --difusas (src/rodar.py).
# Ver README, seção "Política de anotação".
_DIFUSAS = [
    # (padrão, orçamento de erros, âncora, janela à esquerda, janela à direita, tipo)
    (rf"jurisprud\S{{1,4}}ncia{E}+pac\S{{1,4}}fica{E}+desta{E}+Corte", 3,
     r"(?e)(?:pac\S{1,4}fica){e<=2}", 40, 60, "jurisprudencia"),
    (rf"orienta\S{{1,4}}o{E}+jurisprudencial{E}+da{E}+Corte{E}+Superior", 3,
     r"(?e)(?:jurisprudencial){e<=2}", 40, 70, "jurisprudencia"),
    (rf"precedente{E}+firmado{E}+em{E}+sede{E}+de{E}+recurso{E}+repetitivo", 3,
     r"(?e)(?:repetitivo){e<=2}", 90, 30, "jurisprudencia"),
    (rf"verbete{E}+sumular{E}+aplic\S{{1,4}}vel{E}+\S{{1,2}}{E}*esp\S{{1,4}}cie", 3,
     r"(?e)(?:verbete){e<=2}", 20, 70, "jurisprudencia"),
    (rf"entendi\S{{1,4}}ento{E}+sumulado{E}+sobre{E}+a{E}+mat\S{{1,4}}ria", 3,
     r"(?e)(?:sumulado){e<=2}", 50, 60, "jurisprudencia"),
    (rf"jurisprud\S{{1,4}}ncia{E}+consolidada{E}+dos{E}+tribunais{E}+superiores", 3,
     r"(?e)(?:consolidada){e<=2}", 40, 70, "jurisprudencia"),
    (rf"precedentes{E}+desta{E}+Casa{E}+em{E}+situa\S{{1,4}}es{E}+an\S{{1,4}}logas", 3,
     r"(?e)(?:precedentes{E}+desta){e<=2}", 20, 90, "jurisprudencia"),
    (rf"recente{E}+ac\S{{1,4}}rd\S{{1,4}}o{E}+da{E}+Segunda{E}+Turma", 3,
     r"(?e)(?:Segunda{E}+Turma){e<=2}", 60, 20, "jurisprudencia"),
    (rf"dispositivo{E}+constitucional{E}+invocado{E}+na{E}+origem", 3,
     r"(?e)(?:invocado){e<=2}", 50, 40, "lei"),
    (rf"lei{E}+que{E}+disciplina{E}+a{E}+prescri\S{{1,4}}o{E}+no{E}+caso", 3,
     r"(?e)(?:disciplina){e<=2}", 30, 60, "lei"),
    (rf"dispositivo{E}+legal{E}+de{E}+reg\S{{1,4}}ncia", 3,
     r"(?e)(?:reg\S{1,4}ncia){e<=2}", 60, 30, "lei"),
    (rf"legisla\S{{1,4}}o{E}+de{E}+reg\S{{1,4}}ncia{E}+da{E}+mat\S{{1,4}}ria", 3,
     r"(?e)(?:reg\S{1,4}ncia){e<=2}", 60, 40, "lei"),
    (rf"normas{E}+de{E}+reg\S{{1,4}}ncia{E}+da{E}+mat\S{{1,4}}ria", 3,
     r"(?e)(?:reg\S{1,4}ncia){e<=2}", 60, 40, "lei"),
    (rf"reiterados{E}+precedentes{E}+d[oe]{E}+{TRIB}", 3,
     r"(?e)(?:reiterados){e<=2}", 10, 90, "jurisprudencia"),
    (rf"artigo{E}+correspondente{E}+d[oe]{E}+C\S{{1,3}}digo{E}+de{E}+Processo{E}+Civil", 3,
     r"(?e)(?:correspondente){e<=2}", 20, 60, "lei"),
]
VAGAS_DIFUSAS = [(re.compile(rf"(?e)(?:{pat}){{e<={n}}}", re.I | re.ENHANCEMATCH),
                  re.compile(anc.replace("{E}", E), re.I | re.ENHANCEMATCH),
                  esq, dir_, tipo)
                 for pat, n, anc, esq, dir_, tipo in _DIFUSAS]

MODO_DIFUSAS = False


def definir_modo_difusas(ligado: bool) -> None:
    """Ativa as citações difusas da política de anotação anterior a 04/09/2026."""
    global MODO_DIFUSAS
    MODO_DIFUSAS = bool(ligado)


def achar_vagas(texto: str):
    """Gera ``(início, fim, tipo)`` para cada citação vaga do texto."""
    vistos = set()
    # só os padrões de VAGAS terminam no nome do relator; as expressões difusas
    # têm fim fixo e não passam pela extensão de nome
    familias = [(p, True) for p in VAGAS]
    if MODO_DIFUSAS:
        familias += [(p, False) for p in VAGAS_DIFUSAS]
    for (padrao, ancora, esq, dir_, tipo), termina_em_nome in familias:
        for ma in ancora.finditer(texto):
            a0 = max(0, ma.start() - esq)
            b0 = min(len(texto), ma.end() + dir_)
            janela = texto[a0:b0]
            for m in padrao.finditer(janela, concurrent=False):
                ini, fim = a0 + m.start(), a0 + m.end()
                if termina_em_nome:
                    fim = _estender_nome(texto, fim)
                if (ini, fim) not in vistos:
                    vistos.add((ini, fim))
                    yield ini, fim, tipo


# ================================================================ B) LEIS
# Nomes de diploma sofrem ruído de OCR ("Trabaiho", "Consurnidor",
# "Fedcral"). Em vez de regex difusa sobre o nome — lenta e sujeita a
# casamentos espúrios —, o nome é capturado de forma permissiva e comparado
# com os apelidos conhecidos por distância de edição (casar_lei).
ALIASES_LEI = {
    "CPC": ["codigo de processo civil", "cpc", "cpc/2015", "cpc/15",
            "novo codigo de processo civil", "codigo de processo civil de 2015"],
    "CPP": ["codigo de processo penal", "cpp"],
    "CODIGO PENAL MILITAR": ["codigo penal militar", "cpm"],
    "CDC": ["codigo de defesa do consumidor", "cdc"],
    "CODIGO ELEITORAL": ["codigo eleitoral"],
    "CODIGO CIVIL": ["codigo civil", "cc", "cc/2002", "codigo civil de 2002"],
    "CLT": ["consolidacao das leis do trabalho", "clt"],
    "CF": ["constituicao federal", "constituicao da republica", "cf", "cf/88",
           "cf/1988", "crfb", "crfb/88", "crfb/1988", "carta magna", "constituicao",
           "constituicao federal de 1988",
           "constituicao da republica federativa do brasil"],
    "LC 64/1990": ["lc 64/1990", "lei complementar 64/1990"],
    "LEI 13467/2017": [],
    "LEI 9504/1997": [],
}
# Número oficial de cada diploma. Gera as formas "Lei nº 8.078/1990",
# "Decreto-Lei nº 5.452/1943", "LC nº 64/90" etc. São dados legislativos
# estáveis, independentes do conjunto de avaliação.
NUMERO_OFICIAL = {
    "CPC": ("lei", "13105", "2015"),
    "CPP": ("decreto-lei", "3689", "1941"),
    "CODIGO PENAL MILITAR": ("decreto-lei", "1001", "1969"),
    "CDC": ("lei", "8078", "1990"),
    "CODIGO ELEITORAL": ("lei", "4737", "1965"),
    "CODIGO CIVIL": ("lei", "10406", "2002"),
    "CLT": ("decreto-lei", "5452", "1943"),
    "LC 64/1990": ("lei complementar", "64", "1990"),
    "LEI 13467/2017": ("lei", "13467", "2017"),
    "LEI 9504/1997": ("lei", "9504", "1997"),
}
# Diplomas frequentes na jurisprudência que não aparecem na amostra de
# desenvolvimento. A base da avaliação final é outra: um dispositivo desses
# diplomas pode constar dela (a citação é `real`) ou não (é `inventada`, como
# "art. 172 da Lei nº 9.504/1997" no gabarito da amostra). O apelido segue a
# chave que indexar.py atribui a diplomas sem nome próprio ("LEI 2848/1940"),
# de modo que o registro é encontrado qualquer que seja a forma citada.
DIPLOMAS_FREQUENTES = {
    ("decreto-lei", "2848", "1940"): ["codigo penal", "cp"],
    ("lei", "5172", "1966"): ["codigo tributario nacional", "ctn"],
    ("lei", "8069", "1990"): ["estatuto da crianca e do adolescente", "eca"],
    ("lei", "9503", "1997"): ["codigo de transito brasileiro", "ctb"],
    ("lei", "7210", "1984"): ["lei de execucao penal", "lep"],
    ("lei", "11340", "2006"): ["lei maria da penha"],
    ("lei", "11343", "2006"): ["lei de drogas"],
    ("lei", "10741", "2003"): ["estatuto do idoso", "estatuto da pessoa idosa"],
    ("lei", "8429", "1992"): ["lei de improbidade administrativa"],
    ("lei", "9099", "1995"): ["lei dos juizados especiais"],
    ("lei", "12016", "2009"): ["lei do mandado de seguranca"],
    ("decreto-lei", "4657", "1942"): [
        "lei de introducao as normas do direito brasileiro", "lindb"],
    ("lei", "7347", "1985"): ["lei da acao civil publica"],
    ("lei", "11101", "2005"): ["lei de falencias",
                               "lei de recuperacao judicial e falencias"],
    ("lei", "8245", "1991"): ["lei do inquilinato"],
    ("lei", "8906", "1994"): ["estatuto da advocacia", "estatuto da oab"],
    ("lei", "9605", "1998"): ["lei de crimes ambientais"],
    ("lei", "6830", "1980"): ["lei de execucao fiscal", "lef"],
    ("lei", "10826", "2003"): ["estatuto do desarmamento"],
    ("lei", "9096", "1995"): ["lei dos partidos politicos"],
    ("lei", "8072", "1990"): ["lei dos crimes hediondos"],
    ("lei", "9307", "1996"): ["lei de arbitragem"],
    ("decreto-lei", "1002", "1969"): ["codigo de processo penal militar", "cppm"],
    ("lei complementar", "35", "1979"): ["lei organica da magistratura nacional",
                                         "loman"],
    ("lei complementar", "135", "2010"): ["lei da ficha limpa"],
    ("lei", "13709", "2018"): ["lei geral de protecao de dados", "lgpd"],
    ("lei", "12965", "2014"): ["marco civil da internet"],
    ("lei", "13869", "2019"): ["lei de abuso de autoridade"],
    ("lei", "6015", "1973"): ["lei de registros publicos"],
    ("lei", "4717", "1965"): ["lei da acao popular"],
    ("lei", "13146", "2015"): ["estatuto da pessoa com deficiencia"],
    ("lei", "12651", "2012"): ["codigo florestal"],
}
for (_esp, _num, _ano), _nomes in DIPLOMAS_FREQUENTES.items():
    _ap = f"{'LC' if _esp == 'lei complementar' else 'LEI'} {_num}/{_ano}"
    ALIASES_LEI.setdefault(_ap, []).extend(_nomes)
    NUMERO_OFICIAL.setdefault(_ap, (_esp, _num, _ano))
for _ap, (_esp, _num, _ano) in NUMERO_OFICIAL.items():
    _especies = [_esp] + (["lc"] if _esp == "lei complementar" else [])
    for _e in _especies:
        for _a in (_ano, _ano[2:]):
            for _n in ("no ", ""):
                ALIASES_LEI[_ap].append(f"{_e.replace('-', ' ')} {_n}{_num}/{_a}")
TOL_LEI = 0.22          # distância de edição normalizada máxima aceita

# Sem (?i) nestes padrões: com case-insensitive, a alínea do inciso ("'g'")
# passaria a aceitar qualquer letra e consumiria o início do nome do
# diploma ("do CPC" -> lei='C'), e o algarismo romano [IVXLC]+ casaria
# "civil".
# A ordem das alternativas é significativa: se [a-zà-ÿ]{1,4} precedesse
# [Nn][ºo°.], casaria apenas o "n" de "nº" e a captura terminaria em
# "da Lei n" em vez de "da Lei nº 13.105/2015".
# A última alternativa cobre conectivos curtos ("da", "de", "das") e
# aceita os dígitos que o OCR troca por letra (5↔s, 0↔o, 1↔l), como em
# "Consolidação da5 Leis do Trabalho"; exige que a palavra comece por
# letra, para não absorver grupos numéricos.
TOKEN_LEI = (r"(?:[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ][\wÁ-ÿ.º°/-]*|[Nn][ºo°.]{1,2}"
             r"|\d{1,5}(?:[./-]\d{1,4})*|[a-zà-ÿ][a-zà-ÿ015]{0,3})")
INCISO = (r"(?:[IVXLC]{1,6}|§{E}*\d{1,3}[ºo°]?(?:-[A-Z])?"
          r"|['\"’][a-z]['\"’]|(?:inciso|par[aá]grafo|caput){E}*[\wº°]*)")
RE_LEI = re.compile(
    # "art" também sofre OCR: "arl. 477", "arligo", "artlgo", "artigõ"
    rf"\b[AaÃãÁáÀà]r[tlI1](?:[il1íì]g[oõóòô0]|\.)?{E}*(?P<num>\d{{1,4}}(?:\.\d{{3}})*){E}*[ºo°]?"
    rf"(?:{E}*,?{E}*{INCISO.replace('{E}', E)}{E}*,?)*"
    rf"{E}*(?:[Dd][aoe]{E}*)?"
    rf"(?P<lei>{TOKEN_LEI}(?:{E}+{TOKEN_LEI}){{0,6}})")


def _norm_lei(s: str) -> str:
    s = sem_acento(s).lower().replace("º", "o")
    return " ".join(re.sub(r"[^a-z0-9/]+", " ", s).split())


def _distancia(a: str, b: str) -> float:
    """Distância de Levenshtein normalizada pelo comprimento do alias."""
    if not b:
        return 1.0
    ant = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(ant[j] + 1, atual[j - 1] + 1,
                             ant[j - 1] + (ca != cb)))
        ant = atual
    return ant[-1] / len(b)


_OCR_DIGITO_LEI = str.maketrans({"o": "0", "l": "1", "i": "1", "s": "5"})


def _e_token_numerico(tok: str) -> bool:
    """'8.O78/199O' é número com ruído; 'C0rnplernentar' é palavra com ruído."""
    dig = sum(ch.isdigit() for ch in tok)
    return dig > 0 and dig >= sum(ch.isalpha() for ch in tok)


def _ler_numerico(tok: str) -> str:
    """Aplica a leitura de OCR só às partes numéricas: 'CRFB/88' tem uma parte
    de sigla e outra de ano, avaliadas separadamente."""
    return "/".join(p.lower().translate(_OCR_DIGITO_LEI) if _e_token_numerico(p) else p
                    for p in tok.split("/"))


def _digitos_lei(s: str) -> str:
    """Dígitos do número do diploma, recuperando letras de OCR dentro dos
    tokens numéricos ('8.O78/199O' -> '80781990')."""
    partes = [p for tok in s.split() for p in tok.split("/")]
    return "".join(re.sub(r"\D", "", _ler_numerico(p))
                   for p in partes if _e_token_numerico(p))


# Ano de cada diploma, para rejeitar "Código Civil de 1916" como CC/2002.
ANO_DIPLOMA = {ap: ano for ap, (_, _, ano) in NUMERO_OFICIAL.items()}
ANO_DIPLOMA["CF"] = "1988"
# Palavras que podem seguir o nome sem alterar o diploma ("Código Civil Brasileiro").
CONTINUACAO_NEUTRA = {"brasileiro", "brasileira", "vigente", "atual", "patrio",
                      "republicana", "cidada"}
_RE_ANO = re.compile(r"(?:18|19|20)\d{2}")
_SEGUE_DIPLOMA = {"do", "da", "de"}


def _nome_continua(toks: list[str], k: int) -> bool:
    """O nome do diploma prossegue além dos k primeiros tokens?

    'Constituição Estadual', 'Código de Processo Penal Militar' e
    'Constituição do Estado de São Paulo' começam como um diploma da base mas
    nomeiam outro. Aceitar o prefixo converteria uma citação de fora da base
    em ``real``. A continuação é detectada por uma palavra iniciada em
    maiúscula logo após o prefixo, ou por um conectivo seguido dela."""
    if k >= len(toks) or toks[k - 1][-1:] in ".,;:":
        return False                  # a pontuação encerra o nome ("da CLT. Nesse")
    seguinte = toks[k]
    if seguinte[:1].isupper() and _norm_lei(seguinte) not in CONTINUACAO_NEUTRA:
        return not _e_token_numerico(seguinte)
    if (_norm_lei(seguinte) in _SEGUE_DIPLOMA and k + 1 < len(toks)
            and toks[k + 1][:1].isupper()
            and not _RE_ANO.fullmatch(toks[k + 1])):
        return True
    return False


def _ano_divergente(apelido: str, toks: list[str], k: int) -> bool:
    """'Código Civil de 1916' não é o Código Civil de 2002."""
    ano = ANO_DIPLOMA.get(apelido)
    if not ano or k + 1 >= len(toks) or _norm_lei(toks[k]) not in _SEGUE_DIPLOMA:
        return False
    m = _RE_ANO.fullmatch(toks[k + 1].strip(".,;"))
    return bool(m) and m.group(0) != ano


def casar_lei(bruto: str):
    """Devolve (apelido, n_tokens_usados) do melhor alias, ou None.

    Testa prefixos do trecho capturado: 'Constituição Federal, é de rigor'
    casa 'constituicao federal' com 2 tokens e ignora o resto.

    O nome é comparado por distância de edição, para tolerar ruído de OCR, mas
    os DÍGITOS de um alias numérico ('lei no 8078/1990') precisam coincidir
    exatamente. A organização garante que o ruído nunca troca um dígito por
    outro; um dígito diferente é, portanto, outra lei — e aceitá-la por
    proximidade converteria uma lei inventada em real, o erro mais grave da
    métrica.
    """
    toks = bruto.split()
    # TOKEN_LEI aceita palavras minúsculas curtas, e o trecho capturado pode
    # começar pelo conectivo ("da Lei nº 13.105/2015"): descarta esse prefixo.
    while toks and _norm_lei(toks[0]) in ("da", "de", "do", "das", "dos", "e",
                                          "na", "no", "cla", "dae"):
        toks = toks[1:]
    melhor = None
    for k in range(1, len(toks) + 1):
        if _nome_continua(toks, k):
            continue
        cand = _norm_lei(" ".join(_ler_numerico(x) for x in toks[:k]))
        if not cand:
            continue
        dig_cand = _digitos_lei(" ".join(toks[:k]))
        for apelido, aliases in ALIASES_LEI.items():
            for alias in aliases:
                d = _distancia(cand, alias)
                if d > TOL_LEI:
                    continue
                dig_alias = re.sub(r"\D", "", alias)
                if dig_alias and dig_cand != dig_alias:
                    continue
                if _ano_divergente(apelido, toks, k):
                    continue
                # empate na distância: prevalece o nome mais longo
                # ("Constituição da República" em vez de "Constituição")
                if melhor is None or (round(d, 3), -k) < (round(melhor[0], 3), -melhor[2]):
                    melhor = (d, apelido, k)
    if melhor:
        return melhor[1], melhor[2]
    return _casar_lei_numerada(toks)


# Qualquer diploma identificado por espécie, número e ano ("Lei nº
# 12.345/2019", "Decreto-Lei 2.848/40", "LC nº 135/2010") é uma citação com
# identificador completo, conste ele ou não dos apelidos acima. A chave é a
# mesma que indexar.py atribui ao diploma: se a base tiver o dispositivo, a
# citação é `real`; senão, `inventada`. Os dígitos são lidos sem tolerância.
_RE_LEI_NUMERADA = re.compile(
    r"(?P<esp>lei complementar|lc|decreto lei|decreto|lei)(?: no| n)? "
    r"(?P<num>\d{1,3}(?: \d{3})*|\d{1,6})/(?P<ano>\d{4}|\d{2})")


def _casar_lei_numerada(toks: list[str]):
    for k in range(2, min(len(toks), 5) + 1):
        cand = _norm_lei(" ".join(_ler_numerico(x) for x in toks[:k]))
        m = _RE_LEI_NUMERADA.fullmatch(cand)
        if not m:
            continue
        ano = m.group("ano")
        if len(ano) == 2:
            ano = ("19" if int(ano) > 30 else "20") + ano
        num = m.group("num").replace(" ", "").lstrip("0")
        if not num:
            return None
        prefixo = "LC" if m.group("esp") in ("lei complementar", "lc") else "LEI"
        return f"{prefixo} {num}/{ano}", k
    return None


# ==================================================== C) SÚMULAS E TEMAS
# O sufixo "ula" (ou o ponto de "Súm.") é obrigatório: sem ele, um "s"
# trocado por "5" pelo OCR ("sem 5obressaltos") seria lido como súmula.
# As variantes cobrem as confusões de OCR descritas pela organização:
# 5↔S, m↔rn, l↔1.
RE_SUMULA = re.compile(
    rf"(?i)\b[S5][uúùüû]?(?:m|rn)(?:u[l1I]a|\.){E}*(?P<vinc>Vinculante{E}*)?"
    rf"(?:n?[ºo°.]{E}*)?(?P<num>\d{{1,4}})"
    rf"(?:{E}*d[oe]{E}*(?P<trib>STF|STJ|STM|TSE|TST))?")
# "Tema 2.680 da repercussão geral": a base não tem registros de tema, logo
# toda referência a tema com identificador completo é `inventada`.
RE_TEMA = re.compile(
    rf"(?i)\bTem\S{{0,2}}{E}*(?:n?[ºo°.]{E}*)?\d{{1,4}}(?:\.\d{{3}})?"
    rf"(?:{E}*d[ao]{E}*repercuss\S{{1,3}}o{E}*geral|{E}*d[oe]{E}*"
    rf"(?:STF|STJ|recursos?{E}*repetitivos?))")


# ================================================ D) NÚMEROS TOLERANTES A RUÍDO

def variantes_digitos(bruto: str, teto: int = 24) -> list[str]:
    """Converte o trecho bruto de um número em cadeias candidatas de dígitos.

    Letras de OCR cercadas por dígitos são convertidas em dígito
    ('1.45g.779' -> '1459779'). Havendo ambiguidade, devolve até ``teto``
    combinações.
    """
    opcoes: list[list[str]] = []
    chars = [c for c in bruto if not c.isspace() and c not in ".-–— "]
    for idx, c in enumerate(chars):
        if c.isdigit():
            opcoes.append([c])
        elif c in OCR_LETRA_DIGITO:
            viz_esq = idx > 0 and chars[idx - 1].isdigit()
            viz_dir = idx + 1 < len(chars) and chars[idx + 1].isdigit()
            if viz_esq and viz_dir:
                opcoes.append([OCR_LETRA_DIGITO[c]])
            elif viz_esq and idx == len(chars) - 1:
                opcoes.append([OCR_LETRA_DIGITO[c], ""])   # dígito ou ruído
            else:
                return []
        else:
            return []
    if not opcoes:
        return []
    total = 1
    for o in opcoes:
        total *= len(o)
        if total > teto:
            return ["".join(o[0] for o in opcoes)]
    return ["".join(p) for p in product(*opcoes)]


def varrer_numeros(texto: str, max_digitos: int = 22):
    """Gera ``(início, fim, bruto)`` para todo trecho com forma de número,
    inclusive quando fragmentado por espaço, quebra de linha, hífen ou ponto.
    """
    n = len(texto)
    i = 0
    while i < n:
        if not texto[i].isdigit():
            i += 1
            continue
        ini = i
        j = i
        ult_digito = i
        n_dig = 0
        while j < n:
            c = texto[j]
            if c.isdigit():
                n_dig += 1
                if n_dig > max_digitos:
                    break
                ult_digito = j
                j += 1
            elif c in SEPARADORES_INTERNOS or c in OCR_LETRA_DIGITO:
                # só continua se houver dígito nos próximos três caracteres
                k, achou = j, False
                while k < min(n, j + 4):
                    if texto[k].isdigit():
                        achou = True
                        break
                    if texto[k] not in SEPARADORES_INTERNOS and \
                       texto[k] not in OCR_LETRA_DIGITO:
                        break
                    k += 1
                if not achou:
                    break
                # Ponto seguido de espaço pode encerrar a frase. "1.741. 784"
                # continua (grupo de três dígitos); em "...0005. 0 acórdão" o
                # "0" é um "O" lido pelo OCR e inicia a frase seguinte.
                sep = texto[j:k]
                if "." in sep and any(ch.isspace() or ch == " " for ch in sep):
                    # conta dígitos e letras de OCR ("0B3", "2O11", "3Z5");
                    # do contrário, o grupo seguinte pareceria ter um único
                    # dígito e o número seria cortado
                    seguidos = 0
                    while (k + seguidos < n
                           and (texto[k + seguidos].isdigit()
                                or texto[k + seguidos] in OCR_LETRA_DIGITO)):
                        seguidos += 1
                    # "...0005. 0 acórdão": dígito isolado seguido de espaço
                    # é um "O" lido pelo OCR, início da frase seguinte.
                    # "...2020-\n.6.14.0022": dígito isolado seguido de ponto
                    # é continuação do número CNJ.
                    if seguidos < 2:
                        # verifica além do espaço: se o próximo caractere
                        # útil for dígito, o número continua ("1. 6\n66. 812");
                        # se for letra, começou outra frase
                        z = k + seguidos
                        while z < n and (texto[z].isspace() or texto[z] == " "
                                         or texto[z] in ".-–—"):
                            z += 1
                        if z >= n or not texto[z].isdigit():
                            break
                j = k
            else:
                break
        fim = ult_digito + 1
        # letra de OCR colada ao fim do número e seguida de fronteira:
        # em "Nº 170076O (SP)", o "O" é um zero
        if (fim < n and texto[fim] in OCR_LETRA_DIGITO
                and sum(c.isdigit() for c in texto[ini:fim]) >= 4
                and (fim + 1 >= n or not texto[fim + 1].isalnum())):
            fim += 1
        yield ini, fim, texto[ini:fim]
        i = max(fim, ini + 1)


RE_UF_SUFIXO = re.compile(rf"{E}*[-–—/(]?{E}*(?P<uf>{UFS})\b\)?")
RE_UF_EXTENSO = re.compile(
    r"\s*[-–—/(]?\s*(?P<uf>RIO GRANDE DO SUL|RIO GRANDE DO NORTE|RIO DE JANEIRO"
    r"|S[AÃ]O PAULO|MINAS GERAIS|DISTRITO FEDERAL|SANTA CATARINA|MATO GROSSO DO SUL"
    r"|MATO GROSSO|ESP[IÍ]RITO SANTO|BAHIA|PARAN[AÁ]|PERNAMBUCO|CEAR[AÁ]|GOI[AÁ]S"
    r"|PAR[AÁ]|PARA[IÍ]BA|PIAU[IÍ]|MARANH[AÃ]O|ALAGOAS|SERGIPE|AMAZONAS|AMAP[AÁ]"
    r"|ACRE|ROND[OÔ]NIA|RORAIMA|TOCANTINS)\b", re.I)
UF_POR_EXTENSO = {
    "riograndedosul": "RS", "riograndedonorte": "RN", "riodejaneiro": "RJ",
    "saopaulo": "SP", "minasgerais": "MG", "distritofederal": "DF",
    "santacatarina": "SC", "matogrossodosul": "MS", "matogrosso": "MT",
    "espiritosanto": "ES", "bahia": "BA", "parana": "PR", "pernambuco": "PE",
    "ceara": "CE", "goias": "GO", "para": "PA", "paraiba": "PB", "piaui": "PI",
    "maranhao": "MA", "alagoas": "AL", "sergipe": "SE", "amazonas": "AM",
    "amapa": "AP", "acre": "AC", "rondonia": "RO", "roraima": "RR",
    "tocantins": "TO",
}


# ---------------------------------------------- expansão da classe processual
TOKENS_CLASSE = set("""
recurso recursos especial especiais eleitoral extraordinario ordinario agravo
instrumento interno regimental embargos embargo declaracao divergencia habeas
corpus mandado seguranca reclamacao apelacao criminal civel sentido estrito acao
rescisoria revista peticao conflito competencia inquerito suspensao liminar
sentenca medida cautelar representacao processo autos tutela repetitivo
primeiro segundo terceiro quarto quinto sexto
resp aresp aginit agint agrg rhc rmc re rcl recl apl apel rse edcl ed eds rms respe
respei arespei arespel respel agresp agrespe hc ms ai are esp rec sum
sumula tema arr agarr rr
r-rp rp agreg ag int agr ar aresps embs edecl decl emb cc tst e n num numero
airr aiaresp agraresp ednos aginst agravos suspensao
no nos na nas de do da dos das em nao com contra
segundos terceiros decimo decimos referendo
embdecl embdiv edv ediv pext qo adi adpf adc ado aije rced
ro roe arespe agarespe pet sls
""".split())
# Conectivos: atravessados durante a expansão, mas nunca movem o início do
# span por conta própria — um conectivo só entra se houver classe à sua
# esquerda. Por isso "com" é seguro ("de acordo com o REsp" não arrasta "com").
PONTES = {"no", "nos", "na", "nas", "de", "do", "da", "dos", "das", "em", "e",
          "n", "num", "numero", "nao", "com", "contra"}

# Classes presentes na base cujas palavras, isoladas, são comuns na prosa
# jurídica ("penal", "ordem", "contas", "judicial"). Como tokens soltos, fariam
# o span avançar sobre o texto que antecede a citação; por isso só são aceitas
# como locução completa. Grafia normalizada: minúsculas, sem acento.
LOCUCOES_CLASSE = [
    "embargos infringentes e de nulidade", "infringentes e de nulidade",
    "questao de ordem", "pedido de extensao", "conflito de jurisdicao",
    "prestacao de contas", "lista triplice",
    "acao de investigacao judicial eleitoral", "investigacao judicial eleitoral",
    "recurso contra expedicao de diploma", "expedicao de diploma",
    "tutela cautelar antecedente", "cautelar antecedente",
    "correicao parcial militar", "correicao parcial",
    "acao penal", "acao direta de inconstitucionalidade",
    "arguicao de descumprimento de preceito fundamental",
    "cautelar inominada criminal", "cautelar inominada",
]
# Cada letra sujeita a confusão de OCR aceita também sua variante:
# m↔rn, o↔0, l↔1, i↔1, s↔5 (a lista descrita pela organização).
_OCR_LETRA = {"m": "(?:m|rn)", "o": "[o0]", "l": "[l1]", "i": "[i1l]", "s": "[s5]"}


def _padrao_ocr(palavra: str) -> str:
    return "".join(_OCR_LETRA.get(ch, re.escape(ch)) for ch in palavra)


RE_LOCUCAO_FINAL = re.compile(
    r"(?:^|[^a-z0-9])(" + "|".join(
        r"[\s\u00a0]+".join(map(_padrao_ocr, loc.split()))
        for loc in sorted(LOCUCOES_CLASSE, key=len, reverse=True)) + r")$")
_JANELA_LOCUCAO = 60


def _norm_token(t: str) -> str:
    """'nº' -> 'n'; 'H.C.' -> 'hc'; 'AG.REG' -> 'agreg'; 'R.Esp.' -> 'resp'."""
    return re.sub(r"[^a-z0-9\-/]", "", sem_acento(t).lower())


_DIGITO_LETRA_L = str.maketrans("015", "ols")
_DIGITO_LETRA_I = str.maketrans("015", "ois")


def _variantes_ocr(tok: str) -> list[str]:
    """Leituras alternativas de um token corrompido pelo OCR:
    'reclarnacao' -> 'reclamacao', 'agrav0' -> 'agravo', 'ape1acao' -> 'apelacao'."""
    base = tok.replace("rn", "m")
    return list(dict.fromkeys([base.translate(_DIGITO_LETRA_L),
                               base.translate(_DIGITO_LETRA_I),
                               tok.translate(_DIGITO_LETRA_L)]))


def _e_token_de_classe(base: str) -> bool:
    def aceito(s: str) -> bool:
        partes = [p for p in re.split(r"[-/]", s) if p]
        return s in TOKENS_CLASSE or bool(partes) and all(p in TOKENS_CLASSE for p in partes)
    # a forma original tem precedência: a leitura de OCR só é tentada se ela falhar
    return aceito(base) or any(aceito(v) for v in _variantes_ocr(base) if v != base)


def _e_ponte(base: str) -> bool:
    return base in PONTES or any(v in PONTES for v in _variantes_ocr(base))


def _norm_preservando(s: str) -> str:
    """Minúsculas sem acento, preservando o comprimento caractere a caractere
    (necessário para devolver posições no texto original)."""
    return "".join((sem_acento(ch)[:1] or ch).lower() for ch in s)


def _locucao_terminando_em(texto: str, j: int) -> int | None:
    """Se uma locução de classe termina exatamente em ``j``, devolve seu início."""
    a = max(0, j - _JANELA_LOCUCAO)
    m = RE_LOCUCAO_FINAL.search(_norm_preservando(texto[a:j]))
    return a + m.start(1) if m else None


def expandir_classe(texto: str, ini_num: int, limite: int = 220) -> int:
    """Estende o início da citação para a esquerda sobre a classe processual.

    Percorre tokens contíguos pertencentes ao vocabulário de classes (ou a uma
    locução de classe) e para no primeiro que não pertença. O limite cobre as
    classes aninhadas mais longas da base (136 caracteres, no STF).
    """
    i, inicio = ini_num, ini_num
    while True:
        j = i
        while j > 0 and texto[j - 1] in " \t \n":
            j -= 1
        if j == 0 or ini_num - j > limite:
            break
        loc = _locucao_terminando_em(texto, j)
        if loc is not None:
            inicio = i = loc
            continue
        k = j
        while k > 0 and (texto[k - 1].isalnum() or texto[k - 1] in ".-º°/'"):
            k -= 1
        base = _norm_token(texto[k:j])
        if not base:
            break
        if _e_token_de_classe(base):
            if not _e_ponte(base):
                inicio = k
            i = k
            continue
        break
    return inicio


# ----------------------------------------------------- linhas de autuação
RE_LINHA_AUTUACAO = re.compile(
    r"(?i)^\s*(?:processo|autos|proc\.|protocolo|refer\S+|interessado|assunto"
    r"|memorial|parecer|of\S+cio|impetrante|paciente|autoridade|elaborado"
    r"|recorrente|agravante|requerente|embargante|origem|n\S*mero|oab"
    r"|valor\s+da\s+causa)\b[^\n]{0,12}:")
RE_AUTUACAO_SIMPLES = re.compile(r"(?i)^\s*(?:processo|autos|proc\.)\s*n?[ºo°.]?\s")


# Rótulos de linha de autuação com tolerância a uma troca de OCR
# ("Rcferência", "lnteressado"). Só rótulos com seis letras ou mais: em
# rótulos curtos, um erro de edição já casaria palavras comuns.
RE_ROTULO_DIFUSO = re.compile(
    r"(?i)^\s*(?:(?:processo|protocolo|referencia|interessado|assunto|memorial"
    r"|parecer|oficio|impetrante|paciente|autoridade|elaborado|recorrente"
    r"|agravante|requerente|embargante|origem|numero){e<=1})\b[^\n]{0,12}:")
# Linha "Rótulo: valor" no cabeçalho do documento. O cabeçalho vai até o
# primeiro título de seção ("I — RELATÓRIO"); sem título, até 600 caracteres.
RE_ROTULO_GENERICO = re.compile(r"^\s*[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .º°/()-]{1,28}:\s")
RE_TITULO_SECAO = re.compile(r"\n[ \t]*[IVXLC]+[ \t]*[—–-]")


_RE_DIGITO_EM_PALAVRA = re.compile(r"(?<=[A-Za-z])[015]|[015](?=[A-Za-z])")
_DIGITO_PARA_LETRA = {"0": "o", "1": "l", "5": "s"}


def _letras_de_ocr(s: str) -> str:
    """'Auto5 nº 4837911-57' -> 'Autos nº 4837911-57': só dígitos colados a letras."""
    return _RE_DIGITO_EM_PALAVRA.sub(lambda m: _DIGITO_PARA_LETRA[m.group(0)], s)


def _fim_do_cabecalho(texto: str) -> int:
    m = RE_TITULO_SECAO.search(texto)
    return m.start() if m else min(len(texto), 600)


def _e_autuacao(texto: str, pos: int) -> bool:
    """O número está numa linha de autuação do próprio documento?

    O número dos autos, o protocolo e a inscrição na OAB, no cabeçalho, são
    distratores e não citações ('Processo nº ...', 'Referência: autos nº ...').
    Uma referência a outro processo no corpo do texto continua sendo citação.
    """
    ini = texto.rfind("\n", 0, pos) + 1
    fim = texto.find("\n", pos)
    linha = texto[ini:fim if fim != -1 else len(texto)]
    if len(linha) > 90:
        return False
    # o OCR mexe nos acentos ("Aütos", "Autõs") e troca letra por dígito dentro
    # das palavras ("Auto5", "Proces5o"): compara sem acento e com a mesma
    # leitura de OCR usada na expansão da classe processual
    linha = _letras_de_ocr(sem_acento(linha))
    if (RE_LINHA_AUTUACAO.match(linha) or RE_AUTUACAO_SIMPLES.match(linha)
            or RE_ROTULO_DIFUSO.match(linha)):
        return True
    return pos < _fim_do_cabecalho(texto) and bool(RE_ROTULO_GENERICO.match(linha))


RE_FOLHAS = re.compile(r"(?i)(?:fls?\.|folhas?|f\.)\s*$")


# =================================================================== RESOLUÇÃO
# Siglas -> palavras da classe processual. Usado para comparar a citação com
# o cabeçalho do registro quando a base tem mais de um registro com o mesmo
# número de processo.
SIGLAS = {
    "resp": "recurso especial", "aresp": "agravo recurso especial",
    "respe": "recurso especial eleitoral", "agrespe": "agravo recurso especial eleitoral",
    "rr": "recurso revista", "arr": "recurso revista agravo",
    "airr": "agravo instrumento recurso revista", "agarr": "agravo recurso revista agravo",
    "agrg": "agravo regimental", "agint": "agravo interno",
    "edcl": "embargos declaracao", "ed": "embargos declaracao",
    "eds": "embargos declaracao", "rcl": "reclamacao", "recl": "reclamacao",
    "apl": "apelacao", "rse": "recurso sentido estrito",
    "rhc": "recurso habeas corpus", "hc": "habeas corpus",
    "re": "recurso extraordinario", "rms": "recurso mandado seguranca",
    "ai": "agravo instrumento", "are": "agravo recurso extraordinario",
    "ag": "agravo", "int": "interno", "agr": "agravo", "esp": "especial",
    "rec": "recurso", "sum": "sumula", "ar": "acao rescisoria",
    "rp": "representacao", "agreg": "agravo regimental",
    "ro": "recurso ordinario", "roe": "recurso ordinario eleitoral",
    "arespe": "agravo recurso especial eleitoral", "pet": "peticao",
    "qo": "questao ordem", "pext": "pedido extensao",
    "embdecl": "embargos declaracao", "embdiv": "embargos divergencia",
    "edv": "embargos divergencia", "ediv": "embargos divergencia",
}
VAZIAS = {"no", "nos", "na", "nas", "de", "do", "da", "dos", "das", "em", "e",
          "n", "num", "numero", "com", "o", "a"}


def tokens_classe(trecho: str) -> "Counter[str]":
    """Normaliza a classe processual para comparação com o cabeçalho do registro.

    Devolve um multiconjunto, e não um conjunto, porque a repetição é
    informativa: 'AgARR' expande para 'agravo recurso revista agravo' (dois
    agravos), o que o distingue de 'AIRR' ('agravo instrumento recurso
    revista'), com um só. Como conjuntos, os dois empatariam.
    """
    bruto = re.sub(r"[^a-z0-9]+", " ", sem_acento(trecho).lower()).split()
    saida: Counter[str] = Counter()
    for t in bruto:
        if t.isdigit() or t in VAZIAS:
            continue
        saida.update(SIGLAS.get(t, t).split())
    return saida


class Resolvedor:
    def __init__(self, indice: dict):
        self.prim = indice["acordaos"]          # chave -> [id, ...] (número do cabeçalho)
        self.sec = indice.get("acordaos_sec", {})
        self.sumulas = indice["sumulas"]
        self.leis = indice["leis"]
        self.meta = indice.get("meta", {})

    def desempatar(self, candidatos: list[str], trecho_citacao: str) -> str:
        """Escolhe, entre registros com o mesmo número, o de classe mais próxima.

        A proximidade é a similaridade de Jaccard entre multiconjuntos de termos da
        classe processual. Persistindo o empate, prevalece o menor id, o que torna
        a escolha determinística.
        """
        alvo = tokens_classe(trecho_citacao)

        def nota(cid):
            cab = tokens_classe(self.meta.get(cid, {}).get("classe_cab", ""))
            inter = sum((alvo & cab).values())
            uniao = sum((alvo | cab).values())
            jaccard = inter / uniao if uniao else 0.0
            return (jaccard, -int(cid))   # empate -> menor id (determinístico)

        return max(candidatos, key=nota)

    @staticmethod
    def _chaves(digitos: str, uf: str | None):
        d = digitos.lstrip("0") or "0"
        return ([f"{d}|{uf}"] if uf else []) + [d]

    ultimo_diag: dict = {}

    def acordao(self, variantes: list[str], uf: str | None, trecho: str = ""):
        """Resolve um número de processo contra o índice.

        Consulta primeiro a camada primária (número próprio de cada registro, lido
        do cabeçalho) e, só depois, a secundária. Registra em ``ultimo_diag`` a
        camada usada e o número de candidatos, sinais usados na calibração.

        Returns:
            ``(id_canonico, classe)``, em que ``classe`` é ``real``, ``duplicata``
            ou ``inventada``.
        """
        self.ultimo_diag = {"camada": "nenhuma", "n_cand": 0}
        for nome_camada, indice in (("primaria", self.prim), ("secundaria", self.sec)):
            achados: list[str] = []
            for dig in variantes:
                for chave in self._chaves(dig, uf):
                    cands = indice.get(chave)
                    if cands:
                        achados = cands
                        break
                if achados:
                    break
            self.ultimo_diag = {"camada": nome_camada, "n_cand": len(achados)}
            if len(achados) == 1:
                return achados[0], "real"
            if len(achados) > 1:
                # Mais de um registro com o mesmo número (duplicata do acervo).
                # Desempata pela classe processual (desempatar). O gabarito
                # aceita qualquer um dos registros duplicados, e, pela matriz
                # da métrica, um id divergente custa apenas precisão, enquanto
                # uma troca de classe custa precisão e recall.
                return self.desempatar(achados, trecho), "duplicata"
        return None, "inventada"

    def sumula(self, numero: str, tribunal: str | None, vinculante: bool):
        if vinculante:
            c = self.sumulas.get(f"STF|SV|{numero}")
            return (c, "real") if c else (None, "inventada")
        if tribunal:
            c = self.sumulas.get(f"{tribunal}|{numero}")
            return (c, "real") if c else (None, "inventada")
        hits = {v for k, v in self.sumulas.items() if k.split("|")[-1] == numero}
        if len(hits) == 1:
            return hits.pop(), "real"
        return (None, "incompleta" if hits else "inventada")

    def lei(self, apelido: str, artigo: str):
        c = self.leis.get(f"{apelido}|{artigo}")
        return (c, "real") if c else (None, "inventada")


# ================================================================== CONFIANÇA
# O bônus é b = 0,10 · (1 − Brier), e Brier = 0 exige confiança 1,0 em todos
# os pares casados, todos corretos: o teto de 1,1000 só é alcançado com
# confiança máxima. O custo dessa escolha é pequeno. Com confiança
# constante c e acurácia a:
#     Brier = a(1−c)² + (1−a)c²
#     c = 1,0  ->  b = 0,10·a
#     c = a    ->  b = 0,10·(1 − a + a²)     (melhor valor constante)
#     diferença = 0,10·(1−a)²
# A 95% de acurácia a diferença é 0,00025; a 90%, 0,001; a 80%, 0,004.
# Por isso "maxima" é o padrão. "calibrada" usa a taxa de acerto medida
# por tipo de evidência e é preferível se a acurácia cair substancialmente.
MODO_CONFIANCA = "maxima"       # "maxima" | "calibrada"


def definir_modo_confianca(modo: str) -> None:
    global MODO_CONFIANCA
    if modo not in ("maxima", "calibrada"):
        raise ValueError("modo deve ser 'maxima' ou 'calibrada'")
    MODO_CONFIANCA = modo


def conf_de(evidencia: str) -> float:
    return 1.0 if MODO_CONFIANCA == "maxima" else CONF[evidencia]


# Confiança calibrada por tipo de evidência. Como o bônus usa o Brier score,
# a confiança ótima de cada grupo é a sua taxa observada de acerto.
CONF = {
    "vaga": 0.95,
    "lei": 0.96,
    "sumula": 0.95,
    "tema": 0.90,
    "cnj_uf": 0.97,     # número CNJ completo + UF: evidência mais forte
    "cnj": 0.93,
    "num_uf": 0.96,     # número curto + UF
    "num": 0.90,        # número curto sem UF: maior chance de ambiguidade
    "ruidoso": 0.88,    # resolvido por leitura alternativa de OCR
    "duplicata": 0.55,  # mais de um registro com o mesmo número
}


# ==================================================================== PIPELINE

def _dedup(cits: list[Citacao]) -> list[Citacao]:
    """Remove sobreposições com IoU >= 0,5, que a métrica rejeita; mantém o span
    mais longo.
    """
    cits = sorted(cits, key=lambda c: (c.inicio, -(c.fim - c.inicio)))
    saida: list[Citacao] = []
    for c in cits:
        if saida:
            a, b = saida[-1], c
            inter = max(0, min(a.fim, b.fim) - max(a.inicio, b.inicio))
            uniao = (a.fim - a.inicio) + (b.fim - b.inicio) - inter
            if uniao and inter / uniao >= 0.5:
                if (c.fim - c.inicio) > (a.fim - a.inicio):
                    saida[-1] = c
                continue
        saida.append(c)
    return saida


def extrair(texto: str, res: Resolvedor) -> list[Citacao]:
    cits: list[Citacao] = []
    ocupado: list[tuple[int, int]] = []

    def add(a, b, tipo, classe, cid, conf, evid="", sinais=None):
        while b > a and texto[b - 1] in " ,.;:\n -–—":
            b -= 1
        if b <= a or any(b > x and a < y for x, y in ocupado):
            return
        ocupado.append((a, b))
        cits.append(Citacao(a, b, texto[a:b], tipo, classe, cid, conf,
                            evid, sinais or {}))

    # --- A) vagas: são as mais longas e entram primeiro para não serem fatiadas
    for a, b, tipo in achar_vagas(texto):
        if not MODO_DIFUSAS and e_neutra(texto, a, b):
            continue                     # frase de enchimento, não é citação
        add(a, b, tipo, "incompleta", None, conf_de("vaga"), "vaga")

    # --- B) dispositivos legais
    for m in RE_LEI.finditer(texto):
        casado = casar_lei(m.group("lei"))
        if casado is None:
            continue
        apelido, n_toks = casado
        # o span termina no último token que compõe o nome do diploma
        ini_lei = m.start("lei")
        toks = list(re.finditer(r"\S+", m.group("lei")))
        fim_lei = ini_lei + toks[n_toks - 1].end()
        cid, classe = res.lei(apelido, m.group("num").replace(".", ""))
        add(m.start(), fim_lei, "lei", classe, cid, conf_de("lei"), "lei")

    # --- C) súmulas e temas
    for m in RE_SUMULA.finditer(texto):
        cid, classe = res.sumula(m.group("num"), m.group("trib"),
                                 bool(m.group("vinc")))
        add(m.start(), m.end(), "jurisprudencia", classe, cid, conf_de("sumula"), "sumula")
    for m in RE_TEMA.finditer(texto):
        add(m.start(), m.end(), "jurisprudencia", "inventada", None, conf_de("tema"), "tema")

    # --- D) jurisprudência com número
    for ini, fim, bruto in varrer_numeros(texto):
        n_dig = sum(c.isdigit() for c in bruto)
        if n_dig < 3:
            continue
        if _e_autuacao(texto, ini) or RE_FOLHAS.search(texto[max(0, ini - 12):ini]):
            continue

        uf, fim_uf = None, fim
        m_uf = RE_UF_SUFIXO.match(texto, fim)
        if m_uf:
            uf, fim_uf = m_uf.group("uf").upper(), m_uf.end()
        else:
            m_ex = RE_UF_EXTENSO.match(texto, fim)
            if m_ex:
                chave = re.sub(r"[^a-z]", "", sem_acento(m_ex.group("uf")).lower())
                uf, fim_uf = UF_POR_EXTENSO.get(chave), m_ex.end()

        a = expandir_classe(texto, ini)
        if a == ini:                     # número sem classe processual: não é citação
            continue

        variantes = variantes_digitos(bruto)
        if not variantes:
            continue
        cid, classe = res.acordao(variantes, uf, texto[a:fim_uf])
        forcado = classe == "duplicata"
        if forcado:
            classe = "real"

        cnj = n_dig >= 18
        ruidoso = len(variantes) > 1 or any(c in OCR_LETRA_DIGITO for c in bruto)
        if forcado:
            evid = "duplicata"
        elif ruidoso:
            evid = "ruidoso"
        elif cnj:
            evid = "cnj_uf" if uf else "cnj"
        else:
            evid = "num_uf" if uf else "num"
        sinais = {"n_digitos": n_dig, "tem_uf": int(bool(uf)),
                  "n_variantes": len(variantes),
                  "ruidoso": int(ruidoso), "cnj": int(cnj),
                  "n_tokens_classe": len(texto[a:ini].split()),
                  "largura": fim_uf - a, **res.ultimo_diag}
        add(a, fim_uf, "jurisprudencia", classe, cid, conf_de(evid), evid, sinais)

    return _dedup(cits)


# ==================================================== API para rotuladores externos
def classificar_trecho(trecho: str, res: Resolvedor):
    """Classifica um span já delimitado (por exemplo, proposto por um rotulador).

    Aplica a mesma sequência do pipeline principal — citação vaga, lei, súmula
    ou tema, número — e devolve ``incompleta`` quando nada se resolve.

    Returns:
        ``(tipo, classe, id_canonico, confianca, evidencia)``.
    """
    # 1º: o span inteiro é uma citação vaga? ("Rcl de 2021, Rel. Min. Rosa Weber")
    for a, b, tipo in achar_vagas(trecho):
        if (b - a) >= 0.6 * len(trecho.strip()):
            return tipo, "incompleta", None, conf_de("vaga"), "vaga"

    m = RE_LEI.search(trecho)
    if m:
        casado = casar_lei(m.group("lei"))
        if casado:
            cid, classe = res.lei(casado[0], m.group("num").replace(".", ""))
            return "lei", classe, cid, conf_de("lei"), "lei"

    m = RE_SUMULA.search(trecho)
    if m:
        cid, classe = res.sumula(m.group("num"), m.group("trib"),
                                 bool(m.group("vinc")))
        return "jurisprudencia", classe, cid, conf_de("sumula"), "sumula"
    if RE_TEMA.search(trecho):
        return "jurisprudencia", "inventada", None, conf_de("tema"), "tema"

    melhor = None
    for ini, fim, bruto in varrer_numeros(trecho):
        if sum(c.isdigit() for c in bruto) < 3:
            continue
        # "de 2021" é ano, não número de processo
        if (re.fullmatch(r"(19|20)\d{2}", bruto)
                and re.search(r"(?i)\b(de|em|ano)\s*$", trecho[:ini])):
            continue
        if melhor is None or len(bruto) > len(melhor[2]):
            melhor = (ini, fim, bruto)
    if melhor:
        ini, fim, bruto = melhor
        uf = None
        mu = RE_UF_SUFIXO.match(trecho, fim)
        if mu:
            uf = mu.group("uf").upper()
        variantes = variantes_digitos(bruto)
        if variantes:
            cid, classe = res.acordao(variantes, uf, trecho)
            if classe == "duplicata":
                return ("jurisprudencia", "real", cid,
                        conf_de("duplicata"), "duplicata")
            evid = "num_uf" if uf else "num"
            return "jurisprudencia", classe, cid, conf_de(evid), evid

    tipo = "lei" if re.search(r"(?i)\b(lei|artigo|dispositivo|c\S{1,3}digo)",
                              trecho) else "jurisprudencia"
    return tipo, "incompleta", None, conf_de("vaga"), "vaga"
