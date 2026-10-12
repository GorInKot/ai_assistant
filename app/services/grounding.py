"""Проверка обоснованности ответа и цитаты из документов.

Зачем: небольшая локальная модель иногда добавляет в ответ то, чего нет в найденных
фрагментах базы знаний. Для корпоративного ассистента это главный риск, поэтому ответ
разбирается на утверждения, и каждое сверяется с контекстом, из которого его генерировали.
Подтверждённому утверждению подбирается дословная цитата («вот откуда это взято»).

Проверка лексическая и работает без вызова модели (мгновенно, детерминированно):
- слова сравниваются по основе (первые 5 букв — терпит падежи и формы слова);
- числа и даты из утверждения обязаны присутствовать в контексте — именно так обычно
  выглядят выдумки («срок — 14 дней»);
- опору утверждение должно найти в одном-двух фрагментах, а не «в сумме по всему контексту»,
  иначе набор общих слов предметной области «подтверждал» бы что угодно.

Ограничения (о них честно говорим пользователю): лексическая проверка не понимает смысл.
Утверждение с перепутанными фактами, но теми же словами, она пропустит; смысловой
пересказ другими словами может ошибочно получить метку «не подтверждено».
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

# Доля значимых слов утверждения, найденных в лучших фрагментах. Три зоны:
#   >= SUPPORT_THRESHOLD     — подтверждено;
#   ALARM_BELOW .. SUPPORT   — «частично» (модель перефразировала): без тревоги;
#   < ALARM_BELOW или выдуманное число — не подтверждено (тревога).
# Пороги подобраны по замеру на реальных ответах: внедрённые выдумки имеют покрытие <= 0.43,
# нормальные утверждения — в основном >= 0.6 (см. scripts/eval_grounding.py).
SUPPORT_THRESHOLD = 0.6
ALARM_BELOW = 0.45
MIN_CONTENT_WORDS = 3  # короче — заголовок/связка, не утверждение
TOP_CHUNKS_FOR_SUPPORT = 2
MAX_QUOTE_CHARS = 320
MAX_CITATIONS = 4
STEM_LEN = 5

_STOPWORDS = frozenset(
    """
    это этот эта эти этом этих этой тот того там тут здесь так также такой таких такие как что чтобы
    или либо если когда тогда потом затем после перед при для без над под про через между
    его её их ему ему они она оно мы вы ты не нет да бы же ли уже еще ещё лишь только очень
    быть был была были будет будут есть является являются может могут можно нужно необходимо
    должен должна должны должно следует либо который которая которое которые которых которым
    всех все всё весь вся свой свои своих свою свое своё один одна одно одни два две три
    например итак также кроме либо вот ниже выше далее ранее данный данная данные данных
    из от по на за до со во об обо про над под ко
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-–][A-Za-zА-Яа-яЁё0-9]+)*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?%?")
_MD_MARKERS_RE = re.compile(r"[*_`#>]+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [текст](путь) → текст
_SOURCE_NOTE_RE = re.compile(r"\((?:из|согласно|см\.?|в соответствии|по)\s[^)]*\)", re.IGNORECASE)
# Фразы о самом контексте и просьбы пользователю — не фактические утверждения.
_META_RE = re.compile(
    r"из контекста|из блока контекст|в контексте|в базе знаний|в базе нет|в инструкциях не указано|"
    r"не указано в|не приведен|не прописан|уточните|обратитесь|неоднозначност|примечание|"
    r"если (?:требуется|нужно|необходимо|пользователь|в запросе|других)|рекомендуется уточнить",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"(\[])")


STATUS_SUPPORTED = "supported"
STATUS_PARTIAL = "partial"
STATUS_UNSUPPORTED = "unsupported"


@dataclass
class Claim:
    text: str
    coverage: float  # доля значимых слов, найденных в лучших фрагментах (0..1)
    status: str  # supported | partial | unsupported
    missing_numbers: list[str] = field(default_factory=list)
    chunk_index: int | None = None  # индекс лучшего фрагмента в контексте
    quote: str | None = None

    @property
    def supported(self) -> bool:
        return self.status == STATUS_SUPPORTED


@dataclass
class Citation:
    chunk_index: int
    source: str  # относительный путь файла
    file_name: str
    page: int | None
    section: str | None
    quote: str
    claims: list[int]  # номера утверждений, которые эта цитата подтверждает

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "file_name": self.file_name,
            "page": self.page,
            "section": self.section,
            "quote": self.quote,
            "claims": self.claims,
        }


@dataclass
class GroundingResult:
    claims: list[Claim]
    citations: list[Citation]

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.status == STATUS_SUPPORTED)

    @property
    def partial(self) -> int:
        return sum(1 for c in self.claims if c.status == STATUS_PARTIAL)

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if c.status == STATUS_UNSUPPORTED]

    @property
    def ratio(self) -> float:
        """Доля утверждений без тревоги (подтверждённые и частично подтверждённые)."""
        return (self.total - len(self.unsupported)) / self.total if self.total else 1.0

    def summary(self) -> dict:
        """Компактная сводка для API/БД/интерфейса."""
        return {
            "supported": self.supported,
            "partial": self.partial,
            "total": self.total,
            "ratio": round(self.ratio, 3),
            "unsupported": [c.text[:300] for c in self.unsupported[:5]],
            "citations": [c.to_dict() for c in self.citations],
        }


# ------------------------------ токенизация ------------------------------


def _norm_token(token: str) -> str:
    return token.lower().replace("ё", "е")


def content_stems(text: str) -> set[str]:
    """Основы значимых слов текста (без стоп-слов, чисел и совсем коротких токенов)."""
    stems: set[str] = set()
    for raw in _WORD_RE.findall(text):
        token = _norm_token(raw)
        if token in _STOPWORDS or token.isdigit():
            continue
        # аббревиатуры (ЕКТП, ЦУС, ИС) значимы даже короткие
        if len(token) < 3 and not (raw.isupper() and len(raw) >= 2):
            continue
        stems.add(token[:STEM_LEN])
    return stems


def significant_numbers(text: str) -> list[str]:
    """Числа, которые обязаны быть в источнике: ≥2 цифр, проценты, десятичные.

    Одиночные цифры («шаг 3», «пункт 5») игнорируем: это нумерация, а не факт.
    """
    result = []
    for token in _NUMBER_RE.findall(text):
        digits = re.sub(r"\D", "", token)
        if len(digits) >= 2 or token.endswith("%") or "," in token or "." in token:
            result.append(token.replace(",", ".").rstrip("%"))
    return result


def _has_number(number: str, context_text: str) -> bool:
    normalized = context_text.replace(",", ".")
    return re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", normalized) is not None


# ------------------------------ утверждения ------------------------------


def split_claims(answer: str) -> list[str]:
    """Утверждения ответа: строки и предложения без разметки; слишком короткие отбрасываем."""
    claims: list[str] = []
    for line in answer.splitlines():
        line = _LIST_MARKER_RE.sub("", line)
        line = _LINK_RE.sub(r"\1", line)
        line = _SOURCE_NOTE_RE.sub("", line)
        line = _MD_MARKERS_RE.sub("", line).strip()
        if not line:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            # «…перечень:» — подводка к списку, «Если нужно уточнить…» — оговорка модели
            if sentence.endswith(":") or _META_RE.search(sentence):
                continue
            if len(content_stems(sentence)) >= MIN_CONTENT_WORDS:
                claims.append(sentence)
    return claims


def _readable(text: str) -> str:
    """Цитата без markdown-разметки источника (маркеры списков, `код`, **жирный**)."""
    text = _LIST_MARKER_RE.sub("", text)
    return " ".join(_MD_MARKERS_RE.sub("", text).split())


def _best_quote(claim_stems: set[str], chunk_text: str) -> str:
    """Фрагмент текста чанка (предложение/строка), ближе всего к утверждению."""
    pieces: list[str] = []
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces.extend(p.strip() for p in _SENTENCE_SPLIT_RE.split(line) if p.strip())
    if not pieces:
        return _readable(chunk_text)[:MAX_QUOTE_CHARS]

    def score(piece: str) -> tuple[int, int]:
        return (len(claim_stems & content_stems(piece)), -len(piece))

    best_index = max(range(len(pieces)), key=lambda i: score(pieces[i]))
    quote = pieces[best_index]
    # Очень короткий кусок (заголовок «Статусы:») дополняем следующей строкой.
    if len(quote) < 60 and best_index + 1 < len(pieces):
        quote = f"{quote} {pieces[best_index + 1]}"
    return _readable(quote)[:MAX_QUOTE_CHARS].rstrip()


def check_grounding(answer: str, chunks: Sequence) -> GroundingResult:
    """Сверить ответ с фрагментами контекста.

    `chunks` — объекты с полями `text`, `relative_path`, `page`, `section` (ChunkRecord),
    в том порядке, в каком они подавались модели.
    """
    chunk_stems = [content_stems(c.text) for c in chunks]
    context_text = "\n".join(c.text for c in chunks)

    claims: list[Claim] = []
    for text in split_claims(answer):
        stems = content_stems(text)
        overlaps = sorted(
            ((len(stems & cs), i) for i, cs in enumerate(chunk_stems)), reverse=True
        )
        best = overlaps[:TOP_CHUNKS_FOR_SUPPORT]
        covered: set[str] = set()
        for _, i in best:
            covered |= stems & chunk_stems[i]
        coverage = len(covered) / len(stems) if stems else 1.0

        missing = [n for n in significant_numbers(text) if not _has_number(n, context_text)]
        if missing or coverage < ALARM_BELOW:
            status = STATUS_UNSUPPORTED
        elif coverage >= SUPPORT_THRESHOLD:
            status = STATUS_SUPPORTED
        else:
            status = STATUS_PARTIAL

        best_index = best[0][1] if best and best[0][0] > 0 else None
        claims.append(
            Claim(
                text=text,
                coverage=round(coverage, 3),
                status=status,
                missing_numbers=missing,
                chunk_index=best_index,
                quote=_best_quote(stems, chunks[best_index].text)
                if (status == STATUS_SUPPORTED and best_index is not None)
                else None,
            )
        )

    return GroundingResult(claims=claims, citations=_build_citations(claims, chunks))


def _build_citations(claims: list[Claim], chunks: Sequence) -> list[Citation]:
    """Уникальные цитаты (файл + фрагмент), упорядоченные по числу подтверждённых ими утверждений."""
    by_key: dict[tuple[str, str], Citation] = {}
    for n, claim in enumerate(claims):
        if not (claim.supported and claim.quote and claim.chunk_index is not None):
            continue
        chunk = chunks[claim.chunk_index]
        key = (chunk.relative_path, claim.quote)
        citation = by_key.get(key)
        if citation is None:
            file_name = chunk.relative_path.rsplit("/", 1)[-1]
            by_key[key] = Citation(
                chunk_index=claim.chunk_index,
                source=chunk.relative_path,
                file_name=file_name,
                page=chunk.page,
                section=chunk.section,
                quote=claim.quote,
                claims=[n],
            )
        else:
            citation.claims.append(n)
    ordered = sorted(by_key.values(), key=lambda c: (-len(c.claims), c.chunk_index))
    return ordered[:MAX_CITATIONS]
