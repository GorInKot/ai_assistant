"""Загружает страницу нормативного акта с legalacts.ru и сохраняет очищенный
текст в knowledge_base/Законодательство_РФ/<подпапка>/<имя>.md

Использование:
    python scripts/fetch_legal_doc.py <url> <subfolder> <filename.md> "<title>" "<subtitle>"

Пример:
    python scripts/fetch_legal_doc.py \
      https://legalacts.ru/doc/postanovlenie-pravitelstva-rf-ot-21062010-n-468/ \
      Строительный_контроль \
      Постановление_468_от_2010-06-21.md \
      "Постановление Правительства РФ от 21.06.2010 № 468" \
      "О порядке проведения строительного контроля..."
"""
from __future__ import annotations
import sys, re, urllib.request, html.parser as hp
from pathlib import Path

KB_LEGAL = Path(__file__).resolve().parent.parent / "knowledge_base" / "Законодательство_РФ"


class _Extractor(hp.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "header", "footer", "nav", "aside", "form", "iframe"):
            self._skip += 1
            return
        if self._skip:
            return
        if tag in ("h1", "h2", "h3", "h4"):
            self.parts.append("\n\n## ")
        elif tag == "p":
            self.parts.append("\n\n")
        elif tag in ("br", "tr"):
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "header", "footer", "nav", "aside", "form", "iframe"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if self._skip == 0:
            self.parts.append(data)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept-Language": "ru,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        ct = resp.headers.get("Content-Type", "")

    # Кодировка из Content-Type
    enc = None
    m = re.search(r"charset=([\w-]+)", ct, re.IGNORECASE)
    if m:
        enc = m.group(1).lower()

    # Если не указана в header — смотрим meta charset в первых 4КБ
    if not enc:
        head = raw[:4096].decode("latin-1", errors="ignore")
        m = re.search(r'<meta[^>]+charset=["\']?([\w-]+)', head, re.IGNORECASE)
        if m:
            enc = m.group(1).lower()
        else:
            m = re.search(r'<meta[^>]+content=["\'][^"\']*charset=([\w-]+)', head, re.IGNORECASE)
            if m:
                enc = m.group(1).lower()

    if not enc:
        enc = "utf-8"

    # Алиасы
    if enc in ("windows-1251", "cp-1251", "win-1251"):
        enc = "cp1251"

    return raw.decode(enc, errors="replace")


def _extract_body(html_text: str) -> str:
    p = _Extractor()
    p.feed(html_text)
    text = "".join(p.parts)

    # Старт — первое появление одного из заголовков нормативного акта
    start_markers = [
        "ПРАВИТЕЛЬСТВО РОССИЙСКОЙ ФЕДЕРАЦИИ",
        "ФЕДЕРАЛЬНЫЙ ЗАКОН",
        "ПРИКАЗ",
        "ПОСТАНОВЛЕНИЕ",
        "УКАЗ",
    ]
    start = -1
    for m in start_markers:
        idx = text.find(m)
        if idx >= 0 and (start < 0 or idx < start):
            start = idx
    if start < 0:
        start = 0

    body = text[start:]

    # Конец — первый "сайдбарный" блок навигации legalacts.ru.
    # Берём минимальную позицию по всем известным маркерам.
    # `#` и `:` могут попадаться перед заголовком сайдбара (мы маркируем h2/h3 как "## ").
    # Используем "[#:\s]*" чтобы захватить эти префиксы.
    PFX = r"\n[#:\s]*"
    end_patterns = [
        re.compile(PFX + r"ст\.\s*\d+(?:\.\d+)?\s+(?:ТК|ГК|УПК|АПК|КоАП|БК|НК|УК)\s+РФ", re.UNICODE),
        re.compile(PFX + r"N\s*\d+(?:-ФЗ|-1)?\s+от\s+\d{1,2}\.\d{2}\.\d{4}", re.UNICODE),
        re.compile(PFX + r"ФЗ\s+о[а-яё ]", re.UNICODE),
        re.compile(PFX + r"ФЗ\s+об[а-яё ]", re.UNICODE),
        re.compile(PFX + r"Все\s+кодексы\s+РФ", re.UNICODE),
        re.compile(PFX + r"Кодексы\s+РФ\s*\n", re.UNICODE),
        re.compile(PFX + r"(?:Скачать\s+документ|Поделиться|Связанные\s+документы)", re.UNICODE),
        re.compile(PFX + r"Документы,\s+которые\s+также\s+Вас", re.UNICODE),
        re.compile(PFX + r"Популярные\s+(?:статьи|документы|материалы)", re.UNICODE),
        re.compile(PFX + r"Новые\s+документы", re.UNICODE),
        re.compile(PFX + r"Список\s+всех\s+кодексов", re.UNICODE),
        # rulaws.ru footer: список кодексов
        re.compile(PFX + r"Бюджетный\s+кодекс\s*\n", re.UNICODE),
        re.compile(PFX + r"Гражданский\s+кодекс\s+часть\s+1", re.UNICODE),
        re.compile(PFX + r"Арбитражный\s+процессуальный\s+кодекс\s+РФ", re.UNICODE),
    ]
    cut_at = len(body)
    for pat in end_patterns:
        m = pat.search(body)
        if m and m.start() < cut_at:
            cut_at = m.start()
    body = body[:cut_at]

    body = re.sub(r"\r\n", "\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" *\n *", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def main() -> int:
    if len(sys.argv) < 6:
        print(__doc__)
        return 2
    url, subfolder, filename, title, subtitle = sys.argv[1:6]
    html_text = _fetch(url)
    body = _extract_body(html_text)

    header = (
        f"# {title}\n\n"
        f"{subtitle}\n\n"
        f"- **Источник:** {url}\n"
        f"- **Применение в KB:** автоматически загружено для RAG-индекса.\n"
        f"- **Внимание:** актуальную редакцию проверять на pravo.gov.ru.\n\n"
        f"---\n\n"
    )

    target = KB_LEGAL / subfolder / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header + body + "\n", encoding="utf-8")
    print(f"saved: {target}  ({len(body)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
