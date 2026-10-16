"""Проверка автономности: ассистент работает, не выходя за пределы машины.

Запуск (Ollama должна быть запущена):  myenv/bin/python scripts/check_offline.py

Берёт настоящую конфигурацию (.env) и локальные модели. Все сетевые соединения процесса
перехватываются (app/netguard.py): всё, что не к этой машине, блокируется и попадает в отчёт.
Внутри перехвата проходится путь сотрудника: вопрос по базе знаний (поиск, эмбеддинги, ответ
модели), документ без пояснений, выжимка, сравнение двух версий. Документы выдуманные и
собираются здесь же. Отдельно — соединения процессов Ollama и сервера ассистента, который
слушает порт ASSISTANT_PORT (по умолчанию 8000), — снимок lsof.
Код выхода 1, если найдено хоть одно соединение вне машины.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docx import Document  # noqa: E402

from app import netguard  # noqa: E402

LETTER = [
    "ООО «Ромашка»",
    "Исх. № 12-345 от 03.09.2026",
    "Сообщаем, что поставка труб по договору № 15/2026 переносится на 20.10.2026.",
    "Стоимость поставки — 480 000 руб., условия оплаты не меняются.",
    "Просим подтвердить согласие с новым сроком до 25.09.2026.",
]
NOTE = ["Выезд на площадку 14 октября: ограждение установлено, бытовые помещения размещены, "
        "замечаний к подрядчику нет."]


def _docx(paragraphs: list[str]) -> bytes:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _external_peers(pids: list[str]) -> list[str]:
    """Установленные TCP-соединения процессов с адресами не этой машины (снимок lsof)."""
    peers = []
    for pid in pids:
        out = subprocess.run(["lsof", "-nP", "-a", "-iTCP", "-sTCP:ESTABLISHED", "-p", pid],
                             capture_output=True, text=True).stdout
        for line in out.splitlines()[1:]:
            name = line.split()[-2] if line.endswith(")") else line.split()[-1]
            remote = name.split("->")[-1] if "->" in name else ""
            host = remote.rsplit(":", 1)[0].strip("[]")
            if remote and host not in ("127.0.0.1", "::1", "localhost"):
                peers.append(f"pid {pid}: {name}")
    return peers


def _pids(*args: str) -> list[str]:
    return subprocess.run(["pgrep", *args], capture_output=True, text=True).stdout.split()


def main() -> int:
    from app.config import load_settings

    settings = load_settings()
    print("Конфигурация моделей:")
    for title, url, key in (("языковая модель", settings.llm_base_url, settings.llm_api_key),
                            ("эмбеддинги", settings.embeddings_base_url, settings.embeddings_api_key)):
        ok, why = netguard.model_endpoint_allowed(url, key)
        print(f"  {'✔' if ok else '✘'} {title}: {url or '—'} ({why})")

    steps = []
    with netguard.no_external_connections() as log:
        from app.services import chat_documents as cd
        from app.services.ask_service import process_ask
        from app.state import kb_index, kb_lock, llm_service

        def step(title, fn):
            started = time.time()
            try:
                detail = fn()
                steps.append((True, title, time.time() - started, detail))
            except Exception as exc:  # noqa: BLE001 — шаг упал, но проверку соединений продолжаем
                steps.append((False, title, time.time() - started, f"{type(exc).__name__}: {exc}"))

        def ask():
            with kb_lock:
                kb_index.build()
            user = SimpleNamespace(id=0)
            resp = process_ask("Кто согласует заявку на транспорт в ЕКТП?", "offline-check", user, None)
            return f"источников в ответе: {len(resp.sources)}"

        changed = [p.replace("480 000", "520 000") for p in LETTER]
        step("вопрос по базе знаний (поиск, эмбеддинги, ответ модели)", ask)
        step("документ без пояснений (вид оценивает модель)",
             lambda: cd.handle_chat_files("", [("записка.docx", _docx(NOTE))], llm_service).answer.splitlines()[0])
        step("выжимка документа", lambda: cd.handle_chat_files(
            "сделай выжимку", [("письмо.docx", _docx(LETTER))], llm_service).answer[:60] + "…")
        step("сравнение двух версий", lambda: cd.handle_chat_files(
            "сравни, старая — первая", [("письмо.docx", _docx(LETTER)), ("письмо_2.docx", _docx(changed))],
            llm_service).answer.splitlines()[1])

    print("\nПуть сотрудника (все соединения процесса перехвачены):")
    for ok, title, seconds, detail in steps:
        print(f"  {'✔' if ok else '✘'} {title} — {seconds:.1f} с; {detail}")
    local = sorted(set(log.local))
    print(f"\nСоединения процесса с этой машиной: {len(log.local)} ({', '.join(local) or '—'})")
    print(f"Попытки выйти за пределы машины: {len(log.blocked)}")
    for item in log.blocked:
        print(f"  ✘ {item}")

    port = os.getenv("ASSISTANT_PORT", "8000")
    ollama = _pids("-x", "ollama")
    server = subprocess.run(["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                            capture_output=True, text=True).stdout.split()
    peers = _external_peers(ollama + server)
    print(f"\nСоединения Ollama (pid {', '.join(ollama) or '—'}) и сервера ассистента "
          f"(pid {', '.join(server) or '—'}) вне машины: {len(peers)}")
    for item in peers:
        print(f"  ✘ {item}")

    failed = bool(log.blocked or peers or not all(ok for ok, *_ in steps))
    print("\nИтог:", "есть замечания" if failed else "внешних соединений нет, всё работает локально")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
