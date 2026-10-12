"""Оценка проверки обоснованности ответов (app/services/grounding.py).

Два этапа:
  collect  — прогнать вопросы через реальный конвейер (поиск + локальная LLM) и сохранить
             ответы вместе с контекстом (медленно: десятки секунд на вопрос);
  measure  — по сохранённым данным посчитать, сколько корректных утверждений проверка
             подтверждает и сколько внедрённых выдумок ловит (быстро, без LLM).

Запуск:
  python scripts/eval_grounding.py collect --out eval/reports/grounding-cache.json
  python scripts/eval_grounding.py measure --cache eval/reports/grounding-cache.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Правдоподобные выдумки для предметной области: с числами и без (для измерения recall).
FABRICATIONS_WITH_NUMBERS = [
    "Срок рассмотрения заявки составляет 14 рабочих дней.",
    "Заявка автоматически аннулируется через 30 дней без движения.",
    "Размер штрафа за нарушение порядка составляет 25 000 рублей.",
    "Согласование занимает не более 3 часов с момента подачи.",
]
FABRICATIONS_NO_NUMBERS = [
    "Результат согласования автоматически направляется в бухгалтерию для начисления премии.",
    "Для завершения процесса требуется личная подпись генерального директора.",
    "Все данные дополнительно дублируются в архив службы безопасности.",
    "Руководитель юридического отдела обязан лично проверить каждый документ.",
]


def _questions() -> list[str]:
    questions = []
    for case in json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8")):
        if case.get("pre_steps") or case.get("expected_no_exact_match") is True:
            continue
        questions.append(case["question"])
    demo = (ROOT / "demo_queries.md").read_text(encoding="utf-8")
    questions += [m.group(1).strip() for m in re.finditer(r"^\d+\.\s+(.+)$", demo, re.MULTILINE)]
    seen, unique = set(), []
    for q in questions:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique


def collect(out: Path, limit: int) -> None:
    import app.state as state
    from app.state import kb_index, llm_service, settings

    kb_index.build()
    rows = []
    for question in _questions()[:limit]:
        results = kb_index.hybrid_retrieve(question, top_k=max(50, settings.rerank_candidates * 2))
        reranked = llm_service.rerank_results(question, results, top_n=settings.rerank_top_n)
        context = kb_index.select_context_results(question, reranked, max_chunks=5)
        if not kb_index.is_context_strong(context):
            print(f"[пропуск: слабый контекст] {question}")
            continue
        intent = kb_index.classify_query_intent(question)
        answer = llm_service.generate_answer(question, context, intent=intent)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "chunks": [
                    {
                        "text": r.chunk.text,
                        "relative_path": r.chunk.relative_path,
                        "page": r.chunk.page,
                        "section": r.chunk.section,
                    }
                    for r in context
                ],
            }
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{len(rows)}] {question} → {len(answer)} симв.")
    print(f"Сохранено {len(rows)} ответов → {out}")


class _Chunk:
    def __init__(self, d: dict) -> None:
        self.text, self.relative_path = d["text"], d["relative_path"]
        self.page, self.section = d.get("page"), d.get("section")


def measure(cache: Path, verbose: bool) -> None:
    from app.llm import FALLBACK_RU
    from app.services.grounding import check_grounding

    rows = json.loads(cache.read_text(encoding="utf-8"))
    claims_total = claims_supported = claims_alarm = 0
    answers_all_supported = answers = 0
    injected = {"numbers": [0, 0], "plain": [0, 0]}  # [поймано, внедрено]

    for row in rows:
        chunks = [_Chunk(c) for c in row["chunks"]]
        answer = row["answer"]
        if FALLBACK_RU.lower() in answer.lower():
            continue
        result = check_grounding(answer, chunks)
        if not result.total:
            continue
        answers += 1
        claims_total += result.total
        claims_supported += result.supported
        claims_alarm += len(result.unsupported)
        answers_all_supported += not result.unsupported
        if verbose:
            print(f"\n## {row['question']}  [подтв. {result.supported}, частично {result.partial}, тревог {len(result.unsupported)} из {result.total}]")
            for c in result.unsupported:
                print(f"   ✗ ({c.coverage:.2f}{', числа ' + ','.join(c.missing_numbers) if c.missing_numbers else ''}) {c.text[:140]}")

        for kind, bank in (("numbers", FABRICATIONS_WITH_NUMBERS), ("plain", FABRICATIONS_NO_NUMBERS)):
            for fake in bank:
                fabricated = check_grounding(answer + "\n\n" + fake, chunks)
                last = fabricated.claims[-1] if fabricated.claims else None
                injected[kind][1] += 1
                if last is not None and last.text.startswith(fake[:20]) and last.status == 'unsupported':
                    injected[kind][0] += 1

    print("\n=== Итоги ===")
    print(f"Ответов с генерацией: {answers}; утверждений: {claims_total}")
    print(f"Корректные ответы — утверждений с тревогой (не подтверждено): {claims_alarm}/{claims_total} "
          f"({claims_alarm / max(claims_total, 1):.0%}); подтверждено полностью: {claims_supported}; "
          f"ответов без единой тревоги: {answers_all_supported}/{answers} "
          f"({answers_all_supported / max(answers, 1):.0%})")
    for kind, label in (("numbers", "выдумки с числами"), ("plain", "выдумки без чисел")):
        caught, total = injected[kind]
        print(f"Внедрено {label}: поймано {caught}/{total} ({caught / max(total, 1):.0%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--limit", type=int, default=20)
    m = sub.add_parser("measure")
    m.add_argument("--cache", type=Path, required=True)
    m.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    collect(args.out, args.limit) if args.cmd == "collect" else measure(args.cache, args.verbose)


if __name__ == "__main__":
    main()
