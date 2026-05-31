#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Корень проекта в sys.path, чтобы `import app.*` работал при запуске
# `python scripts/run_eval.py` (sys.path[0] иначе указывает на scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class EvalCaseResult:
    case_id: str
    ok: bool
    skipped: bool
    reasons: list[str]
    no_exact_match: bool
    source_paths: list[str]
    answer_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local eval suite for the assistant.")
    parser.add_argument("--cases", default="eval/cases.json", help="Path to JSON eval cases file.")
    parser.add_argument("--output", default="eval/reports/latest.json", help="Path for report output JSON.")
    parser.add_argument("--mode", choices=["http", "local"], default="local", help="How to call the app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API URL (for --mode http).")
    parser.add_argument("--stub-llm", action="store_true", help="In local mode, stub LLM calls for deterministic/no-network eval.")
    parser.add_argument("--fail-on-errors", action="store_true", help="Return non-zero exit code if any case fails.")
    return parser.parse_args()


class AskClient:
    """Клиент к /api/ask + (в local-режиме) прямой доступ к intent-классификатору.

    /api/ask требует авторизации (security-фикс), поэтому клиент всегда добавляет
    Bearer-токен:
    - local: поднимает временную изолированную БД (eval не читает и не мутирует
      dev app_data.db), строит KB in-process и минтит токен eval-пользователю.
    - http: регистрирует/логинит технического пользователя на запущенном сервере.

    Intent-кейсы (kind="intent") в local-режиме вызывают classify_intent напрямую
    in-process — это самый точный замер качества классификатора без склейки с
    slot-filling и HTTP.
    """

    def __init__(self, mode: str, base_url: str, stub_llm: bool) -> None:
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.stub_llm = stub_llm
        self._local_client = None
        self._state = None
        self._catalog_summary = None
        self.headers: dict[str, str] = {}

        if mode == "local":
            self._init_local_client(stub_llm)
        else:
            self._init_http_auth()

    def _init_local_client(self, stub_llm: bool) -> None:
        import os
        import tempfile

        # Изолированная временная БД: eval не зависит от dev-данных и не создаёт
        # в них мусорные заявки/пользователей. Должно стоять ДО импорта app.*.
        if not os.getenv("DATABASE_URL", "").startswith("sqlite:///") or "app_data.db" in os.getenv("DATABASE_URL", ""):
            fd, path = tempfile.mkstemp(prefix="eval_", suffix=".db")
            os.close(fd)
            os.environ["DATABASE_URL"] = f"sqlite:///{path}"

        from fastapi.testclient import TestClient

        import app.state as state
        from app.main import app
        from app.services.ask_service import _build_catalog_summary

        self._state = state
        self._catalog_summary = _build_catalog_summary

        if stub_llm:
            # Без сети: валидируем retrieval/источники. classify_intent при
            # client=None отдаёт qa-fallback, поэтому intent/slot-кейсы скипаем.
            state.llm_service.enable_rerank = False
            state.llm_service.client = None
            state.llm_service.generate_answer = (
                lambda question, context_results, intent="procedure": "STUB_ANSWER"
            )

        with state.kb_lock:
            state.kb_index.build()

        self._local_client = TestClient(app)
        self.headers = {"Authorization": f"Bearer {self._mint_local_token()}"}

    def _mint_local_token(self) -> str:
        """Гарантируем eval-пользователя в БД и выдаём ему JWT in-process."""
        from app.auth import create_access_token, get_password_hash
        from app.db import ROLE_USER, Role, SessionLocal, User

        email = "eval-runner@local.test"
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                user = User(
                    email=email,
                    full_name="Eval Runner",
                    hashed_password=get_password_hash("eval-pass"),
                )
                role = db.query(Role).filter(Role.name == ROLE_USER).first()
                if role:
                    user.roles.append(role)
                db.add(user)
                db.commit()
        finally:
            db.close()
        return create_access_token({"sub": email})

    def _init_http_auth(self) -> None:
        """Регистрируем технического пользователя на внешнем сервере, берём токен.

        Безопасность: email и пароль генерируются случайно на каждый прогон —
        раннер не оставляет предсказуемой учётки с известным паролем (на случай,
        если http-режим по ошибке нацелили на боевой сервер). Дополнительно
        http-режим против не-loopback хоста требует явного EVAL_ALLOW_REMOTE=1.
        """
        import os
        import secrets
        from urllib.parse import urlparse

        host = (urlparse(self.base_url).hostname or "").lower()
        if host not in {"localhost", "127.0.0.1", "::1", ""} and os.getenv("EVAL_ALLOW_REMOTE") != "1":
            raise RuntimeError(
                f"http-режим нацелен на не-локальный хост '{host}'. Eval регистрирует "
                "тестового пользователя — против боевого сервера это нежелательно. "
                "Если действительно нужно, запустите с EVAL_ALLOW_REMOTE=1."
            )

        # Случайные креды на прогон: не оставляем предсказуемой учётки с известным
        # паролем. /api/auth/register сразу возвращает access_token.
        email = f"eval-runner+{secrets.token_hex(6)}@local.test"
        password = secrets.token_urlsafe(24)
        register_payload = {
            "email": email,
            "password": password,
            "confirm_password": password,
            "last_name": "Runner",
            "first_name": "Eval",
            "division": "ЦА",
        }
        reg = self._http_json("POST", "/api/auth/register", register_payload)
        token = (reg.get("body") or {}).get("access_token")
        if not token:
            raise RuntimeError(f"HTTP auth failed: {reg}")
        self.headers = {"Authorization": f"Bearer {token}"}

    def _http_json(self, method: str, path: str, payload: dict) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                return {"status": response.status, "body": json.loads(body) if body else {}}
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8") if error.fp else ""
            parsed: dict | Any = {}
            if body:
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = {"detail": body}
            return {"status": error.code, "body": parsed}

    def ask(self, question: str, session_id: str) -> dict[str, Any]:
        payload = {"question": question, "session_id": session_id}

        if self.mode == "local":
            assert self._local_client is not None
            response = self._local_client.post("/api/ask", json=payload, headers=self.headers)
            return {"status": response.status_code, "body": response.json()}

        return self._http_json("POST", "/api/ask", payload)

    def classify(self, message: str) -> dict[str, Any]:
        """Прямой вызов intent-классификатора (только local + реальный LLM)."""
        assert self._state is not None and self._catalog_summary is not None
        return self._state.llm_service.classify_intent(message, self._catalog_summary())



def source_process(relative_path: str) -> str:
    return relative_path.split("/", 1)[0] if "/" in relative_path else "Общее"



def _skipped(case_id: str, reason: str) -> EvalCaseResult:
    return EvalCaseResult(
        case_id=case_id,
        ok=True,
        skipped=True,
        reasons=[reason],
        no_exact_match=False,
        source_paths=[],
        answer_preview="",
    )


def run_intent_case(client: AskClient, case: dict[str, Any], case_id: str) -> EvalCaseResult:
    """Кейс kind='intent': прямой замер LLM-классификатора намерения.

    Доступен только в local-режиме с реальным LLM — в stub/http скипаем
    (нет in-process клиента или сетевого классификатора).
    """
    if client.mode != "local" or client.stub_llm:
        return _skipped(case_id, "intent case requires local mode with real LLM")

    message = str(case.get("question") or case.get("message") or "")
    result = client.classify(message)
    intent = result.get("intent")
    request_type = result.get("request_type")

    reasons: list[str] = []
    expected_intent = case.get("expected_intent")
    if expected_intent and intent != expected_intent:
        reasons.append(f"expected_intent={expected_intent}, got={intent}")

    expected_request_type = case.get("expected_request_type")
    if expected_request_type is not None and request_type != expected_request_type:
        reasons.append(f"expected_request_type={expected_request_type}, got={request_type}")

    return EvalCaseResult(
        case_id=case_id,
        ok=not reasons,
        skipped=False,
        reasons=reasons,
        no_exact_match=False,
        source_paths=[],
        answer_preview=f"intent={intent} request_type={request_type}",
    )


def run_case(client: AskClient, case: dict[str, Any], idx: int) -> EvalCaseResult:
    case_id = str(case.get("id") or f"case_{idx}")
    session_id = str(case.get("session_id") or f"eval-{case_id}")

    if str(case.get("kind", "qa")) == "intent":
        return run_intent_case(client, case, case_id)

    if client.stub_llm and case.get("skip_when_stub_llm"):
        return _skipped(case_id, "skipped in stub-llm mode")

    for pre_step in case.get("pre_steps", []):
        pre_question = str(pre_step["question"])
        pre_session = str(pre_step.get("session_id") or session_id)
        client.ask(pre_question, pre_session)

    response = client.ask(str(case["question"]), session_id)
    status = response["status"]
    body = response["body"]

    reasons: list[str] = []
    if status != 200:
        detail = body.get("detail") if isinstance(body, dict) else str(body)
        reasons.append(f"HTTP {status}: {detail}")
        return EvalCaseResult(
            case_id=case_id,
            ok=False,
            skipped=False,
            reasons=reasons,
            no_exact_match=True,
            source_paths=[],
            answer_preview="",
        )

    no_exact_match = bool(body.get("no_exact_match", True))
    answer = str(body.get("answer", ""))
    sources = body.get("sources", []) or []
    source_paths = [str(item.get("relative_path", "")) for item in sources]

    expected_no_exact_match = case.get("expected_no_exact_match")
    if expected_no_exact_match is not None and bool(expected_no_exact_match) != no_exact_match:
        reasons.append(f"expected_no_exact_match={expected_no_exact_match}, got={no_exact_match}")

    expected_source_any = case.get("expected_source_any") or []
    if expected_source_any:
        if not any(any(token in path for token in expected_source_any) for path in source_paths):
            reasons.append(f"no source matched any of {expected_source_any}")

    expected_source_all = case.get("expected_source_all") or []
    if expected_source_all:
        for token in expected_source_all:
            if not any(token in path for path in source_paths):
                reasons.append(f"no source matched required token '{token}'")

    expected_process = case.get("expected_process")
    if expected_process:
        if not any(source_process(path) == expected_process for path in source_paths):
            reasons.append(f"no source from expected process '{expected_process}'")

    if case.get("expected_single_process") and source_paths:
        processes = {source_process(path) for path in source_paths}
        if len(processes) > 1:
            reasons.append(f"expected single process in sources, got {sorted(processes)}")

    min_sources = case.get("min_sources")
    if min_sources is not None and len(source_paths) < int(min_sources):
        reasons.append(f"expected at least {min_sources} sources, got {len(source_paths)}")

    max_sources = case.get("max_sources")
    if max_sources is not None and len(source_paths) > int(max_sources):
        reasons.append(f"expected at most {max_sources} sources, got {len(source_paths)}")

    contains_any = case.get("contains_in_answer_any") or []
    if contains_any and not any(token.lower() in answer.lower() for token in contains_any):
        reasons.append(f"answer does not contain any of {contains_any}")

    forbid_any = case.get("forbid_in_answer") or []
    for token in forbid_any:
        if token.lower() in answer.lower():
            reasons.append(f"answer contains forbidden token '{token}'")

    return EvalCaseResult(
        case_id=case_id,
        ok=not reasons,
        skipped=False,
        reasons=reasons,
        no_exact_match=no_exact_match,
        source_paths=source_paths,
        answer_preview=answer[:220],
    )



def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases file must be a JSON array")
    return payload



def build_report(results: list[EvalCaseResult], mode: str, cases_path: str) -> dict[str, Any]:
    total = len(results)
    skipped = sum(1 for item in results if item.skipped)
    executed = total - skipped
    passed = sum(1 for item in results if item.ok and not item.skipped)
    failed = sum(1 for item in results if (not item.ok) and (not item.skipped))
    no_exact = sum(1 for item in results if item.no_exact_match and not item.skipped)

    report_results: list[dict[str, Any]] = []
    for item in results:
        report_results.append(
            {
                "id": item.case_id,
                "ok": item.ok,
                "skipped": item.skipped,
                "reasons": item.reasons,
                "no_exact_match": item.no_exact_match,
                "source_paths": item.source_paths,
                "answer_preview": item.answer_preview,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "cases_path": cases_path,
        "summary": {
            "total": total,
            "executed": executed,
            "skipped": skipped,
            "passed": passed,
            "failed": failed,
            "pass_rate": round((passed / executed) if executed else 0.0, 4),
            "fallback_rate": round((no_exact / executed) if executed else 0.0, 4),
        },
        "results": report_results,
    }



def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases)
    output_path = Path(args.output)

    cases = load_cases(cases_path)
    client = AskClient(mode=args.mode, base_url=args.base_url, stub_llm=args.stub_llm)

    results: list[EvalCaseResult] = []
    for idx, case in enumerate(cases, start=1):
        results.append(run_case(client, case, idx))

    report = build_report(results, mode=args.mode, cases_path=str(cases_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(
        f"Eval done: total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
        f"pass_rate={summary['pass_rate']} fallback_rate={summary['fallback_rate']}"
    )
    print(f"Report: {output_path}")

    if args.fail_on_errors and summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
