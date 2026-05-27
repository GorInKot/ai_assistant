"""Логика поиска ответственного по области и подразделению с fallback.

Используется и в админ-CRUD, и в Этапе 5 (заявки маршрутизируются ответственному).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db import Employee, Responsibility, ResponsibilityArea


def lookup_responsible(
    db: Session,
    area_slug: str,
    division: str | None = None,
    subdivision: str | None = None,
) -> Employee | None:
    """Найти первого подходящего активного ответственного по slug области.

    Приоритет: subdivision-match > division-match > глобальный (scope=NULL).
    Среди равных — is_primary=True вперёд, затем по id.
    """
    area = db.query(ResponsibilityArea).filter(ResponsibilityArea.slug == area_slug).first()
    if not area:
        return None

    query = (
        db.query(Responsibility, Employee)
        .join(Employee, Employee.id == Responsibility.employee_id)
        .filter(Responsibility.area_id == area.id, Employee.is_active == True)  # noqa: E712
    )
    candidates = query.all()
    if not candidates:
        return None

    def score(resp: Responsibility) -> tuple[int, int, int]:
        # Чем больше — тем приоритетнее. Возвращаем кортеж для сортировки.
        match_subdiv = 1 if (subdivision and resp.scope_subdivision == subdivision) else 0
        match_div = 1 if (division and resp.scope_division == division) else 0
        # Global fallback: scope_division IS NULL означает «по всей компании»
        is_global = 1 if (resp.scope_division is None and resp.scope_subdivision is None) else 0
        return (match_subdiv, match_div, is_global)

    def is_eligible(resp: Responsibility) -> bool:
        # Если у responsibility задан scope, то он должен совпасть с запросом
        # либо быть NULL (глобальный). Конфликтный scope (другое подразделение) — отбрасываем.
        if resp.scope_subdivision and subdivision and resp.scope_subdivision != subdivision:
            return False
        if resp.scope_division and division and resp.scope_division != division:
            return False
        # Если в запросе НЕ задан division, но ответственный привязан к конкретному
        # division, всё равно показываем (как минимум подходит лучше чем ничего).
        return True

    eligible = [(resp, emp) for resp, emp in candidates if is_eligible(resp)]
    if not eligible:
        return None

    eligible.sort(
        key=lambda pair: (
            score(pair[0])[0],
            score(pair[0])[1],
            score(pair[0])[2],
            1 if pair[0].is_primary else 0,
            -pair[1].id,
        ),
        reverse=True,
    )
    return eligible[0][1]
