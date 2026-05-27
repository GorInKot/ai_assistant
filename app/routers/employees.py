"""CRUD сотрудников и областей ответственности (только для admin/manager).

Также lookup ответственного по area+division+subdivision — публичный,
доступен любому авторизованному пользователю (нужно для Этапа 5).
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_db, require_role
from app.db import (
    ROLE_ADMIN,
    ROLE_MANAGER,
    Employee,
    Responsibility,
    ResponsibilityArea,
    User,
)
from app.schemas import (
    EmployeeImportResult,
    EmployeeIn,
    EmployeeOut,
    ResponsibilityAreaIn,
    ResponsibilityAreaOut,
    ResponsibilityAssign,
    ResponsibilityOut,
)
from app.services.employee_import import import_employees
from app.services.responsibilities import lookup_responsible


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

admin_only = require_role(ROLE_ADMIN, ROLE_MANAGER)

router = APIRouter(prefix="/api")


# ---------- Employees ----------

def _employee_to_dict(emp: Employee) -> dict:
    return {
        "id": emp.id,
        "user_id": emp.user_id,
        "email": emp.email,
        "full_name": emp.full_name,
        "position": emp.position,
        "division": emp.division,
        "subdivision": emp.subdivision,
        "phone": emp.phone,
        "is_active": emp.is_active,
        "responsibility_area_slugs": sorted(
            {r.area.slug for r in emp.responsibilities if r.area}
        ),
    }


def _sync_employee_areas(
    db: Session,
    employee: Employee,
    area_slugs: list[str],
) -> None:
    """Синхронизирует список областей ответственности сотрудника.

    Не задаёт scope — записи создаются с scope_division=NULL (глобальный fallback).
    Для тонкой настройки scope — отдельный endpoint POST /api/admin/responsibilities.
    """
    existing_areas = db.query(ResponsibilityArea).filter(
        ResponsibilityArea.slug.in_(area_slugs)
    ).all()
    found_slugs = {a.slug for a in existing_areas}
    missing = [slug for slug in area_slugs if slug not in found_slugs]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестные области: {', '.join(missing)}",
        )

    desired_area_ids = {a.id for a in existing_areas}
    current_area_ids = {r.area_id for r in employee.responsibilities if r.scope_division is None and r.scope_subdivision is None}

    # Удаляем глобальные responsibilities, которых больше нет в новом списке.
    for resp in list(employee.responsibilities):
        if resp.scope_division is None and resp.scope_subdivision is None:
            if resp.area_id not in desired_area_ids:
                db.delete(resp)

    # Создаём новые глобальные responsibilities для добавленных областей.
    for area in existing_areas:
        if area.id not in current_area_ids:
            db.add(Responsibility(
                employee_id=employee.id,
                area_id=area.id,
                scope_division=None,
                scope_subdivision=None,
                is_primary=True,
            ))


@router.get("/admin/employees", response_model=list[EmployeeOut])
def list_employees(
    division: str | None = Query(default=None),
    subdivision: str | None = Query(default=None),
    is_active: int | None = Query(default=None),
    q: str | None = Query(default=None, description="Поиск по ФИО/email/должности"),
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = db.query(Employee)
    if division:
        query = query.filter(Employee.division == division)
    if subdivision:
        query = query.filter(Employee.subdivision == subdivision)
    if is_active is not None:
        query = query.filter(Employee.is_active == bool(is_active))
    if q:
        like = f"%{q.lower()}%"
        query = query.filter(
            (Employee.full_name.ilike(like))
            | (Employee.email.ilike(like))
            | (Employee.position.ilike(like))
        )
    items = query.order_by(Employee.full_name).all()
    return [_employee_to_dict(emp) for emp in items]


@router.post("/admin/employees", response_model=EmployeeOut, status_code=201)
def create_employee(
    payload: EmployeeIn,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    email = payload.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Некорректный email")

    if db.query(Employee).filter(Employee.email == email).first():
        raise HTTPException(status_code=400, detail="Сотрудник с таким email уже существует")

    employee = Employee(
        email=email,
        full_name=payload.full_name.strip(),
        position=(payload.position or "").strip() or None,
        division=(payload.division or "").strip() or None,
        subdivision=(payload.subdivision or "").strip() or None,
        phone=(payload.phone or "").strip() or None,
        is_active=payload.is_active,
    )
    db.add(employee)
    db.flush()

    if payload.responsibility_area_slugs:
        _sync_employee_areas(db, employee, payload.responsibility_area_slugs)

    db.commit()
    db.refresh(employee)
    return _employee_to_dict(employee)


@router.get("/admin/employees/{employee_id}", response_model=EmployeeOut)
def get_employee(
    employee_id: int,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")
    return _employee_to_dict(emp)


@router.put("/admin/employees/{employee_id}", response_model=EmployeeOut)
def update_employee(
    employee_id: int,
    payload: EmployeeIn,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")

    email = payload.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Некорректный email")
    if email != emp.email and db.query(Employee).filter(Employee.email == email).first():
        raise HTTPException(status_code=400, detail="Email уже занят другим сотрудником")

    emp.email = email
    emp.full_name = payload.full_name.strip()
    emp.position = (payload.position or "").strip() or None
    emp.division = (payload.division or "").strip() or None
    emp.subdivision = (payload.subdivision or "").strip() or None
    emp.phone = (payload.phone or "").strip() or None
    emp.is_active = payload.is_active

    _sync_employee_areas(db, emp, payload.responsibility_area_slugs)

    db.commit()
    db.refresh(emp)
    return _employee_to_dict(emp)


@router.post("/admin/employees/import", response_model=EmployeeImportResult)
async def import_employees_route(
    file: UploadFile = File(..., description="CSV или XLSX файл со списком сотрудников"),
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> EmployeeImportResult:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл пустой")

    summary = import_employees(db, content=content, filename=file.filename or "")
    return EmployeeImportResult(
        created=summary.created,
        updated=summary.updated,
        skipped=summary.skipped,
        errors=summary.errors,
    )


@router.delete("/admin/employees/{employee_id}")
def delete_employee(
    employee_id: int,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")
    # Soft-delete: помечаем неактивным. responsibilities остаются, но сотрудник
    # не будет выбран в lookup_responsible (там фильтр по is_active).
    emp.is_active = False
    db.commit()
    return {"status": "deactivated"}


# ---------- Responsibility Areas ----------

@router.get("/admin/areas", response_model=list[ResponsibilityAreaOut])
def list_areas(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ResponsibilityArea]:
    return db.query(ResponsibilityArea).order_by(ResponsibilityArea.name).all()


@router.post("/admin/areas", response_model=ResponsibilityAreaOut, status_code=201)
def create_area(
    payload: ResponsibilityAreaIn,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> ResponsibilityArea:
    if db.query(ResponsibilityArea).filter(ResponsibilityArea.slug == payload.slug).first():
        raise HTTPException(status_code=400, detail="Область с таким slug уже существует")
    area = ResponsibilityArea(
        slug=payload.slug,
        name=payload.name,
        description=payload.description,
    )
    db.add(area)
    db.commit()
    db.refresh(area)
    return area


@router.delete("/admin/areas/{area_id}")
def delete_area(
    area_id: int,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    area = db.query(ResponsibilityArea).filter(ResponsibilityArea.id == area_id).first()
    if not area:
        raise HTTPException(status_code=404, detail="Область не найдена")
    db.delete(area)
    db.commit()
    return {"status": "deleted"}


# ---------- Responsibility assignments ----------

@router.get("/admin/responsibilities", response_model=list[ResponsibilityOut])
def list_responsibilities(
    area_slug: str | None = Query(default=None),
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = (
        db.query(Responsibility, Employee, ResponsibilityArea)
        .join(Employee, Employee.id == Responsibility.employee_id)
        .join(ResponsibilityArea, ResponsibilityArea.id == Responsibility.area_id)
    )
    if area_slug:
        query = query.filter(ResponsibilityArea.slug == area_slug)
    rows = query.order_by(ResponsibilityArea.name, Employee.full_name).all()
    return [
        {
            "id": resp.id,
            "employee_id": emp.id,
            "employee_full_name": emp.full_name,
            "employee_email": emp.email,
            "area_slug": area.slug,
            "area_name": area.name,
            "scope_division": resp.scope_division,
            "scope_subdivision": resp.scope_subdivision,
            "is_primary": resp.is_primary,
        }
        for resp, emp, area in rows
    ]


@router.post("/admin/responsibilities", response_model=ResponsibilityOut, status_code=201)
def assign_responsibility(
    payload: ResponsibilityAssign,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict:
    emp = db.query(Employee).filter(Employee.id == payload.employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")
    area = db.query(ResponsibilityArea).filter(ResponsibilityArea.slug == payload.area_slug).first()
    if not area:
        raise HTTPException(status_code=404, detail="Область не найдена")

    duplicate = (
        db.query(Responsibility)
        .filter(
            Responsibility.employee_id == emp.id,
            Responsibility.area_id == area.id,
            Responsibility.scope_division == payload.scope_division,
            Responsibility.scope_subdivision == payload.scope_subdivision,
        )
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=400, detail="Такое назначение уже существует")

    resp = Responsibility(
        employee_id=emp.id,
        area_id=area.id,
        scope_division=payload.scope_division,
        scope_subdivision=payload.scope_subdivision,
        is_primary=payload.is_primary,
    )
    db.add(resp)
    db.commit()
    db.refresh(resp)
    return {
        "id": resp.id,
        "employee_id": emp.id,
        "employee_full_name": emp.full_name,
        "employee_email": emp.email,
        "area_slug": area.slug,
        "area_name": area.name,
        "scope_division": resp.scope_division,
        "scope_subdivision": resp.scope_subdivision,
        "is_primary": resp.is_primary,
    }


@router.delete("/admin/responsibilities/{responsibility_id}")
def remove_responsibility(
    responsibility_id: int,
    current_user: User = Depends(admin_only),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    resp = db.query(Responsibility).filter(Responsibility.id == responsibility_id).first()
    if not resp:
        raise HTTPException(status_code=404, detail="Назначение не найдено")
    db.delete(resp)
    db.commit()
    return {"status": "deleted"}


# ---------- Public lookup (для Этапа 5) ----------

@router.get("/responsibilities/lookup")
def lookup_responsible_route(
    area: str = Query(..., description="slug области ответственности"),
    division: str | None = Query(default=None),
    subdivision: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Возвращает первого подходящего активного ответственного.

    Если ничего не найдено — 404 с понятным сообщением (пользователь увидит:
    «по этой области пока нет ответственного, обратитесь в админ»).
    """
    employee = lookup_responsible(db, area_slug=area, division=division, subdivision=subdivision)
    if not employee:
        raise HTTPException(
            status_code=404,
            detail=f"По области '{area}' ответственный не назначен",
        )
    return {
        "id": employee.id,
        "full_name": employee.full_name,
        "email": employee.email,
        "position": employee.position,
        "division": employee.division,
        "subdivision": employee.subdivision,
        "phone": employee.phone,
    }
