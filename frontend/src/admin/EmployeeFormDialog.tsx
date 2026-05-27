import { useState, FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { Employee, ResponsibilityArea } from "../api/types";

interface Props {
  employee: Employee | null;
  areas: ResponsibilityArea[];
  onClose: () => void;
  onSaved: () => Promise<void> | void;
}

export function EmployeeFormDialog({ employee, areas, onClose, onSaved }: Props) {
  const isEdit = employee !== null;
  const [email, setEmail] = useState(employee?.email ?? "");
  const [fullName, setFullName] = useState(employee?.full_name ?? "");
  const [position, setPosition] = useState(employee?.position ?? "");
  const [division, setDivision] = useState(employee?.division ?? "");
  const [subdivision, setSubdivision] = useState(employee?.subdivision ?? "");
  const [phone, setPhone] = useState(employee?.phone ?? "");
  const [isActive, setIsActive] = useState(employee?.is_active ?? true);
  const [selectedAreas, setSelectedAreas] = useState<Set<string>>(
    new Set(employee?.responsibility_area_slugs ?? []),
  );
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const toggleArea = (slug: string) => {
    const next = new Set(selectedAreas);
    if (next.has(slug)) next.delete(slug);
    else next.add(slug);
    setSelectedAreas(next);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    const payload = {
      email,
      full_name: fullName,
      position: position || null,
      division: division || null,
      subdivision: subdivision || null,
      phone: phone || null,
      is_active: isActive,
      responsibility_area_slugs: Array.from(selectedAreas),
    };
    try {
      if (isEdit && employee) {
        await api.put(`/api/admin/employees/${employee.id}`, payload);
      } else {
        await api.post("/api/admin/employees", payload);
      }
      await onSaved();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка сохранения");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-4">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-xl border border-slate-200">
        <header className="flex items-center justify-between px-5 py-3 border-b border-slate-200">
          <h2 className="text-sm font-semibold text-slate-800">
            {isEdit ? "Редактировать сотрудника" : "Добавить сотрудника"}
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700 text-lg leading-none">
            ×
          </button>
        </header>

        <form className="p-5 space-y-3 max-h-[80vh] overflow-y-auto" onSubmit={handleSubmit}>
          <Field label="Email" required>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className={inputClass}
              disabled={isEdit}
            />
          </Field>
          <Field label="ФИО" required>
            <input
              required
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className={inputClass}
            />
          </Field>
          <Field label="Должность">
            <input value={position} onChange={(e) => setPosition(e.target.value)} className={inputClass} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Филиал">
              <input value={division} onChange={(e) => setDivision(e.target.value)} className={inputClass} />
            </Field>
            <Field label="Подразделение">
              <input
                value={subdivision}
                onChange={(e) => setSubdivision(e.target.value)}
                className={inputClass}
              />
            </Field>
          </div>
          <Field label="Телефон">
            <input value={phone} onChange={(e) => setPhone(e.target.value)} className={inputClass} />
          </Field>

          <div>
            <span className="text-xs font-medium text-slate-600 mb-1.5 block">
              Области ответственности
            </span>
            <div className="flex flex-wrap gap-1.5">
              {areas.map((area) => (
                <button
                  type="button"
                  key={area.slug}
                  onClick={() => toggleArea(area.slug)}
                  className={
                    "text-xs rounded-full border px-3 py-1 transition " +
                    (selectedAreas.has(area.slug)
                      ? "bg-accent text-white border-accent"
                      : "bg-white text-slate-700 border-slate-300 hover:border-accent")
                  }
                >
                  {area.name}
                </button>
              ))}
            </div>
          </div>

          {isEdit && (
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={isActive}
                onChange={(e) => setIsActive(e.target.checked)}
              />
              Активен
            </label>
          )}

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="flex justify-end gap-2 pt-2 border-t border-slate-200">
            <button type="button" onClick={onClose} className={btnSecondary}>
              Отмена
            </button>
            <button type="submit" disabled={saving} className={btnPrimary}>
              {saving ? "Сохранение…" : "Сохранить"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

const inputClass =
  "w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent disabled:bg-slate-100 disabled:text-slate-500";
const btnPrimary =
  "rounded-lg bg-accent text-white font-medium px-4 py-2 text-sm hover:bg-accent-dark disabled:opacity-60";
const btnSecondary =
  "rounded-lg border border-slate-300 text-slate-700 font-medium px-4 py-2 text-sm hover:bg-slate-100";

function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-600 mb-1 block">
        {label}
        {required && <span className="text-red-500 ml-0.5">*</span>}
      </span>
      {children}
    </label>
  );
}
