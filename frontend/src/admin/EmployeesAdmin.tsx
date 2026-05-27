import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Employee, EmployeeImportResult, ResponsibilityArea } from "../api/types";
import { EmployeeFormDialog } from "./EmployeeFormDialog";

interface Props {
  onBackToChat: () => void;
}

export function EmployeesAdmin({ onBackToChat }: Props) {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [areas, setAreas] = useState<ResponsibilityArea[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<Employee | null>(null);
  const [creating, setCreating] = useState(false);
  const [importResult, setImportResult] = useState<EmployeeImportResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [emps, ars] = await Promise.all([
        api.get<Employee[]>("/api/admin/employees" + (search ? `?q=${encodeURIComponent(search)}` : "")),
        api.get<ResponsibilityArea[]>("/api/admin/areas"),
      ]);
      setEmployees(emps);
      setAreas(ars);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  }, [search]);

  useEffect(() => {
    loadAll().catch(() => undefined);
  }, [loadAll]);

  const handleDelete = async (employee: Employee) => {
    if (!confirm(`Деактивировать «${employee.full_name}»?`)) return;
    try {
      await api.delete(`/api/admin/employees/${employee.id}`);
      await loadAll();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка удаления");
    }
  };

  const handleImport = async (file: File) => {
    try {
      setError(null);
      const result = await api.upload<EmployeeImportResult>("/api/admin/employees/import", file);
      setImportResult(result);
      await loadAll();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка импорта");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-3">
        <button
          onClick={onBackToChat}
          className="text-sm text-slate-600 hover:text-slate-900"
        >
          ← Назад в чат
        </button>
        <h1 className="text-sm font-semibold text-slate-700">База сотрудников</h1>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-6 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="search"
              placeholder="Поиск по ФИО / email / должности…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="flex-1 min-w-[200px] rounded-lg border border-slate-300 px-3 py-2 text-sm"
            />
            <button onClick={() => setCreating(true)} className={btnPrimary}>
              + Добавить
            </button>
            <button onClick={() => fileInputRef.current?.click()} className={btnSecondary}>
              Импорт CSV/XLSX
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.txt"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleImport(file);
              }}
            />
          </div>

          {error && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {importResult && (
            <div className="text-sm bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2 text-emerald-800">
              <div>
                Импорт: создано {importResult.created}, обновлено {importResult.updated},
                пропущено {importResult.skipped}
              </div>
              {importResult.errors.length > 0 && (
                <ul className="mt-1 text-red-700 list-disc pl-5">
                  {importResult.errors.slice(0, 10).map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              )}
              <button
                onClick={() => setImportResult(null)}
                className="mt-1 text-xs underline text-emerald-700"
              >
                закрыть
              </button>
            </div>
          )}

          {loading ? (
            <p className="text-sm text-slate-500">Загрузка…</p>
          ) : employees.length === 0 ? (
            <p className="text-sm text-slate-500">
              Сотрудники пока не добавлены. Используйте кнопки выше — «Добавить» или «Импорт».
            </p>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-xs font-semibold text-slate-600">
                  <tr>
                    <th className="px-3 py-2 text-left">ФИО</th>
                    <th className="px-3 py-2 text-left">Email</th>
                    <th className="px-3 py-2 text-left">Должность</th>
                    <th className="px-3 py-2 text-left">Подразделение</th>
                    <th className="px-3 py-2 text-left">Области</th>
                    <th className="px-3 py-2 text-left">Статус</th>
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {employees.map((emp) => (
                    <tr key={emp.id} className="hover:bg-slate-50">
                      <td className="px-3 py-2 font-medium text-slate-800">{emp.full_name}</td>
                      <td className="px-3 py-2 text-slate-600">{emp.email}</td>
                      <td className="px-3 py-2 text-slate-600">{emp.position || "—"}</td>
                      <td className="px-3 py-2 text-slate-600">
                        {emp.division || "—"}
                        {emp.subdivision ? ` / ${emp.subdivision}` : ""}
                      </td>
                      <td className="px-3 py-2">
                        {emp.responsibility_area_slugs.length === 0 ? (
                          <span className="text-slate-400">—</span>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {emp.responsibility_area_slugs.map((slug) => {
                              const area = areas.find((a) => a.slug === slug);
                              return (
                                <span
                                  key={slug}
                                  className="text-xs px-2 py-0.5 rounded bg-accent-soft text-accent-dark"
                                >
                                  {area?.name ?? slug}
                                </span>
                              );
                            })}
                          </div>
                        )}
                      </td>
                      <td className="px-3 py-2">
                        {emp.is_active ? (
                          <span className="text-xs px-2 py-0.5 rounded bg-emerald-50 text-emerald-700">
                            активен
                          </span>
                        ) : (
                          <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-500">
                            деактивирован
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        <button
                          onClick={() => setEditing(emp)}
                          className="text-xs text-accent hover:underline mr-2"
                        >
                          ✎
                        </button>
                        {emp.is_active && (
                          <button
                            onClick={() => handleDelete(emp)}
                            className="text-xs text-red-600 hover:underline"
                          >
                            ✕
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="text-xs text-slate-400">
            <strong>Шаблон CSV:</strong> заголовки{" "}
            <code>email, full_name, position, division, subdivision, phone, responsibility_areas</code>.
            Области через <code>;</code> или <code>,</code>. Поддерживаются slug-и: {areas.map((a) => a.slug).join(", ")}.
          </div>
        </div>
      </div>

      {(editing || creating) && (
        <EmployeeFormDialog
          employee={editing}
          areas={areas}
          onClose={() => {
            setEditing(null);
            setCreating(false);
          }}
          onSaved={async () => {
            setEditing(null);
            setCreating(false);
            await loadAll();
          }}
        />
      )}
    </div>
  );
}

const btnPrimary =
  "rounded-lg bg-accent text-white font-medium px-3 py-2 text-sm hover:bg-accent-dark";
const btnSecondary =
  "rounded-lg border border-slate-300 text-slate-700 font-medium px-3 py-2 text-sm hover:bg-slate-100";
