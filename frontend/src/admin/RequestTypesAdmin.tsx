import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { RequestTypeDef, ResponsibilityArea } from "../api/types";
import { RequestTypeFormDialog } from "./RequestTypeFormDialog";

interface Props {
  onBackToChat: () => void;
}

export function RequestTypesAdmin({ onBackToChat }: Props) {
  const [types, setTypes] = useState<RequestTypeDef[]>([]);
  const [areas, setAreas] = useState<ResponsibilityArea[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<RequestTypeDef | null>(null);
  const [creating, setCreating] = useState(false);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [tps, ars] = await Promise.all([
        api.get<RequestTypeDef[]>("/api/admin/request-types"),
        api.get<ResponsibilityArea[]>("/api/admin/areas"),
      ]);
      setTypes(tps);
      setAreas(ars);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll().catch(() => undefined);
  }, [loadAll]);

  const handleDelete = async (rt: RequestTypeDef) => {
    if (!confirm(`Удалить тип «${rt.title}» (${rt.type_slug})?\nУже созданные заявки этого типа останутся.`)) return;
    try {
      await api.delete(`/api/admin/request-types/${rt.type_slug}`);
      await loadAll();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка удаления");
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-3">
        <button onClick={onBackToChat} className="text-sm text-slate-600 hover:text-slate-900">
          ← Назад в чат
        </button>
        <h1 className="text-sm font-semibold text-slate-700">Каталог типов заявок</h1>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-6 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <p className="flex-1 text-xs text-slate-500">
              Типы, по которым ассистент собирает заявки. Изменения применяются мгновенно
              ко всем активным диалогам.
            </p>
            <button onClick={() => setCreating(true)} className={btnPrimary}>
              + Новый тип
            </button>
          </div>

          {error && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {loading ? (
            <p className="text-sm text-slate-500">Загрузка…</p>
          ) : types.length === 0 ? (
            <p className="text-sm text-slate-500">Каталог пуст. Добавьте первый тип.</p>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-xs font-semibold text-slate-600">
                  <tr>
                    <th className="px-3 py-2 text-left">Slug</th>
                    <th className="px-3 py-2 text-left">Название</th>
                    <th className="px-3 py-2 text-left">Область</th>
                    <th className="px-3 py-2 text-left">Слоты</th>
                    <th className="px-3 py-2 text-left">Флаги</th>
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {types.map((rt) => {
                    const area = areas.find((a) => a.slug === rt.responsibility_area_slug);
                    return (
                      <tr key={rt.type_slug} className="hover:bg-slate-50">
                        <td className="px-3 py-2 font-mono text-xs text-slate-600">{rt.type_slug}</td>
                        <td className="px-3 py-2 font-medium text-slate-800">{rt.title}</td>
                        <td className="px-3 py-2 text-slate-600">
                          {area?.name ?? rt.responsibility_area_slug}
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex flex-wrap gap-1">
                            {rt.slots.map((s) => (
                              <span
                                key={s.name}
                                className={`text-xs px-2 py-0.5 rounded ${
                                  s.required
                                    ? "bg-accent-soft text-accent-dark"
                                    : "bg-slate-100 text-slate-600"
                                }`}
                                title={s.question}
                              >
                                {s.name}
                                {s.required ? "*" : ""}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex flex-wrap gap-1">
                            {rt.is_anonymous && (
                              <span className="text-xs px-2 py-0.5 rounded bg-violet-50 text-violet-700">
                                анонимка
                              </span>
                            )}
                            {!rt.is_active && (
                              <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-500">
                                скрыт
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap">
                          <button
                            onClick={() => setEditing(rt)}
                            className="text-xs text-accent hover:underline mr-2"
                          >
                            ✎
                          </button>
                          <button
                            onClick={() => handleDelete(rt)}
                            className="text-xs text-red-600 hover:underline"
                          >
                            ✕
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="text-xs text-slate-400">
            <strong>Подсказка:</strong> trigger-ключевые слова и примеры формулировок
            используются LLM для распознавания намерения. Чем точнее — тем меньше «не понял запрос».
            Имя слота не должно содержать пробелов (snake_case).
          </div>
        </div>
      </div>

      {(editing || creating) && (
        <RequestTypeFormDialog
          requestType={editing}
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
