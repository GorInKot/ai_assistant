import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { AdminUser, CurrentUser } from "../api/types";

interface Props {
  onBackToChat: () => void;
}

const KNOWN_ROLES = ["admin", "manager", "user"] as const;

export function UsersAdmin({ onBackToChat }: Props) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [me, setMe] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [list, currentMe] = await Promise.all([
        api.get<AdminUser[]>("/api/admin/users"),
        api.get<CurrentUser>("/api/user/profile"),
      ]);
      setUsers(list);
      setMe(currentMe);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load().catch(() => undefined);
  }, [load]);

  const toggleRole = async (user: AdminUser, role: string) => {
    const next = new Set(user.roles);
    if (next.has(role)) next.delete(role);
    else next.add(role);
    setBusyId(user.id);
    try {
      const updated = await api.put<AdminUser>(`/api/admin/users/${user.id}/roles`, {
        roles: Array.from(next),
      });
      setUsers((prev) => prev.map((u) => (u.id === user.id ? updated : u)));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось сменить роли");
    } finally {
      setBusyId(null);
    }
  };

  const resetPassword = async (user: AdminUser) => {
    const newPassword = prompt(`Новый пароль для ${user.email} (мин. 6 символов):`);
    if (!newPassword) return;
    if (newPassword.length < 6) {
      alert("Пароль должен быть не короче 6 символов");
      return;
    }
    setBusyId(user.id);
    try {
      await api.post(`/api/admin/users/${user.id}/password`, { new_password: newPassword });
      alert("Пароль сброшен");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось сбросить пароль");
    } finally {
      setBusyId(null);
    }
  };

  const deleteUser = async (user: AdminUser) => {
    if (!confirm(`Удалить пользователя ${user.email}? Это удалит его беседы и историю.`)) return;
    setBusyId(user.id);
    try {
      await api.delete(`/api/admin/users/${user.id}`);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось удалить");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-3">
        <button onClick={onBackToChat} className="text-sm text-slate-600 hover:text-slate-900">
          ← Назад в чат
        </button>
        <h1 className="text-sm font-semibold text-slate-700">Пользователи системы</h1>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-4">
          {error && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              {error}
              <button onClick={() => setError(null)} className="ml-2 underline">
                закрыть
              </button>
            </div>
          )}

          {loading ? (
            <p className="text-sm text-slate-500">Загрузка…</p>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-xs font-semibold text-slate-600">
                  <tr>
                    <th className="px-3 py-2 text-left">Email</th>
                    <th className="px-3 py-2 text-left">ФИО</th>
                    <th className="px-3 py-2 text-left">Подразделение</th>
                    <th className="px-3 py-2 text-left">Роли</th>
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {users.map((user) => {
                    const isSelf = me?.email === user.email;
                    return (
                      <tr key={user.id} className="hover:bg-slate-50">
                        <td className="px-3 py-2 font-medium text-slate-800">
                          {user.email}
                          {isSelf && (
                            <span className="ml-2 text-xs text-slate-400">(вы)</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-slate-600">{user.full_name || "—"}</td>
                        <td className="px-3 py-2 text-slate-600">
                          {user.division || "—"}
                          {user.subdivision ? ` / ${user.subdivision}` : ""}
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex flex-wrap gap-1">
                            {KNOWN_ROLES.map((role) => {
                              const active = user.roles.includes(role);
                              const disabled =
                                busyId === user.id || (isSelf && role === "admin" && active);
                              return (
                                <button
                                  key={role}
                                  type="button"
                                  disabled={disabled}
                                  title={
                                    isSelf && role === "admin" && active
                                      ? "Нельзя снять admin у самого себя"
                                      : undefined
                                  }
                                  onClick={() => toggleRole(user, role)}
                                  className={
                                    "text-xs rounded-full border px-3 py-0.5 transition " +
                                    (active
                                      ? "bg-accent text-white border-accent"
                                      : "bg-white text-slate-600 border-slate-300 hover:border-accent") +
                                    (disabled ? " opacity-50 cursor-not-allowed" : "")
                                  }
                                >
                                  {role}
                                </button>
                              );
                            })}
                          </div>
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-right">
                          <button
                            onClick={() => resetPassword(user)}
                            disabled={busyId === user.id}
                            className="text-xs text-accent hover:underline mr-3 disabled:opacity-50"
                          >
                            сброс пароля
                          </button>
                          {!isSelf && (
                            <button
                              onClick={() => deleteUser(user)}
                              disabled={busyId === user.id}
                              className="text-xs text-red-600 hover:underline disabled:opacity-50"
                            >
                              удалить
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="text-xs text-slate-400">
            Роли: <code>admin</code> — полный доступ, <code>manager</code> — CRUD сотрудников,
            <code> user</code> — обычный пользователь чата.
            Сброс пароля назначает новый без подтверждения старого.
          </div>
        </div>
      </div>
    </div>
  );
}
