import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { RequestItem, RequestStatus } from "../api/types";

interface Props {
  onBackToChat: () => void;
}

type Tab = "inbox" | "my";

const STATUS_LABELS: Record<RequestStatus, string> = {
  new: "Новая",
  in_progress: "В работе",
  done: "Выполнена",
  rejected: "Отклонена",
};

const STATUS_COLORS: Record<RequestStatus, string> = {
  new: "bg-blue-50 text-blue-700",
  in_progress: "bg-amber-50 text-amber-700",
  done: "bg-emerald-50 text-emerald-700",
  rejected: "bg-slate-100 text-slate-500",
};

export function RequestsPage({ onBackToChat }: Props) {
  const [tab, setTab] = useState<Tab>("inbox");
  const [items, setItems] = useState<RequestItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<RequestItem | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const path = tab === "inbox" ? "/api/requests/inbox" : "/api/requests/my";
      const data = await api.get<RequestItem[]>(path);
      setItems(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка загрузки");
    } finally {
      setLoading(false);
    }
  }, [tab]);

  useEffect(() => {
    load().catch(() => undefined);
  }, [load]);

  const handleStatusChange = async (req: RequestItem, status: RequestStatus) => {
    try {
      const updated = await api.put<RequestItem>(`/api/requests/${req.id}/status`, { status });
      setItems((prev) => prev.map((r) => (r.id === req.id ? updated : r)));
      if (selected?.id === req.id) setSelected(updated);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось сменить статус");
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-3">
        <button onClick={onBackToChat} className="text-sm text-slate-600 hover:text-slate-900">
          ← Назад в чат
        </button>
        <h1 className="text-sm font-semibold text-slate-700">Заявки</h1>
        <div className="ml-auto flex gap-1 text-sm">
          <TabButton active={tab === "inbox"} onClick={() => setTab("inbox")}>
            Входящие
          </TabButton>
          <TabButton active={tab === "my"} onClick={() => setTab("my")}>
            Мои
          </TabButton>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-6 space-y-4">
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
          ) : items.length === 0 ? (
            <div className="text-sm text-slate-500 bg-white border border-slate-200 rounded-xl p-6">
              {tab === "inbox"
                ? "На вас пока не назначено ни одной заявки."
                : "Вы пока не создавали заявок. Попросите ассистента в чате, например: «хочу записаться на обучение»."}
            </div>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-xs font-semibold text-slate-600">
                  <tr>
                    <th className="px-3 py-2 text-left">#</th>
                    <th className="px-3 py-2 text-left">Тип</th>
                    <th className="px-3 py-2 text-left">Краткое описание</th>
                    {tab === "inbox" ? (
                      <th className="px-3 py-2 text-left">От кого</th>
                    ) : (
                      <th className="px-3 py-2 text-left">Ответственный</th>
                    )}
                    <th className="px-3 py-2 text-left">Статус</th>
                    <th className="px-3 py-2 text-left">Создана</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {items.map((req) => (
                    <tr
                      key={req.id}
                      onClick={() => setSelected(req)}
                      className="hover:bg-slate-50 cursor-pointer"
                    >
                      <td className="px-3 py-2 font-medium text-slate-800">#{req.id}</td>
                      <td className="px-3 py-2 text-slate-700">{req.type_title}</td>
                      <td className="px-3 py-2 text-slate-600">{req.summary || "—"}</td>
                      <td className="px-3 py-2 text-slate-600">
                        {tab === "inbox"
                          ? req.is_anonymous
                            ? <span className="italic text-slate-400">анонимно</span>
                            : req.requester_name || "—"
                          : req.assigned_employee_name || (
                              <span className="text-amber-600">не назначен</span>
                            )}
                      </td>
                      <td className="px-3 py-2">
                        <span className={"text-xs px-2 py-0.5 rounded " + STATUS_COLORS[req.status]}>
                          {STATUS_LABELS[req.status]}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-500">
                        {new Date(req.created_at).toLocaleString("ru-RU")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {selected && (
        <RequestDetailDialog
          request={selected}
          canChangeStatus={tab === "inbox"}
          onClose={() => setSelected(null)}
          onStatusChange={(s) => handleStatusChange(selected, s)}
        />
      )}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "rounded-lg px-3 py-1.5 text-xs font-semibold transition " +
        (active
          ? "bg-accent text-white"
          : "text-slate-600 hover:bg-slate-100")
      }
    >
      {children}
    </button>
  );
}

function RequestDetailDialog({
  request,
  canChangeStatus,
  onClose,
  onStatusChange,
}: {
  request: RequestItem;
  canChangeStatus: boolean;
  onClose: () => void;
  onStatusChange: (status: RequestStatus) => void;
}) {
  const statuses: RequestStatus[] = ["new", "in_progress", "done", "rejected"];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-4">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-xl border border-slate-200">
        <header className="flex items-center justify-between px-5 py-3 border-b border-slate-200">
          <h2 className="text-sm font-semibold text-slate-800">
            Заявка #{request.id} — {request.type_title}
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700 text-lg leading-none">
            ×
          </button>
        </header>
        <div className="p-5 space-y-3 text-sm max-h-[70vh] overflow-y-auto">
          <DetailRow label="Статус">
            <span className={"px-2 py-0.5 rounded text-xs " + STATUS_COLORS[request.status]}>
              {STATUS_LABELS[request.status]}
            </span>
          </DetailRow>
          <DetailRow label="Создана">
            {new Date(request.created_at).toLocaleString("ru-RU")}
          </DetailRow>
          {!request.is_anonymous && request.requester_name && (
            <DetailRow label="Автор">
              {request.requester_name}{" "}
              {request.requester_email && (
                <span className="text-slate-400 text-xs">({request.requester_email})</span>
              )}
            </DetailRow>
          )}
          {request.is_anonymous && (
            <DetailRow label="Автор">
              <span className="italic text-slate-500">анонимно</span>
            </DetailRow>
          )}
          {request.assigned_employee_name && (
            <DetailRow label="Ответственный">{request.assigned_employee_name}</DetailRow>
          )}

          <div className="border-t border-slate-100 pt-3 mt-3">
            <div className="text-xs font-semibold text-slate-600 mb-1">Поля заявки</div>
            {Object.keys(request.payload).length === 0 ? (
              <div className="text-slate-400 text-xs">—</div>
            ) : (
              <dl className="space-y-1.5">
                {Object.entries(request.payload).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <dt className="text-xs text-slate-500 min-w-[140px]">{key}:</dt>
                    <dd className="text-sm text-slate-800 break-words">{value || "—"}</dd>
                  </div>
                ))}
              </dl>
            )}
          </div>

          {canChangeStatus && (
            <div className="border-t border-slate-100 pt-3 mt-3">
              <div className="text-xs font-semibold text-slate-600 mb-1.5">Сменить статус</div>
              <div className="flex flex-wrap gap-1.5">
                {statuses.map((s) => (
                  <button
                    key={s}
                    onClick={() => onStatusChange(s)}
                    disabled={request.status === s}
                    className={
                      "text-xs rounded-full border px-3 py-1 transition " +
                      (request.status === s
                        ? "bg-slate-100 text-slate-500 border-slate-200 cursor-not-allowed"
                        : "bg-white text-slate-700 border-slate-300 hover:border-accent")
                    }
                  >
                    {STATUS_LABELS[s]}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="text-xs text-slate-500 min-w-[100px]">{label}:</span>
      <span className="text-sm text-slate-800">{children}</span>
    </div>
  );
}
