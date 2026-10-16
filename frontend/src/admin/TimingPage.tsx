import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import { getToken } from "../auth/store";
import type { TimingOverview, TimingRun, TimingSummaryRow, TimingTask } from "../api/types";

interface Props {
  onBackToChat: () => void;
  onActiveChange: (run: TimingRun | null) => void;
}

// Длительность «м:сс» или «ч:мм:сс».
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

// Секунды с начала замера — обновляются каждую секунду.
export function useElapsed(startedAt: string | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!startedAt) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [startedAt]);
  return startedAt ? (now - new Date(startedAt).getTime()) / 1000 : 0;
}

// Какие результаты сверки с эталоном вносятся для задачи.
const RESULT_FIELDS: Record<TimingTask, { key: keyof TimingRun; label: string }[]> = {
  spec_to_excel: [
    { key: "sheets", label: "листов" },
    { key: "positions", label: "позиций" },
    { key: "flagged", label: "помечено «Проверить»" },
    { key: "errors_wrong", label: "неверных значений" },
    { key: "errors_missed", label: "пропущено позиций" },
    { key: "errors_extra", label: "лишних позиций" },
  ],
  compare_versions: [
    { key: "changes_missed", label: "пропущено изменений" },
    { key: "changes_false", label: "ложных изменений" },
  ],
  search: [],
};

export function TimingPage({ onBackToChat, onActiveChange }: Props) {
  const [data, setData] = useState<TimingOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<number | null>(null);
  const [form, setForm] = useState({
    participant: "",
    task: "spec_to_excel",
    document: "",
    doc_type: "clean",
    mode: "manual",
  });

  const load = useCallback(async () => {
    try {
      const overview = await api.get<TimingOverview>("/api/admin/timing");
      setData(overview);
      onActiveChange(overview.active);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить замеры");
    }
  }, [onActiveChange]);

  useEffect(() => {
    load().catch(() => undefined);
  }, [load]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось выполнить действие");
    } finally {
      setBusy(false);
    }
  };

  const start = () => act(() => api.post("/api/admin/timing/runs", form));
  const stop = (id: number, failed = false) =>
    act(async () => {
      await api.post(`/api/admin/timing/runs/${id}/stop`, { failed });
      if (!failed) setEditing(id);
    });
  const remove = (run: TimingRun) => {
    if (confirm(`Удалить замер №${run.id}?`)) act(() => api.delete(`/api/admin/timing/runs/${run.id}`));
  };

  // CSV скачивается через fetch: ссылке нужен заголовок авторизации.
  const exportCsv = async () => {
    try {
      const response = await fetch("/api/admin/timing/export", {
        headers: { Authorization: `Bearer ${getToken() ?? ""}` },
      });
      if (!response.ok) throw new Error(response.statusText);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "timing.csv";
      link.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("Не удалось выгрузить CSV");
    }
  };

  const label = (kind: "tasks" | "modes" | "doc_types", value: string) =>
    data?.options[kind].find((o) => o.value === value)?.label ?? value;

  return (
    <div className="flex flex-col h-screen bg-surface-muted">
      <header className="border-b border-line bg-surface px-6 py-3 flex items-center gap-3">
        <button onClick={onBackToChat} className="text-sm text-ink-muted hover:text-ink">
          ← Назад в чат
        </button>
        <h1 className="font-display text-base font-bold uppercase tracking-wide text-ink">Замер времени</h1>
        <button
          onClick={exportCsv}
          disabled={!data?.runs.length}
          className="btn-soft ml-auto text-xs rounded-lg px-3 py-1 font-medium disabled:opacity-50"
        >
          ⬇ CSV для Excel
        </button>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
          <p className="text-sm text-ink-muted">
            Одна и та же задача — вручную и с ассистентом, на сопоставимых документах. В режиме «с ассистентом»
            машинное время набегает само, пока вы работаете в чате. Участник — код (У1, У2), не ФИО.
          </p>

          {error && (
            <div className="text-sm text-danger bg-danger-soft border border-danger-line rounded-lg px-3 py-2">
              {error}
              <button onClick={() => setError(null)} className="ml-2 underline">
                закрыть
              </button>
            </div>
          )}

          {data?.active ? (
            <ActivePanel
              run={data.active}
              taskLabel={label("tasks", data.active.task)}
              modeLabel={label("modes", data.active.mode)}
              busy={busy}
              onStop={() => stop(data.active!.id)}
              onFail={() => stop(data.active!.id, true)}
            />
          ) : (
            data && (
              <section className="bg-surface border border-line rounded-xl p-5 space-y-4">
                <h2 className="text-sm font-semibold text-ink">Новый замер</h2>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Участник">
                    <input
                      value={form.participant}
                      onChange={(e) => setForm({ ...form, participant: e.target.value })}
                      placeholder="У1"
                      className={inputClass}
                    />
                  </Field>
                  <Field label="Документ">
                    <input
                      value={form.document}
                      onChange={(e) => setForm({ ...form, document: e.target.value })}
                      placeholder="Лист 3 / пара 2"
                      className={inputClass}
                    />
                  </Field>
                  <Field label="Задача">
                    <select
                      value={form.task}
                      onChange={(e) => setForm({ ...form, task: e.target.value })}
                      className={inputClass}
                    >
                      {data.options.tasks.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Тип документа">
                    <select
                      value={form.doc_type}
                      onChange={(e) => setForm({ ...form, doc_type: e.target.value })}
                      className={inputClass}
                    >
                      {data.options.doc_types.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </Field>
                </div>
                <div className="flex items-center gap-3">
                  {data.options.modes.map((o) => (
                    <button
                      key={o.value}
                      onClick={() => setForm({ ...form, mode: o.value })}
                      className={
                        "text-sm rounded-lg border px-4 py-2 transition " +
                        (form.mode === o.value
                          ? "bg-primary text-primary-fg border-primary font-medium"
                          : "bg-surface text-ink-muted border-line-strong hover:border-primary")
                      }
                    >
                      {o.label}
                    </button>
                  ))}
                  <button
                    onClick={start}
                    disabled={busy || !form.participant.trim() || !form.document.trim()}
                    className="btn-primary ml-auto rounded-lg px-6 py-2 text-sm font-semibold disabled:opacity-50"
                  >
                    ▶ Старт
                  </button>
                </div>
              </section>
            )
          )}

          {data && <Summary rows={data.summary} />}

          {data && data.runs.length > 0 && (
            <section className="bg-surface border border-line rounded-xl overflow-hidden">
              <h2 className="text-sm font-semibold text-ink px-5 pt-4 pb-2">Журнал замеров</h2>
              <table className="w-full text-sm">
                <thead className="text-xs text-ink-subtle">
                  <tr className="border-b border-line">
                    <th className="text-left font-medium px-5 py-2">№</th>
                    <th className="text-left font-medium py-2">Участник</th>
                    <th className="text-left font-medium py-2">Задача и документ</th>
                    <th className="text-left font-medium py-2">Режим</th>
                    <th className="text-right font-medium py-2">Всего</th>
                    <th className="text-right font-medium py-2">Машина</th>
                    <th className="text-left font-medium py-2 pl-4">Статус</th>
                    <th className="py-2 pr-5" />
                  </tr>
                </thead>
                <tbody>
                  {data.runs.map((run) => (
                    <RunRow
                      key={run.id}
                      run={run}
                      taskLabel={label("tasks", run.task)}
                      modeLabel={label("modes", run.mode)}
                      editing={editing === run.id}
                      onEdit={() => setEditing(editing === run.id ? null : run.id)}
                      onDelete={() => remove(run)}
                      onSave={(fields) =>
                        act(async () => {
                          await api.patch(`/api/admin/timing/runs/${run.id}`, fields);
                          setEditing(null);
                        })
                      }
                    />
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

function ActivePanel({
  run,
  taskLabel,
  modeLabel,
  busy,
  onStop,
  onFail,
}: {
  run: TimingRun;
  taskLabel: string;
  modeLabel: string;
  busy: boolean;
  onStop: () => void;
  onFail: () => void;
}) {
  const elapsed = useElapsed(run.started_at);
  return (
    <section className="bg-inverse text-inverse-fg rounded-xl p-6 flex items-center gap-8">
      <div className="font-display text-6xl font-bold tabular-nums text-primary">{formatDuration(elapsed)}</div>
      <div className="flex-1 space-y-1 text-sm">
        <div className="font-semibold">
          {taskLabel} · {modeLabel}
        </div>
        <div className="text-inverse-muted">
          {run.participant} · {run.document}
          {run.mode === "assisted" && <> · машинное время: {formatDuration(run.machine_seconds)}</>}
        </div>
        {run.mode === "assisted" && (
          <div className="text-inverse-muted">Перейдите в чат — замер продолжится, время ассистента запишется само.</div>
        )}
      </div>
      <div className="flex flex-col gap-2">
        <button onClick={onStop} disabled={busy} className="btn-primary rounded-lg px-6 py-2 text-sm font-semibold">
          ■ Стоп
        </button>
        <button
          onClick={onFail}
          disabled={busy}
          className="rounded-lg border border-inverse-line px-6 py-1.5 text-xs text-inverse-muted hover:bg-inverse-hover"
        >
          Сбой — не считать
        </button>
      </div>
    </section>
  );
}

function Summary({ rows }: { rows: TimingSummaryRow[] }) {
  const stat = (s: TimingSummaryRow["manual"]) =>
    s ? (
      <>
        <span className="font-semibold text-ink">{formatDuration(s.median)}</span>
        <span className="text-ink-subtle"> · n={s.n}</span>
      </>
    ) : (
      <span className="text-ink-faint">—</span>
    );
  return (
    <section className="bg-surface border border-line rounded-xl p-5">
      <h2 className="text-sm font-semibold text-ink mb-3">Сводка: медиана на единицу работы</h2>
      <table className="w-full text-sm">
        <thead className="text-xs text-ink-subtle">
          <tr className="border-b border-line">
            <th className="text-left font-medium py-2">Задача</th>
            <th className="text-left font-medium py-2">Вручную</th>
            <th className="text-left font-medium py-2">С ассистентом</th>
            <th className="text-left font-medium py-2">Экономия</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.task} className="border-b border-line last:border-0">
              <td className="py-2 text-ink">
                {r.title} <span className="text-ink-subtle">(на {r.unit})</span>
              </td>
              <td className="py-2">{stat(r.manual)}</td>
              <td className="py-2">{stat(r.assisted)}</td>
              <td className="py-2">
                {r.saved_seconds !== null && r.saved_share !== null ? (
                  <span className="font-semibold text-success">
                    {formatDuration(r.saved_seconds)} ({Math.round(r.saved_share * 100)} %)
                  </span>
                ) : (
                  <span className="text-ink-faint">нужны оба режима</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function RunRow({
  run,
  taskLabel,
  modeLabel,
  editing,
  onEdit,
  onDelete,
  onSave,
}: {
  run: TimingRun;
  taskLabel: string;
  modeLabel: string;
  editing: boolean;
  onEdit: () => void;
  onDelete: () => void;
  onSave: (fields: Record<string, number | string | null>) => void;
}) {
  const fields = RESULT_FIELDS[run.task];
  const [values, setValues] = useState<Record<string, string>>({});
  useEffect(() => {
    if (!editing) return;
    const initial: Record<string, string> = { comment: run.comment };
    for (const f of fields) initial[f.key] = run[f.key] === null ? "" : String(run[f.key]);
    setValues(initial);
  }, [editing, run, fields]);

  const save = () => {
    const out: Record<string, number | string | null> = { comment: values.comment ?? "" };
    for (const f of fields) {
      const raw = (values[f.key] ?? "").trim();
      out[f.key] = raw === "" ? null : Number(raw);
    }
    onSave(out);
  };

  const status =
    run.status === "running" ? (
      <span className="text-primary-strong font-medium">идёт</span>
    ) : run.status === "failed" ? (
      <span className="text-danger">сбой</span>
    ) : (
      <span className="text-ink-muted">завершён</span>
    );

  return (
    <>
      <tr className="border-b border-line align-top">
        <td className="px-5 py-2 text-ink-subtle">{run.id}</td>
        <td className="py-2 text-ink">{run.participant}</td>
        <td className="py-2 text-ink">
          {taskLabel}
          <div className="text-xs text-ink-subtle">{run.document}</div>
        </td>
        <td className="py-2 text-ink-muted">{modeLabel}</td>
        <td className="py-2 text-right tabular-nums text-ink">{formatDuration(run.total_seconds)}</td>
        <td className="py-2 text-right tabular-nums text-ink-muted">
          {run.mode === "assisted" ? formatDuration(run.machine_seconds) : "—"}
        </td>
        <td className="py-2 pl-4">{status}</td>
        <td className="py-2 pr-5 text-right whitespace-nowrap">
          {run.status !== "running" && (
            <button onClick={onEdit} className="text-xs text-primary-strong hover:underline mr-3">
              результаты
            </button>
          )}
          <button onClick={onDelete} className="text-xs text-ink-subtle hover:text-danger" title="Удалить">
            ✕
          </button>
        </td>
      </tr>
      {editing && (
        <tr className="border-b border-line bg-surface-muted">
          <td colSpan={8} className="px-5 py-3">
            <div className="flex flex-wrap items-end gap-3">
              {fields.map((f) => (
                <label key={f.key} className="text-xs text-ink-muted">
                  {f.label}
                  <input
                    type="number"
                    min={0}
                    value={values[f.key] ?? ""}
                    onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
                    className={inputClass + " w-28 mt-1"}
                  />
                </label>
              ))}
              <label className="text-xs text-ink-muted flex-1 min-w-[200px]">
                комментарий
                <input
                  value={values.comment ?? ""}
                  onChange={(e) => setValues({ ...values, comment: e.target.value })}
                  className={inputClass + " mt-1"}
                />
              </label>
              <button onClick={save} className="btn-primary rounded-lg px-4 py-2 text-sm font-medium">
                Сохранить
              </button>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-ink-muted mb-1 block">{label}</span>
      {children}
    </label>
  );
}

const inputClass =
  "w-full rounded-lg border border-line-strong bg-surface px-3 py-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary";
