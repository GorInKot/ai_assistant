import { useState, FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { RequestTypeDef, RequestTypeSlot, ResponsibilityArea } from "../api/types";

interface Props {
  requestType: RequestTypeDef | null;
  areas: ResponsibilityArea[];
  onClose: () => void;
  onSaved: () => Promise<void> | void;
}

const SLOT_NAME_RE = /^[a-z0-9_]+$/;

export function RequestTypeFormDialog({ requestType, areas, onClose, onSaved }: Props) {
  const isEdit = requestType !== null;
  const [typeSlug, setTypeSlug] = useState(requestType?.type_slug ?? "");
  const [title, setTitle] = useState(requestType?.title ?? "");
  const [areaSlug, setAreaSlug] = useState(
    requestType?.responsibility_area_slug ?? areas[0]?.slug ?? "",
  );
  const [isAnonymous, setIsAnonymous] = useState(requestType?.is_anonymous ?? false);
  const [isActive, setIsActive] = useState(requestType?.is_active ?? true);
  const [triggerText, setTriggerText] = useState(
    (requestType?.trigger_keywords ?? []).join(", "),
  );
  const [examplesText, setExamplesText] = useState(
    (requestType?.examples ?? []).join("\n"),
  );
  const [slots, setSlots] = useState<RequestTypeSlot[]>(
    requestType?.slots ?? [
      { name: "", question: "", required: true },
    ],
  );
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const addSlot = () => setSlots((prev) => [...prev, { name: "", question: "", required: false }]);
  const removeSlot = (idx: number) =>
    setSlots((prev) => prev.filter((_, i) => i !== idx));
  const updateSlot = (idx: number, patch: Partial<RequestTypeSlot>) =>
    setSlots((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);

    // Локальная валидация — раньше, чем уходим на бэкенд.
    if (!SLOT_NAME_RE.test(typeSlug)) {
      setError("Slug типа: только латиница нижнего регистра, цифры и подчёркивания");
      return;
    }
    const cleanedSlots = slots
      .map((s) => ({ name: s.name.trim(), question: s.question.trim(), required: s.required }))
      .filter((s) => s.name || s.question);
    for (const s of cleanedSlots) {
      if (!s.name || !s.question) {
        setError(`Слот «${s.name || "(без имени)"}»: заполните и имя, и вопрос`);
        return;
      }
      if (!SLOT_NAME_RE.test(s.name)) {
        setError(`Имя слота «${s.name}» должно быть snake_case (a-z, 0-9, _)`);
        return;
      }
    }
    const names = cleanedSlots.map((s) => s.name);
    if (new Set(names).size !== names.length) {
      setError("Имена слотов должны быть уникальными");
      return;
    }

    const payload = {
      type_slug: typeSlug,
      title: title.trim(),
      responsibility_area_slug: areaSlug,
      is_anonymous: isAnonymous,
      is_active: isActive,
      trigger_keywords: triggerText
        .split(/[,\n]/)
        .map((t) => t.trim())
        .filter(Boolean),
      examples: examplesText
        .split(/\n/)
        .map((t) => t.trim())
        .filter(Boolean),
      slots: cleanedSlots,
    };

    setSaving(true);
    try {
      if (isEdit && requestType) {
        await api.put(`/api/admin/request-types/${requestType.type_slug}`, payload);
      } else {
        await api.post("/api/admin/request-types", payload);
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
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-xl border border-slate-200">
        <header className="flex items-center justify-between px-5 py-3 border-b border-slate-200">
          <h2 className="text-sm font-semibold text-slate-800">
            {isEdit ? `Тип заявки: ${requestType?.type_slug}` : "Новый тип заявки"}
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700 text-lg leading-none">
            ×
          </button>
        </header>

        <form className="p-5 space-y-3 max-h-[80vh] overflow-y-auto" onSubmit={handleSubmit}>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Slug" required>
              <input
                required
                value={typeSlug}
                onChange={(e) => setTypeSlug(e.target.value.toLowerCase())}
                className={inputClass + " font-mono"}
                placeholder="my_request_type"
              />
            </Field>
            <Field label="Название" required>
              <input
                required
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className={inputClass}
                placeholder="Заявка на…"
              />
            </Field>
          </div>

          <Field label="Область ответственности" required>
            <select
              value={areaSlug}
              onChange={(e) => setAreaSlug(e.target.value)}
              className={inputClass}
              required
            >
              {areas.map((a) => (
                <option key={a.slug} value={a.slug}>
                  {a.name} ({a.slug})
                </option>
              ))}
            </select>
          </Field>

          <div className="flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input type="checkbox" checked={isAnonymous} onChange={(e) => setIsAnonymous(e.target.checked)} />
              Анонимное обращение
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input type="checkbox" checked={isActive} onChange={(e) => setIsActive(e.target.checked)} />
              Активен (виден ассистенту)
            </label>
          </div>

          <Field label="Trigger-ключевые слова (через запятую)">
            <input
              value={triggerText}
              onChange={(e) => setTriggerText(e.target.value)}
              className={inputClass}
              placeholder="обучение, курс, тренинг"
            />
          </Field>

          <Field label="Примеры формулировок (по одной на строке)">
            <textarea
              value={examplesText}
              onChange={(e) => setExamplesText(e.target.value)}
              className={inputClass + " h-20 resize-y"}
              placeholder={"хочу записаться на курс\nнужно обучение по X"}
            />
          </Field>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-slate-600">Слоты формы</span>
              <button type="button" onClick={addSlot} className="text-xs text-accent hover:underline">
                + добавить слот
              </button>
            </div>
            <div className="space-y-2">
              {slots.map((s, idx) => (
                <div key={idx} className="grid grid-cols-12 gap-2 items-start border border-slate-200 rounded-lg p-2">
                  <input
                    value={s.name}
                    onChange={(e) => updateSlot(idx, { name: e.target.value.toLowerCase() })}
                    className={inputClass + " col-span-3 font-mono text-xs"}
                    placeholder="name"
                  />
                  <input
                    value={s.question}
                    onChange={(e) => updateSlot(idx, { question: e.target.value })}
                    className={inputClass + " col-span-7"}
                    placeholder="Что спросить у пользователя?"
                  />
                  <label className="col-span-1 text-xs text-slate-600 flex items-center gap-1 pt-2">
                    <input
                      type="checkbox"
                      checked={s.required}
                      onChange={(e) => updateSlot(idx, { required: e.target.checked })}
                    />
                    обяз.
                  </label>
                  <button
                    type="button"
                    onClick={() => removeSlot(idx)}
                    className="col-span-1 text-red-500 hover:text-red-700 text-sm pt-1.5"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>

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
