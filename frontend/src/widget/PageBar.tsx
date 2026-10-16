import { useState } from "react";
import type { PageContextPreview } from "../api/types";

// Плашка над полем ввода: что ассистент видит на странице портала. Можно выключить
// и посмотреть ровно тот текст, что уйдёт модели вместе с вопросом.

interface Props {
  preview: PageContextPreview | null;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

export function PageBar({ preview, enabled, onToggle }: Props) {
  const [open, setOpen] = useState(false);

  if (!enabled) {
    return (
      <div className="px-3 pb-1 text-xs text-ink-faint flex items-center gap-2">
        <span>Страница портала не передаётся</span>
        <button onClick={() => onToggle(true)} className="underline hover:text-ink">
          включить
        </button>
      </div>
    );
  }
  if (!preview) return null;

  const details = [preview.rows ? `${preview.rows} строк` : "", preview.fields ? `${preview.fields} полей` : ""]
    .filter(Boolean)
    .join(", ");
  return (
    <div className="px-3 pb-1 text-xs">
      <div className="flex items-center gap-2 text-ink-subtle">
        <span className="truncate" title="Ассистент учитывает, что открыто на этой странице">
          👁 Видит: {preview.label}
          {details && ` · ${details}`}
        </span>
        <span className="flex-1" />
        <button onClick={() => setOpen((v) => !v)} className="shrink-0 underline hover:text-ink">
          {open ? "скрыть" : "что передаётся"}
        </button>
        <button onClick={() => onToggle(false)} className="shrink-0 underline hover:text-ink">
          выключить
        </button>
      </div>
      {open && (
        <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-line bg-surface p-2 text-[11px] leading-snug text-ink-soft">
          {preview.text}
        </pre>
      )}
    </div>
  );
}
