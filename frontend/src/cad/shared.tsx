import { useEffect, useMemo } from "react";
import type { CadHistoryItem } from "../api/types";

// Общие части вкладок конструктора: история запросов, переключатели, скачивание файлов.

export const OUTCOME_LABEL: Record<CadHistoryItem["outcome"], { text: string; className: string }> = {
  built: { text: "✔ построено", className: "bg-success-soft text-success" },
  clarify: { text: "? нужно уточнить", className: "bg-info-soft text-info" },
  unsupported: { text: "⚠ пока не поддерживается", className: "bg-warning-soft text-warning" },
  error: { text: "✖ не получилось", className: "bg-danger-soft text-danger" },
};

export function formatWhen(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function HistoryPanel({
  items,
  openingId,
  onOpen,
  onRepeat,
  onDelete,
}: {
  items: CadHistoryItem[];
  openingId: number | null;
  onOpen: (item: CadHistoryItem) => void;
  onRepeat: (item: CadHistoryItem) => void;
  onDelete: (item: CadHistoryItem) => void;
}) {
  return (
    <aside className="w-80 shrink-0 border-l border-line bg-surface flex flex-col min-h-0">
      <div className="px-4 py-3 border-b border-line text-xs font-semibold text-ink-muted">
        История запросов
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {items.length === 0 && (
          <p className="text-xs text-ink-faint px-2 py-4 text-center">
            Здесь будут ваши запросы к конструктору.
          </p>
        )}
        {items.map((item) => {
          const label = OUTCOME_LABEL[item.outcome];
          return (
            <div key={item.id} className="rounded-lg border border-line p-2.5 text-xs">
              <div className="flex items-center gap-2 mb-1">
                <span className={`rounded-full px-2 py-0.5 font-medium ${label.className}`}>{label.text}</span>
                <span className="text-ink-faint ml-auto">{formatWhen(item.created_at)}</span>
              </div>
              <div className="text-ink-soft line-clamp-3" title={item.description}>
                {item.description}
              </div>
              {item.detail && item.outcome !== "built" && (
                <div className="mt-1 text-ink-subtle line-clamp-2" title={item.detail}>
                  {item.detail}
                </div>
              )}
              <div className="mt-2 flex items-center gap-3">
                {item.can_rebuild && (
                  <button
                    onClick={() => onOpen(item)}
                    disabled={openingId === item.id}
                    className="text-primary-strong hover:underline disabled:opacity-50"
                  >
                    {openingId === item.id ? "Собираю…" : "Открыть"}
                  </button>
                )}
                <button onClick={() => onRepeat(item)} className="text-primary-strong hover:underline">
                  Повторить
                </button>
                <button
                  onClick={() => onDelete(item)}
                  className="ml-auto text-ink-faint hover:text-danger"
                  title="Удалить из истории"
                  aria-label="Удалить из истории"
                >
                  ✕
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}

export function TabButton({
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
        "text-xs rounded-full border px-3 py-1 transition " +
        (active
          ? "bg-primary text-primary-fg border-primary"
          : "bg-surface text-ink-muted border-line-strong hover:border-primary")
      }
    >
      {children}
    </button>
  );
}

export function base64ToBlob(data: string, mime: string): Blob {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

// Object-URL для скачивания; освобождается при смене данных и размонтировании.
export function useBlobUrl(make: () => Blob | null, deps: unknown[]): string | null {
  const url = useMemo(() => {
    const blob = make();
    return blob ? URL.createObjectURL(blob) : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(() => () => (url ? URL.revokeObjectURL(url) : undefined), [url]);
  return url;
}
