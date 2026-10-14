import { useEffect, useMemo } from "react";
import type { CadHistoryItem } from "../api/types";

// Общие части вкладок конструктора: история запросов, переключатели, скачивание файлов.

export const OUTCOME_LABEL: Record<CadHistoryItem["outcome"], { text: string; className: string }> = {
  built: { text: "✔ построено", className: "bg-emerald-50 text-emerald-800" },
  clarify: { text: "? нужно уточнить", className: "bg-sky-50 text-sky-800" },
  unsupported: { text: "⚠ пока не поддерживается", className: "bg-amber-50 text-amber-800" },
  error: { text: "✖ не получилось", className: "bg-red-50 text-red-700" },
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
    <aside className="w-80 shrink-0 border-l border-slate-200 bg-white flex flex-col min-h-0">
      <div className="px-4 py-3 border-b border-slate-200 text-xs font-semibold text-slate-600">
        История запросов
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {items.length === 0 && (
          <p className="text-xs text-slate-400 px-2 py-4 text-center">
            Здесь будут ваши запросы к конструктору.
          </p>
        )}
        {items.map((item) => {
          const label = OUTCOME_LABEL[item.outcome];
          return (
            <div key={item.id} className="rounded-lg border border-slate-200 p-2.5 text-xs">
              <div className="flex items-center gap-2 mb-1">
                <span className={`rounded-full px-2 py-0.5 font-medium ${label.className}`}>{label.text}</span>
                <span className="text-slate-400 ml-auto">{formatWhen(item.created_at)}</span>
              </div>
              <div className="text-slate-700 line-clamp-3" title={item.description}>
                {item.description}
              </div>
              {item.detail && item.outcome !== "built" && (
                <div className="mt-1 text-slate-500 line-clamp-2" title={item.detail}>
                  {item.detail}
                </div>
              )}
              <div className="mt-2 flex items-center gap-3">
                {item.can_rebuild && (
                  <button
                    onClick={() => onOpen(item)}
                    disabled={openingId === item.id}
                    className="text-accent hover:underline disabled:opacity-50"
                  >
                    {openingId === item.id ? "Собираю…" : "Открыть"}
                  </button>
                )}
                <button onClick={() => onRepeat(item)} className="text-accent hover:underline">
                  Повторить
                </button>
                <button
                  onClick={() => onDelete(item)}
                  className="ml-auto text-slate-400 hover:text-red-600"
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
          ? "bg-accent text-white border-accent"
          : "bg-white text-slate-600 border-slate-300 hover:border-accent")
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
