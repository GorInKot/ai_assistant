import { useEffect, useState } from "react";
import type { ConversationSummary } from "../api/types";

interface Props {
  conversations: ConversationSummary[];
  activeId: number | null;
  isAdmin: boolean;
  isManager: boolean;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onSelect: (id: number) => void;
  onNew: () => void;
  onRename: (id: number, title: string) => void;
  onDelete: (id: number) => void;
  onOpenProfile: () => void;
  onOpenUsersAdmin: () => void;
  onOpenGaps: () => void;
  onOpenCad: () => void;
  onOpenDrawing: () => void;
  onLogout: () => void;
}

export function Sidebar({
  conversations,
  activeId,
  isAdmin,
  isManager,
  collapsed,
  onToggleCollapsed,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onOpenProfile,
  onOpenUsersAdmin,
  onOpenGaps,
  onOpenCad,
  onOpenDrawing,
  onLogout,
}: Props) {
  // В свёрнутом режиме список бесед открывается выезжающей панелью поверх чата.
  const [historyOpen, setHistoryOpen] = useState(false);

  useEffect(() => {
    if (!historyOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setHistoryOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [historyOpen]);

  useEffect(() => {
    if (!collapsed) setHistoryOpen(false);
  }, [collapsed]);

  // Свёрнутая панель: узкая полоса с кнопками. Навигация по разделам остаётся доступной,
  // а беседы доступны по кнопке «Список бесед».
  if (collapsed) {
    return (
      <>
      <aside className="w-14 shrink-0 bg-slate-900 text-slate-100 flex flex-col items-center h-screen py-3 gap-1">
        <button
          onClick={onToggleCollapsed}
          className={railBtn}
          title="Показать историю обращений"
          aria-label="Показать историю обращений"
        >
          »
        </button>
        <button
          onClick={() => {
            setHistoryOpen(false);
            onNew();
          }}
          className={railBtn}
          title="Новая беседа"
          aria-label="Новая беседа"
        >
          +
        </button>
        <button
          onClick={() => setHistoryOpen((open) => !open)}
          className={`${railBtn} ${historyOpen ? "bg-slate-700" : ""}`}
          title="Список бесед"
          aria-label="Список бесед"
          aria-pressed={historyOpen}
        >
          💬
        </button>
        <div className="flex-1" />
        <button onClick={onOpenCad} className={railBtn} title="Конструктор деталей" aria-label="Конструктор деталей">
          🧩
        </button>
        <button onClick={onOpenDrawing} className={railBtn} title="Чертёж трассы" aria-label="Чертёж трассы">
          📐
        </button>
        {isManager && (
          <button onClick={onOpenGaps} className={railBtn} title="Пробелы в знаниях" aria-label="Пробелы в знаниях">
            📊
          </button>
        )}
        {isAdmin && (
          <button onClick={onOpenUsersAdmin} className={railBtn} title="Пользователи" aria-label="Пользователи">
            🔑
          </button>
        )}
        <button onClick={onOpenProfile} className={railBtn} title="Профиль" aria-label="Профиль">
          ⚙
        </button>
        <button onClick={onLogout} className={railBtn} title="Выйти" aria-label="Выйти">
          ⎋
        </button>
      </aside>

      {historyOpen && (
        <>
          <div className="fixed top-0 bottom-0 left-14 right-0 z-20" onClick={() => setHistoryOpen(false)} />
          <div className="fixed top-0 bottom-0 left-14 w-72 z-30 bg-slate-900 text-slate-100 border-l border-slate-800 shadow-2xl flex flex-col">
            <div className="px-4 py-3 border-b border-slate-800 text-xs font-semibold uppercase tracking-wide text-slate-400">
              История обращений
            </div>
            <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
              {conversations.length === 0 && (
                <p className="text-xs text-slate-400 px-2 py-4 text-center">Здесь будут ваши беседы.</p>
              )}
              {conversations.map((conv) => (
                <ConversationItem
                  key={conv.id}
                  conv={conv}
                  isActive={conv.id === activeId}
                  onSelect={() => {
                    setHistoryOpen(false);
                    onSelect(conv.id);
                  }}
                  onRename={(title) => onRename(conv.id, title)}
                  onDelete={() => onDelete(conv.id)}
                />
              ))}
            </div>
          </div>
        </>
      )}
      </>
    );
  }

  return (
    <aside className="w-72 shrink-0 bg-slate-900 text-slate-100 flex flex-col h-screen">
      <div className="p-3 border-b border-slate-800 flex items-center gap-2">
        <button
          onClick={onNew}
          className="flex-1 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-3 py-2 text-sm text-left flex items-center gap-2"
        >
          <span className="text-lg leading-none">+</span> Новая беседа
        </button>
        <button
          onClick={onToggleCollapsed}
          className="shrink-0 rounded-lg px-2.5 py-2 text-slate-300 hover:bg-slate-800"
          title="Свернуть историю обращений"
          aria-label="Свернуть историю обращений"
        >
          «
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {conversations.length === 0 && (
          <p className="text-xs text-slate-400 px-2 py-4 text-center">Здесь будут ваши беседы.</p>
        )}
        {conversations.map((conv) => (
          <ConversationItem
            key={conv.id}
            conv={conv}
            isActive={conv.id === activeId}
            onSelect={() => onSelect(conv.id)}
            onRename={(title) => onRename(conv.id, title)}
            onDelete={() => onDelete(conv.id)}
          />
        ))}
      </div>

      <div className="p-3 border-t border-slate-800 space-y-1">
        <button onClick={onOpenCad} className={footerBtn}>
          🧩 Конструктор деталей
        </button>
        <button onClick={onOpenDrawing} className={footerBtn}>
          📐 Чертёж трассы
        </button>
        {isManager && (
          <button onClick={onOpenGaps} className={footerBtn}>
            📊 Пробелы в знаниях
          </button>
        )}
        {isAdmin && (
          <button onClick={onOpenUsersAdmin} className={footerBtn}>
            🔑 Пользователи
          </button>
        )}
        <button onClick={onOpenProfile} className={footerBtn}>
          ⚙ Профиль
        </button>
        <button onClick={onLogout} className={footerBtn}>
          ⎋ Выйти
        </button>
      </div>
    </aside>
  );
}

function ConversationItem({
  conv,
  isActive,
  onSelect,
  onRename,
  onDelete,
}: {
  conv: ConversationSummary;
  isActive: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conv.title);

  if (editing) {
    return (
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const next = draft.trim();
          if (next && next !== conv.title) onRename(next);
          setEditing(false);
        }}
        className="px-2 py-1.5"
      >
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => setEditing(false)}
          className="w-full text-sm rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
        />
      </form>
    );
  }

  return (
    <div
      className={`group rounded-lg px-2 py-1.5 cursor-pointer flex items-center gap-1 ${
        isActive ? "bg-slate-700" : "hover:bg-slate-800"
      }`}
      onClick={onSelect}
    >
      <span className="flex-1 text-sm truncate" title={conv.title}>
        {conv.title}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setDraft(conv.title);
          setEditing(true);
        }}
        className="opacity-0 group-hover:opacity-70 hover:!opacity-100 text-xs px-1"
        title="Переименовать"
      >
        ✎
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          if (confirm(`Удалить беседу «${conv.title}»?`)) onDelete();
        }}
        className="opacity-0 group-hover:opacity-70 hover:!opacity-100 text-xs px-1"
        title="Удалить"
      >
        ✕
      </button>
    </div>
  );
}

const railBtn =
  "w-10 h-10 flex items-center justify-center rounded-lg text-lg text-slate-200 hover:bg-slate-800";

const footerBtn =
  "w-full text-left text-sm rounded-lg px-3 py-2 hover:bg-slate-800 text-slate-200";
