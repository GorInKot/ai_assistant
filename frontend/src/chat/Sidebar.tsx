import { useState } from "react";
import type { ConversationSummary } from "../api/types";

interface Props {
  conversations: ConversationSummary[];
  activeId: number | null;
  isAdmin: boolean;
  inboxUnreadCount: number;
  onSelect: (id: number) => void;
  onNew: () => void;
  onRename: (id: number, title: string) => void;
  onDelete: (id: number) => void;
  onOpenProfile: () => void;
  onOpenAdmin: () => void;
  onOpenUsersAdmin: () => void;
  onOpenRequests: () => void;
  onLogout: () => void;
}

export function Sidebar({
  conversations,
  activeId,
  isAdmin,
  inboxUnreadCount,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onOpenProfile,
  onOpenAdmin,
  onOpenUsersAdmin,
  onOpenRequests,
  onLogout,
}: Props) {
  return (
    <aside className="w-72 shrink-0 bg-slate-900 text-slate-100 flex flex-col h-screen">
      <div className="p-3 border-b border-slate-800">
        <button
          onClick={onNew}
          className="w-full rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-3 py-2 text-sm text-left flex items-center gap-2"
        >
          <span className="text-lg leading-none">+</span> Новая беседа
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
        <button onClick={onOpenRequests} className={footerBtn + " flex items-center justify-between"}>
          <span>📋 Заявки</span>
          {inboxUnreadCount > 0 && (
            <span className="text-xs rounded-full bg-accent text-white px-2 py-0.5 font-semibold">
              {inboxUnreadCount}
            </span>
          )}
        </button>
        {isAdmin && (
          <>
            <button onClick={onOpenAdmin} className={footerBtn}>
              👥 Сотрудники
            </button>
            <button onClick={onOpenUsersAdmin} className={footerBtn}>
              🔑 Пользователи
            </button>
          </>
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

const footerBtn =
  "w-full text-left text-sm rounded-lg px-3 py-2 hover:bg-slate-800 text-slate-200";
