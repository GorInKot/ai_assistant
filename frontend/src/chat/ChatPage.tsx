import { useEffect, useState } from "react";
import { GapsAdmin } from "../admin/GapsAdmin";
import { formatDuration, TimingPage, useElapsed } from "../admin/TimingPage";
import { UsersAdmin } from "../admin/UsersAdmin";
import { api } from "../api/client";
import { clearToken } from "../auth/store";
import { useCurrentUser } from "../auth/useCurrentUser";
import type { TimingRun } from "../api/types";
import { Composer } from "./Composer";
import { MessageList } from "./MessageList";
import { ProfileDialog } from "./ProfileDialog";
import { Sidebar } from "./Sidebar";
import { useChat } from "./useChat";

const SIDEBAR_KEY = "ai_assistant_sidebar_collapsed";

function readSidebarCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_KEY) === "1";
  } catch {
    return false; // хранилище недоступно — панель развёрнута
  }
}

// «Открыть полностью» из панели ведёт сюда с номером беседы: /?conversation=42.
function conversationFromUrl(): number | null {
  const id = Number(new URLSearchParams(window.location.search).get("conversation"));
  return Number.isInteger(id) && id > 0 ? id : null;
}

export function ChatPage() {
  const chat = useChat();
  const { openConversation, setError } = chat;
  const [profileOpen, setProfileOpen] = useState(false);
  const [view, setView] = useState<"chat" | "admin-users" | "admin-gaps" | "timing">("chat");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(readSidebarCollapsed);
  const { isAdmin, isManager } = useCurrentUser();
  // Идущий замер времени (страница «Замер») — плашка над чатом, пока участник работает.
  const [activeRun, setActiveRun] = useState<TimingRun | null>(null);

  useEffect(() => {
    if (!isManager) return;
    api
      .get<{ active: TimingRun | null }>("/api/admin/timing/active")
      .then((data) => setActiveRun(data.active))
      .catch(() => undefined);
  }, [isManager, view]);

  useEffect(() => {
    const id = conversationFromUrl();
    if (id === null) return;
    history.replaceState(null, "", window.location.pathname);
    openConversation(id).catch((err) => setError(err?.message ?? "Не удалось открыть беседу"));
  }, [openConversation, setError]);

  const toggleSidebar = () => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(SIDEBAR_KEY, next ? "1" : "0");
      } catch {
        /* состояние не сохранится между сеансами, но панель всё равно переключится */
      }
      return next;
    });
  };

  const handleLogout = () => {
    if (confirm("Выйти из учётной записи?")) {
      clearToken();
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        conversations={chat.conversations}
        activeId={chat.activeId}
        isAdmin={isAdmin}
        isManager={isManager}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={toggleSidebar}
        onSelect={(id) => {
          setView("chat");
          openConversation(id).catch((err) => setError(err?.message));
        }}
        onNew={() => {
          setView("chat");
          chat.newConversation();
        }}
        onRename={chat.rename}
        onDelete={chat.remove}
        onOpenProfile={() => setProfileOpen(true)}
        onOpenUsersAdmin={() => setView("admin-users")}
        onOpenGaps={() => setView("admin-gaps")}
        onOpenTiming={() => setView("timing")}
        onLogout={handleLogout}
      />
      {view === "admin-users" ? (
        <main className="flex-1">
          <UsersAdmin onBackToChat={() => setView("chat")} />
        </main>
      ) : view === "timing" ? (
        <main className="flex-1">
          <TimingPage onBackToChat={() => setView("chat")} onActiveChange={setActiveRun} />
        </main>
      ) : view === "admin-gaps" ? (
        <main className="flex-1">
          <GapsAdmin onBackToChat={() => setView("chat")} />
        </main>
      ) : (
        <main className="flex-1 flex flex-col bg-surface-muted">
          <Header />
          {activeRun && <TimingBanner run={activeRun} onOpen={() => setView("timing")} />}
          <MessageList messages={chat.messages} loading={chat.sending} onSuggest={chat.send} />
          {chat.error && (
            <div className="mx-auto max-w-3xl w-full px-6">
              <div className="text-sm text-danger bg-danger-soft border border-danger-line rounded-lg px-3 py-2 mb-2">
                {chat.error}
                <button onClick={() => setError(null)} className="ml-2 underline text-danger">
                  закрыть
                </button>
              </div>
            </div>
          )}
          <Composer onSend={chat.send} disabled={chat.sending} />
        </main>
      )}
      {profileOpen && <ProfileDialog onClose={() => setProfileOpen(false)} />}
    </div>
  );
}

function Header() {
  return (
    <header className="border-b border-line bg-surface px-6 py-3">
      <h1 className="font-display text-base font-bold uppercase tracking-wide text-ink">Корпоративный ИИ-ассистент</h1>
    </header>
  );
}

function TimingBanner({ run, onOpen }: { run: TimingRun; onOpen: () => void }) {
  const elapsed = useElapsed(run.started_at);
  return (
    <div className="bg-inverse text-inverse-fg px-6 py-2 text-sm flex items-center gap-3">
      <span className="font-display text-lg font-bold tabular-nums text-primary">{formatDuration(elapsed)}</span>
      <span>
        Идёт замер: {run.participant} · {run.document} · {run.mode === "assisted" ? "с ассистентом" : "вручную"}
      </span>
      <button onClick={onOpen} className="ml-auto text-xs underline text-inverse-muted hover:text-inverse-fg">
        открыть замер
      </button>
    </div>
  );
}
