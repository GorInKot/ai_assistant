import { useCallback, useEffect, useState } from "react";
import { EmployeesAdmin } from "../admin/EmployeesAdmin";
import { UsersAdmin } from "../admin/UsersAdmin";
import { RequestsPage } from "./RequestsPage";
import type { RequestItem } from "../api/types";
import { api, ApiError } from "../api/client";
import { clearToken } from "../auth/store";
import { useCurrentUser } from "../auth/useCurrentUser";
import type {
  AskResponse,
  ChatMessage,
  ConversationDetail,
  ConversationSummary,
} from "../api/types";
import { Composer } from "./Composer";
import { MessageList } from "./MessageList";
import { ProfileDialog } from "./ProfileDialog";
import { Sidebar } from "./Sidebar";

let nextLocalId = -1;
const localId = () => nextLocalId--;

export function ChatPage() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [view, setView] = useState<"chat" | "admin-employees" | "admin-users" | "requests">("chat");
  const [error, setError] = useState<string | null>(null);
  const [inboxUnread, setInboxUnread] = useState(0);
  const { isAdmin } = useCurrentUser();

  // Лёгкий пуллинг счётчика новых входящих. Раз в 30 секунд.
  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      try {
        const items = await api.get<RequestItem[]>("/api/requests/inbox");
        if (!cancelled) {
          setInboxUnread(items.filter((r) => r.status === "new").length);
        }
      } catch {
        // молча — для счётчика 401 уже разлогинит через клиент
      }
    };
    refresh();
    const handle = setInterval(refresh, 30000);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, [view]);

  const loadConversations = useCallback(async () => {
    const data = await api.get<{ conversations: ConversationSummary[] }>("/api/conversations");
    setConversations(data.conversations);
    return data.conversations;
  }, []);

  const openConversation = useCallback(async (id: number) => {
    setActiveId(id);
    setMessages([]);
    const data = await api.get<ConversationDetail>(`/api/conversations/${id}`);
    setMessages(data.messages);
  }, []);

  useEffect(() => {
    loadConversations().catch((err) => setError(err?.message ?? "Не удалось загрузить беседы"));
  }, [loadConversations]);

  const handleNew = () => {
    setActiveId(null);
    setMessages([]);
  };

  const handleSend = async (text: string) => {
    setError(null);
    setSending(true);

    let conversationId = activeId;
    try {
      if (conversationId === null) {
        const conv = await api.post<ConversationSummary>("/api/conversations", {});
        conversationId = conv.id;
        setActiveId(conv.id);
        setConversations((prev) => [conv, ...prev]);
      }

      const userMsg: ChatMessage = {
        id: localId(),
        role: "user",
        content: text,
        sources: [],
        no_exact_match: false,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      const response = await api.post<AskResponse>("/api/ask", {
        question: text,
        conversation_id: conversationId,
      });

      const assistantMsg: ChatMessage = {
        id: localId(),
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        no_exact_match: response.no_exact_match,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Перезагрузим заголовки бесед — title мог обновиться через auto-title.
      loadConversations().catch(() => {
        /* безопасный no-op для UI */
      });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка отправки");
    } finally {
      setSending(false);
    }
  };

  const handleRename = async (id: number, title: string) => {
    try {
      const updated = await api.patch<ConversationSummary>(`/api/conversations/${id}`, { title });
      setConversations((prev) => prev.map((c) => (c.id === id ? updated : c)));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось переименовать");
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await api.delete<{ status: string }>(`/api/conversations/${id}`);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось удалить");
    }
  };

  const handleLogout = () => {
    if (confirm("Выйти из учётной записи?")) {
      clearToken();
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        isAdmin={isAdmin}
        onSelect={(id) => {
          setView("chat");
          openConversation(id).catch((err) => setError(err?.message));
        }}
        onNew={() => {
          setView("chat");
          handleNew();
        }}
        onRename={handleRename}
        onDelete={handleDelete}
        onOpenProfile={() => setProfileOpen(true)}
        onOpenAdmin={() => setView("admin-employees")}
        onOpenUsersAdmin={() => setView("admin-users")}
        onOpenRequests={() => setView("requests")}
        inboxUnreadCount={inboxUnread}
        onLogout={handleLogout}
      />
      {view === "admin-employees" ? (
        <main className="flex-1">
          <EmployeesAdmin onBackToChat={() => setView("chat")} />
        </main>
      ) : view === "admin-users" ? (
        <main className="flex-1">
          <UsersAdmin onBackToChat={() => setView("chat")} />
        </main>
      ) : view === "requests" ? (
        <main className="flex-1">
          <RequestsPage onBackToChat={() => setView("chat")} />
        </main>
      ) : (
        <main className="flex-1 flex flex-col bg-slate-50">
          <Header />
          <MessageList messages={messages} loading={sending} />
          {error && (
            <div className="mx-auto max-w-3xl w-full px-6">
              <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2 mb-2">
                {error}
                <button onClick={() => setError(null)} className="ml-2 underline text-red-600">
                  закрыть
                </button>
              </div>
            </div>
          )}
          <Composer onSend={handleSend} disabled={sending} />
        </main>
      )}
      {profileOpen && <ProfileDialog onClose={() => setProfileOpen(false)} />}
    </div>
  );
}

function Header() {
  return (
    <header className="border-b border-slate-200 bg-white px-6 py-3">
      <h1 className="text-sm font-semibold text-slate-700">Корпоративный ИИ-ассистент</h1>
    </header>
  );
}
