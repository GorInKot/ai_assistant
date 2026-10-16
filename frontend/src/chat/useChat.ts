import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { AskResponse, ChatMessage, ConversationDetail, ConversationSummary, PageContext } from "../api/types";

// Беседа с ассистентом: общая для полной версии (ChatPage) и компактной панели (WidgetPage),
// поэтому беседы из панели видны в обычной истории.

let nextLocalId = -1;
const localId = () => nextLocalId--;

export function useChat({ loadList = true }: { loadList?: boolean } = {}) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    if (!loadList) return;
    loadConversations().catch((err) => setError(err?.message ?? "Не удалось загрузить беседы"));
  }, [loadList, loadConversations]);

  const newConversation = useCallback(() => {
    setActiveId(null);
    setMessages([]);
  }, []);

  // page — что открыто на странице портала (панель кнопки): уходит вместе с вопросом,
  // в истории остаётся только метка раздела.
  const send = async (
    text: string,
    files: File[] = [],
    page?: { context: PageContext; label: string } | null,
  ) => {
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

      const userContent =
        files.length > 0
          ? `${text ? text + "\n\n" : ""}📎 ${files.map((f) => f.name).join(", ")}`
          : text;
      const userMsg: ChatMessage = {
        id: localId(),
        role: "user",
        content: userContent,
        sources: [],
        no_exact_match: false,
        page_label: files.length === 0 ? page?.label ?? null : null,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      const response =
        files.length > 0
          ? await api.uploadMany<AskResponse>("/api/ask-files", files, {
              message: text,
              conversation_id: conversationId,
            })
          : await api.post<AskResponse>("/api/ask", {
              question: text,
              conversation_id: conversationId,
              page_context: page?.context ?? null,
            });

      const assistantMsg: ChatMessage = {
        id: response.message_id ?? localId(),
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        grounding: response.grounding ?? null,
        no_exact_match: response.no_exact_match,
        created_at: new Date().toISOString(),
        attachment: response.attachment ?? null,
        suggestions: response.suggestions ?? [],
        files: response.suggestions?.length ? files : undefined,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Перезагрузим заголовки бесед — title мог обновиться через auto-title.
      if (loadList) {
        loadConversations().catch(() => {
          /* безопасный no-op для UI */
        });
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка отправки");
    } finally {
      setSending(false);
    }
  };

  const rename = async (id: number, title: string) => {
    try {
      const updated = await api.patch<ConversationSummary>(`/api/conversations/${id}`, { title });
      setConversations((prev) => prev.map((c) => (c.id === id ? updated : c)));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось переименовать");
    }
  };

  const remove = async (id: number) => {
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

  return {
    conversations,
    activeId,
    messages,
    sending,
    error,
    setError,
    openConversation,
    newConversation,
    send,
    rename,
    remove,
  };
}
