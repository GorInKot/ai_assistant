import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { PageContext, PageContextPreview } from "../api/types";
import { Composer } from "../chat/Composer";
import { MessageList } from "../chat/MessageList";
import { useChat } from "../chat/useChat";
import { embedded, notifyHost, requestPageContext } from "./host";
import { PageBar } from "./PageBar";

// Компактный ассистент для панели на портале (/widget): шапка, сообщения, поле ввода.
// Без боковой панели, администрирования и замера — это есть в полной версии.
// Беседы те же, что в полной версии, и видны в её истории.

// Открытая беседа переживает переходы по страницам портала (панель загружается заново).
const CONVERSATION_KEY = "ai_assistant_widget_conversation";

function readConversation(): number | null {
  try {
    const id = Number(localStorage.getItem(CONVERSATION_KEY));
    return Number.isInteger(id) && id > 0 ? id : null;
  } catch {
    return null; // хранилище недоступно — начнём новую беседу
  }
}

// Передавать ли содержимое страницы портала вместе с вопросом (плашка «Видит: …»).
const PAGE_KEY = "ai_assistant_page_context";

function readPageEnabled(): boolean {
  try {
    return localStorage.getItem(PAGE_KEY) !== "off";
  } catch {
    return true;
  }
}

function writeConversation(id: number | null): void {
  try {
    if (id === null) localStorage.removeItem(CONVERSATION_KEY);
    else localStorage.setItem(CONVERSATION_KEY, String(id));
  } catch {
    /* беседа не переживёт переход по странице, но панель работает */
  }
}

export function WidgetPage() {
  const chat = useChat({ loadList: false });
  const { activeId, openConversation, newConversation, setError } = chat;

  useEffect(() => {
    const id = readConversation();
    if (id === null) return;
    openConversation(id).catch(() => {
      // беседу удалили в полной версии — начинаем новую без сообщения об ошибке
      writeConversation(null);
      newConversation();
      setError(null);
    });
  }, [openConversation, newConversation, setError]);

  useEffect(() => {
    writeConversation(activeId);
  }, [activeId]);

  const [pageEnabled, setPageEnabled] = useState(readPageEnabled);
  const [preview, setPreview] = useState<PageContextPreview | null>(null);

  // Что на странице сейчас: собирает widget.js, а метку и текст для модели даёт сервер.
  const readPage = useCallback(async (): Promise<{ context: PageContext; label: string } | null> => {
    const context = await requestPageContext();
    if (!context) {
      setPreview(null);
      return null;
    }
    try {
      const seen = await api.post<PageContextPreview>("/api/page-context/preview", context);
      setPreview(seen);
      return { context, label: seen.label };
    } catch {
      setPreview(null);
      return null;
    }
  }, []);

  useEffect(() => {
    if (embedded && pageEnabled) readPage();
  }, [pageEnabled, readPage]);

  const togglePage = (enabled: boolean) => {
    setPageEnabled(enabled);
    try {
      localStorage.setItem(PAGE_KEY, enabled ? "on" : "off");
    } catch {
      /* выбор не переживёт перезагрузку панели */
    }
  };

  // Перед вопросом страница читается заново — поля и фильтры могли измениться. С файлами —
  // без страницы: файлы обрабатываются отдельно.
  const send = async (text: string, files: File[] = []) => {
    const page = embedded && pageEnabled && files.length === 0 ? await readPage() : null;
    await chat.send(text, files, page);
  };

  // На портале — полная версия на плитке портала (вход тем же пропуском), иначе — новая вкладка ассистента.
  const openFull = () => {
    if (notifyHost({ type: "open-full", conversation: activeId })) return;
    const url = activeId === null ? "/" : `/?conversation=${activeId}`;
    window.open(url, "_blank", "noopener");
  };

  return (
    <div className="flex flex-col h-screen bg-surface-muted">
      <header className="bg-inverse text-inverse-fg sidebar-rule px-3 py-2 flex items-center gap-1">
        {/* на совсем узком экране название прячется, чтобы кнопки шапки уместились в строку */}
        <span className="font-display text-sm font-bold uppercase tracking-wide whitespace-nowrap max-[400px]:hidden">
          ИИ-ассистент
        </span>
        <span className="flex-1" />
        <button onClick={newConversation} disabled={chat.sending} className={headerBtn} title="Начать новую беседу">
          + Новая беседа
        </button>
        <button onClick={openFull} className={headerBtn} title="Открыть полную версию в новой вкладке">
          Открыть полностью ↗
        </button>
        {embedded && (
          <button
            onClick={() => notifyHost({ type: "close" })}
            className={`${headerBtn} text-base leading-none`}
            title="Закрыть панель"
            aria-label="Закрыть панель"
          >
            ✕
          </button>
        )}
      </header>
      <MessageList messages={chat.messages} loading={chat.sending} onSuggest={send} compact />
      {chat.error && (
        <div className="px-3">
          <div className="text-xs text-danger bg-danger-soft border border-danger-line rounded-lg px-3 py-2 mb-2">
            {chat.error}
            <button onClick={() => setError(null)} className="ml-2 underline text-danger">
              закрыть
            </button>
          </div>
        </div>
      )}
      {embedded && <PageBar preview={preview} enabled={pageEnabled} onToggle={togglePage} />}
      <Composer onSend={send} disabled={chat.sending} compact />
    </div>
  );
}

const headerBtn =
  "whitespace-nowrap rounded-md px-2 py-1 text-xs text-inverse-muted hover:text-inverse-fg hover:bg-inverse-raised disabled:opacity-50";
