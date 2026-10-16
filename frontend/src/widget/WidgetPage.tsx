import { useEffect } from "react";
import { Composer } from "../chat/Composer";
import { MessageList } from "../chat/MessageList";
import { useChat } from "../chat/useChat";
import { embedded, notifyHost } from "./host";

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
      <MessageList messages={chat.messages} loading={chat.sending} onSuggest={chat.send} compact />
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
      <Composer onSend={chat.send} disabled={chat.sending} compact />
    </div>
  );
}

const headerBtn =
  "whitespace-nowrap rounded-md px-2 py-1 text-xs text-inverse-muted hover:text-inverse-fg hover:bg-inverse-raised disabled:opacity-50";
