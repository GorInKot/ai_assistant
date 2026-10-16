import type { PageContext } from "../api/types";

// Связь компактной панели (/widget) со страницей портала, в которую она встроена (widget.js).
// Адрес портала ассистент запоминает при входе из панели, только если он в PORTAL_ORIGINS
// (routers/portal_sso.py); сообщения уходят только на этот адрес и принимаются только от него
// и только от родительского окна, а widget.js принимает их только от адреса ассистента.

const HOST_KEY = "ai_assistant_host";
const PAGE_TIMEOUT_MS = 1500;

function hostOrigin(): string | null {
  if (window.parent === window) return null;
  try {
    return sessionStorage.getItem(HOST_KEY);
  } catch {
    return null;
  }
}

// Панель встроена в страницу портала, и ей известен адрес этой страницы.
export const embedded = hostOrigin() !== null;

type HostMessage = { type: "close" } | { type: "open-full"; conversation: number | null };

export function notifyHost(message: HostMessage): boolean {
  const origin = hostOrigin();
  if (origin === null) return false;
  window.parent.postMessage({ source: "ai-assistant", ...message }, origin);
  return true;
}

let nextRequestId = 1;

// Содержимое страницы портала (что видит пользователь); null — нет портала или он не ответил.
export function requestPageContext(): Promise<PageContext | null> {
  const origin = hostOrigin();
  if (origin === null) return Promise.resolve(null);
  const id = nextRequestId++;
  return new Promise((resolve) => {
    const done = (value: PageContext | null) => {
      window.removeEventListener("message", onMessage);
      clearTimeout(timer);
      resolve(value);
    };
    const onMessage = (event: MessageEvent) => {
      if (event.origin !== origin || event.source !== window.parent) return;
      const data = event.data;
      if (!data || data.source !== "ai-assistant-host" || data.type !== "page-context" || data.id !== id) return;
      done(data.context && typeof data.context === "object" ? (data.context as PageContext) : null);
    };
    const timer = setTimeout(() => done(null), PAGE_TIMEOUT_MS);
    window.addEventListener("message", onMessage);
    window.parent.postMessage({ source: "ai-assistant", type: "page-context-request", id }, origin);
  });
}
