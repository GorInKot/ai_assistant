// Связь компактной панели (/widget) со страницей портала, в которую она встроена (widget.js).
// Адрес портала ассистент запоминает при входе из панели, только если он в PORTAL_ORIGINS
// (routers/portal_sso.py); сообщения уходят только на этот адрес, а widget.js принимает их
// только от адреса ассистента.

const HOST_KEY = "ai_assistant_host";

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
