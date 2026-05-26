const TOKEN_KEY = "ai_assistant_token";

type Listener = (token: string | null) => void;
const listeners = new Set<Listener>();

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
  listeners.forEach((cb) => cb(token));
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  listeners.forEach((cb) => cb(null));
}

export function subscribeToken(cb: Listener): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
