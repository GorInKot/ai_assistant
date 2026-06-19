import { getToken, clearToken } from "../auth/store";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  // Только для строкового тела (JSON). Для FormData Content-Type НЕ ставим —
  // браузер сам выставит multipart/form-data с boundary, иначе сервер не
  // распарсит загрузку (422).
  if (!headers.has("Content-Type") && typeof options.body === "string") {
    headers.set("Content-Type", "application/json");
  }
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    // Токен невалиден/просрочен — выкидываем пользователя на login.
    clearToken();
    throw new ApiError(401, "Требуется авторизация");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      const d = data?.detail;
      if (typeof d === "string") {
        detail = d;
      } else if (Array.isArray(d)) {
        // FastAPI-валидация (422): массив объектов {loc, msg, type}.
        detail = d.map((e) => e?.msg ?? JSON.stringify(e)).join("; ");
      } else if (d) {
        detail = typeof d === "object" ? JSON.stringify(d) : String(d);
      }
    } catch {
      // ignore
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body !== undefined ? JSON.stringify(body) : undefined }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PUT", body: body !== undefined ? JSON.stringify(body) : undefined }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PATCH", body: body !== undefined ? JSON.stringify(body) : undefined }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  upload: <T>(path: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    // fetch сам выставит Content-Type с boundary — наш request() trigger-ит JSON-content-type
    // только если body — строка; FormData проходит мимо ветки.
    return request<T>(path, { method: "POST", body: form });
  },
  uploadMany: <T>(path: string, files: File[], fields: Record<string, string | number | null>) => {
    const form = new FormData();
    for (const [key, value] of Object.entries(fields)) {
      if (value !== null && value !== undefined) form.append(key, String(value));
    }
    for (const file of files) form.append("files", file);
    return request<T>(path, { method: "POST", body: form });
  },
};
