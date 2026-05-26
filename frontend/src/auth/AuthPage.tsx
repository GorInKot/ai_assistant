import { useState, FormEvent } from "react";
import { api, ApiError } from "../api/client";
import { setToken } from "./store";
import type { TokenResponse } from "../api/types";

// Подразделения дублируются с бэкендом (app/profile.py). При расширении —
// синхронизировать с /api/profile options.
const DIVISIONS = ["Филиал Уфа", "Филиал Тюмень", "Филиал Красноярск", "ЦА"];

type Mode = "login" | "register";

export function AuthPage() {
  const [mode, setMode] = useState<Mode>("login");

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-xl border border-slate-200 p-8">
        <header className="mb-6 text-center">
          <h1 className="text-xl font-semibold text-slate-800">Корпоративный ИИ-ассистент</h1>
          <p className="text-sm text-slate-500 mt-1">
            {mode === "login" ? "Войдите, чтобы продолжить" : "Создание учётной записи"}
          </p>
        </header>

        {mode === "login" ? <LoginForm /> : <RegisterForm onRegistered={() => setMode("login")} />}

        <div className="mt-6 text-center text-sm">
          {mode === "login" ? (
            <>
              Нет аккаунта?{" "}
              <button
                type="button"
                onClick={() => setMode("register")}
                className="text-accent font-medium hover:underline"
              >
                Зарегистрироваться
              </button>
            </>
          ) : (
            <>
              Уже есть аккаунт?{" "}
              <button
                type="button"
                onClick={() => setMode("login")}
                className="text-accent font-medium hover:underline"
              >
                Войти
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const data = await api.post<TokenResponse>("/api/auth/login", { email, password });
      setToken(data.access_token);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка входа");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <Field label="Email">
        <input
          type="email"
          required
          autoFocus
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className={inputClass}
        />
      </Field>
      <Field label="Пароль">
        <input
          type="password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className={inputClass}
        />
      </Field>
      {error && <p className="text-sm text-red-600">{error}</p>}
      <button type="submit" disabled={submitting} className={primaryBtnClass}>
        {submitting ? "Вход..." : "Войти"}
      </button>
    </form>
  );
}

function RegisterForm({ onRegistered }: { onRegistered: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [lastName, setLastName] = useState("");
  const [firstName, setFirstName] = useState("");
  const [middleName, setMiddleName] = useState("");
  const [division, setDivision] = useState(DIVISIONS[0]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    if (password !== confirmPassword) {
      setError("Пароли не совпадают");
      return;
    }
    setSubmitting(true);
    try {
      await api.post<TokenResponse>("/api/auth/register", {
        email,
        password,
        confirm_password: confirmPassword,
        last_name: lastName,
        first_name: firstName,
        middle_name: middleName || null,
        division,
        subdivision: null,
      });
      onRegistered();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка регистрации");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="space-y-3" onSubmit={handleSubmit}>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Фамилия">
          <input required value={lastName} onChange={(e) => setLastName(e.target.value)} className={inputClass} />
        </Field>
        <Field label="Имя">
          <input required value={firstName} onChange={(e) => setFirstName(e.target.value)} className={inputClass} />
        </Field>
      </div>
      <Field label="Отчество (необязательно)">
        <input value={middleName} onChange={(e) => setMiddleName(e.target.value)} className={inputClass} />
      </Field>
      <Field label="Подразделение">
        <select value={division} onChange={(e) => setDivision(e.target.value)} className={inputClass}>
          {DIVISIONS.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </Field>
      <Field label="Email">
        <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} className={inputClass} />
      </Field>
      <Field label="Пароль">
        <input
          type="password"
          required
          minLength={6}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className={inputClass}
        />
      </Field>
      <Field label="Подтверждение пароля">
        <input
          type="password"
          required
          minLength={6}
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          className={inputClass}
        />
      </Field>
      {error && <p className="text-sm text-red-600">{error}</p>}
      <button type="submit" disabled={submitting} className={primaryBtnClass}>
        {submitting ? "Регистрация..." : "Создать аккаунт"}
      </button>
    </form>
  );
}

const inputClass =
  "w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent";

const primaryBtnClass =
  "w-full rounded-lg bg-accent text-white font-medium py-2.5 hover:bg-accent-dark disabled:opacity-60 disabled:cursor-not-allowed transition";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-600 mb-1 block">{label}</span>
      {children}
    </label>
  );
}
