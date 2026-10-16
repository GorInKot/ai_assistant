import { useEffect, useState, FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { ProfileData, ProfileResponse } from "../api/types";

interface Props {
  onClose: () => void;
}

export function ProfileDialog({ onClose }: Props) {
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [divisions, setDivisions] = useState<string[]>([]);
  const [subsByDivision, setSubsByDivision] = useState<Record<string, string[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api
      .get<ProfileResponse>("/api/profile")
      .then((data) => {
        setProfile(data.profile);
        setDivisions(data.options.divisions);
        setSubsByDivision(data.options.subdivisions_by_division);
      })
      .catch((err) => setError(err?.message ?? "Не удалось загрузить профиль"));
  }, []);

  const handleSubmit = async (event: FormEvent) => {
    if (!profile) return;
    event.preventDefault();
    setError(null);
    setSaving(true);
    try {
      const updated = await api.post<{ profile: ProfileData }>("/api/profile", profile);
      setProfile(updated.profile);
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось сохранить");
    } finally {
      setSaving(false);
    }
  };

  if (!profile) {
    return (
      <ModalShell onClose={onClose} title="Профиль">
        <p className="text-sm text-ink-subtle">Загрузка…</p>
        {error && <p className="mt-2 text-sm text-danger">{error}</p>}
      </ModalShell>
    );
  }

  const subdivisionOptions = subsByDivision[profile.division] ?? [];
  const showSubdivision = profile.division !== "ЦА";

  return (
    <ModalShell onClose={onClose} title="Профиль">
      <form className="space-y-3" onSubmit={handleSubmit}>
        <Field label="ФИО">
          <input
            required
            value={profile.full_name}
            onChange={(e) => setProfile({ ...profile, full_name: e.target.value })}
            className={inputClass}
          />
        </Field>
        <Field label="Подразделение">
          <select
            value={profile.division}
            onChange={(e) =>
              setProfile({ ...profile, division: e.target.value, subdivision: "", subdivision_type: "" })
            }
            className={inputClass}
          >
            {divisions.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </Field>
        {showSubdivision && (
          <Field label="ПУ / АУП">
            <select
              required
              value={profile.subdivision}
              onChange={(e) =>
                setProfile({ ...profile, subdivision: e.target.value, subdivision_type: e.target.value })
              }
              className={inputClass}
            >
              <option value="">— выберите —</option>
              {subdivisionOptions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </Field>
        )}
        <Field label="Должность">
          <input
            required
            value={profile.job_title}
            onChange={(e) => setProfile({ ...profile, job_title: e.target.value })}
            className={inputClass}
          />
        </Field>
        <Field label="Email">
          <input value={profile.email} disabled className={inputClass + " bg-surface-sunken text-ink-subtle"} />
        </Field>
        {error && <p className="text-sm text-danger">{error}</p>}
        <div className="flex justify-end gap-2 pt-2">
          <button type="button" onClick={onClose} className={secondaryBtn}>
            Отмена
          </button>
          <button type="submit" disabled={saving} className={primaryBtn}>
            {saving ? "Сохранение…" : "Сохранить"}
          </button>
        </div>
      </form>

      <div className="pt-4 mt-4 border-t border-line">
        <PasswordChangeSection />
      </div>
    </ModalShell>
  );
}

function PasswordChangeSection() {
  const [open, setOpen] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="text-sm text-primary-strong hover:underline"
      >
        Сменить пароль
      </button>
    );
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setStatus(null);
    if (newPassword !== confirmPassword) {
      setError("Новые пароли не совпадают");
      return;
    }
    setSubmitting(true);
    try {
      await api.post("/api/profile/password", {
        current_password: currentPassword,
        new_password: newPassword,
      });
      setStatus("Пароль обновлён");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка смены пароля");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="space-y-3" onSubmit={handleSubmit}>
      <div className="text-xs font-semibold text-ink-soft">Смена пароля</div>
      <Field label="Текущий пароль">
        <input
          type="password"
          required
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
          className={inputClass}
        />
      </Field>
      <Field label="Новый пароль (мин. 6 символов)">
        <input
          type="password"
          required
          minLength={6}
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          className={inputClass}
        />
      </Field>
      <Field label="Повторите новый пароль">
        <input
          type="password"
          required
          minLength={6}
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          className={inputClass}
        />
      </Field>
      {error && <p className="text-sm text-danger">{error}</p>}
      {status && <p className="text-sm text-success">{status}</p>}
      <div className="flex justify-end gap-2">
        <button type="button" onClick={() => setOpen(false)} className={secondaryBtn}>
          Закрыть
        </button>
        <button type="submit" disabled={submitting} className={primaryBtn}>
          {submitting ? "Меняем…" : "Сменить"}
        </button>
      </div>
    </form>
  );
}

function ModalShell({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-inverse/40 px-4">
      <div className="w-full max-w-md bg-surface rounded-2xl shadow-xl border border-line">
        <header className="flex items-center justify-between px-5 py-3 border-b border-line">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          <button onClick={onClose} className="text-ink-faint hover:text-ink-soft text-lg leading-none">
            ×
          </button>
        </header>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

const inputClass =
  "w-full rounded-lg border border-line-strong px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary";
const primaryBtn =
  "btn-primary rounded-lg font-medium px-4 py-2 text-sm disabled:opacity-60";
const secondaryBtn =
  "rounded-lg border border-line-strong text-ink-soft font-medium px-4 py-2 text-sm hover:bg-surface-sunken";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-ink-muted mb-1 block">{label}</span>
      {children}
    </label>
  );
}
