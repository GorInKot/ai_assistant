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
        <p className="text-sm text-slate-500">Загрузка…</p>
        {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
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
          <input value={profile.email} disabled className={inputClass + " bg-slate-100 text-slate-500"} />
        </Field>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <div className="flex justify-end gap-2 pt-2">
          <button type="button" onClick={onClose} className={secondaryBtn}>
            Отмена
          </button>
          <button type="submit" disabled={saving} className={primaryBtn}>
            {saving ? "Сохранение…" : "Сохранить"}
          </button>
        </div>
      </form>
    </ModalShell>
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 px-4">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-xl border border-slate-200">
        <header className="flex items-center justify-between px-5 py-3 border-b border-slate-200">
          <h2 className="text-sm font-semibold text-slate-800">{title}</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700 text-lg leading-none">
            ×
          </button>
        </header>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

const inputClass =
  "w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent";
const primaryBtn =
  "rounded-lg bg-accent text-white font-medium px-4 py-2 text-sm hover:bg-accent-dark disabled:opacity-60";
const secondaryBtn =
  "rounded-lg border border-slate-300 text-slate-700 font-medium px-4 py-2 text-sm hover:bg-slate-100";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-600 mb-1 block">{label}</span>
      {children}
    </label>
  );
}
