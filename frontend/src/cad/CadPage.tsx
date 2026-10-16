import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { CadHistoryItem, CadResult, CadStatus } from "../api/types";
import { base64ToBlob, HistoryPanel, TabButton, useBlobUrl } from "./shared";

const EXAMPLES = [
  "Стальная пластина 100 на 60 мм, толщина 5 мм, четыре отверстия диаметром 6 мм по углам на расстоянии 10 мм от краёв, углы скруглить радиусом 5",
  "Фланец из нержавеющей стали диаметром 200 мм, толщина 16 мм, центральное отверстие 80 мм, восемь отверстий по 18 мм на окружности диаметром 160 мм",
  "Алюминиевая планка 300 на 40 мм, толщина 8, три отверстия диаметром 10 мм вдоль оси: первое в 50 мм от края, шаг 100 мм",
  "Две соединённые трубы под углом 60 градусов: наружный диаметр 100 мм, толщина стенки 5 мм, длина каждой трубы 500 мм",
];

type Tab = "build" | "editor";

export function CadPage() {
  const [status, setStatus] = useState<CadStatus | null>(null);
  const [tab, setTab] = useState<Tab>("build");
  const [description, setDescription] = useState("");
  const [result, setResult] = useState<CadResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<CadHistoryItem[]>([]);
  const [openingId, setOpeningId] = useState<number | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.get<{ items: CadHistoryItem[] }>("/api/cad/history");
      setHistory(data.items);
    } catch {
      /* история вторична: без неё вкладка работает */
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    api
      .get<CadStatus>("/api/cad/status")
      .then(setStatus)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Не удалось проверить SolidCraft"));
  }, []);

  const build = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.post<CadResult>("/api/cad/generate", { description }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось построить деталь");
    } finally {
      setBusy(false);
      loadHistory();
    }
  };

  // Открыть запись истории: деталь собирается заново по сохранённой спецификации, без модели.
  const openFromHistory = async (item: CadHistoryItem) => {
    setDescription(item.description);
    setError(null);
    setResult(null);
    setOpeningId(item.id);
    try {
      setResult(await api.post<CadResult>(`/api/cad/history/${item.id}/rebuild`));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось собрать деталь заново");
    } finally {
      setOpeningId(null);
    }
  };

  const removeFromHistory = async (item: CadHistoryItem) => {
    if (!confirm("Удалить запрос из истории?")) return;
    try {
      await api.delete(`/api/cad/history/${item.id}`);
      setHistory((prev) => prev.filter((h) => h.id !== item.id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось удалить запись");
    }
  };

  return (
    <div className="flex flex-col h-full bg-surface-muted">
      <header className="border-b border-line bg-surface px-6 py-3 flex items-center gap-3">
        <h1 className="font-display text-base font-bold uppercase tracking-wide text-ink">Конструктор деталей</h1>
        <div className="ml-auto flex items-center gap-2">
          <TabButton active={tab === "build"} onClick={() => setTab("build")}>
            Описание → деталь
          </TabButton>
          <TabButton active={tab === "editor"} onClick={() => setTab("editor")}>
            Редактор
          </TabButton>
        </div>
      </header>

      {tab === "editor" ? (
        <EditorTab status={status} />
      ) : (
        <div className="flex-1 flex min-h-0">
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-5xl mx-auto px-6 py-6 space-y-5">
            {status && !status.available && (
              <div className="rounded-xl border border-warning-line bg-warning-soft p-4 text-sm text-warning">
                SolidCraft не установлен. Выполните в папке проекта{" "}
                <code className="font-mono">scripts/fetch_solidcraft.sh</code> и перезапустите сервер.
              </div>
            )}

            <section className="rounded-xl border border-line bg-surface p-4 space-y-3">
              <label className="text-xs font-semibold text-ink-muted" htmlFor="cad-description">
                Опишите деталь словами: форма, размеры в мм, отверстия, материал
              </label>
              <textarea
                id="cad-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                maxLength={1000}
                placeholder="Например: стальная пластина 100 на 60 мм, толщина 5 мм, четыре отверстия диаметром 6 мм по углам"
                className="w-full rounded-lg border border-line-strong px-3 py-2 text-sm focus:outline-none focus:border-primary"
              />
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={build}
                  disabled={busy || !description.trim() || status?.available === false}
                  className="btn-primary rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  {busy ? "Строю…" : "Построить"}
                </button>
                {busy && <span className="text-xs text-ink-subtle">Обычно 5–20 секунд</span>}
              </div>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((text, i) => (
                  <button
                    key={i}
                    onClick={() => setDescription(text)}
                    className="rounded-full border border-line-strong bg-surface px-3 py-1 text-xs text-ink-muted hover:border-primary"
                  >
                    Пример {i + 1}
                  </button>
                ))}
              </div>
            </section>

            {error && (
              <div className="text-sm text-danger bg-danger-soft border border-danger-line rounded-lg px-3 py-2">
                {error}
              </div>
            )}

            {result?.clarify && (
              <div className="rounded-xl border border-info-line bg-info-soft p-4 text-sm text-info">
                <div className="font-semibold mb-1">Нужно уточнить</div>
                {result.clarify}
                <div className="text-xs text-info mt-1">Дополните описание выше и нажмите «Построить».</div>
              </div>
            )}

            {result?.unsupported && (
              <div className="rounded-xl border border-warning-line bg-warning-soft p-4 text-sm text-warning">
                <div className="font-semibold mb-1">Пока не поддерживается</div>
                {result.unsupported}
                {(result.can_build ?? []).length > 0 && (
                  <div className="mt-2 text-xs text-warning">
                    Сейчас можно построить:
                    <ul className="list-disc pl-5 mt-0.5 space-y-0.5">
                      {(result.can_build ?? []).map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {result && !result.clarify && !result.unsupported && <ResultView result={result} />}

            <p className="text-xs text-ink-faint">
              Модель только заполняет схему детали, команды CAD строит программа; размеры и объём
              результата сверяются с расчётом. Построение — открытый CAD{" "}
              <a className="underline" href="https://github.com/bherbruck/solidcraft" target="_blank" rel="noreferrer">
                SolidCraft
              </a>
              {status?.version ? ` ${status.version}` : ""} (MIT/Apache-2.0). Поддерживаются простые детали:
              плита, цилиндр или труба со сквозными отверстиями, две трубы под углом.
            </p>
          </div>
        </div>
        <HistoryPanel
          items={history}
          openingId={openingId}
          onOpen={openFromHistory}
          onRepeat={(item) => setDescription(item.description)}
          onDelete={removeFromHistory}
        />
        </div>
      )}
    </div>
  );
}

function ResultView({ result }: { result: CadResult }) {
  const stepUrl = useBlobUrl(
    () => (result.step_base64 ? base64ToBlob(result.step_base64, "model/step") : null),
    [result.step_base64],
  );
  const scriptUrl = useBlobUrl(
    () =>
      result.script
        ? new Blob([JSON.stringify(result.script, null, 2)], { type: "application/json" })
        : null,
    [result.script],
  );
  const allOk = (result.checks ?? []).every((c) => c.ok);

  return (
    <section className="rounded-xl border border-line bg-surface p-4 space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-ink">{result.title}</h2>
        {result.attempts > 1 && (
          <span className="text-xs rounded-full bg-surface-sunken px-2 py-0.5 text-ink-subtle">
            спецификацию пришлось исправить: попыток {result.attempts}
          </span>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="space-y-2">
          {Object.entries(result.images ?? {}).map(([view, data]) => (
            <img
              key={view}
              src={`data:image/png;base64,${data}`}
              alt={view === "iso" ? "Вид в изометрии" : "Вид сверху"}
              className="w-full rounded-lg border border-line"
            />
          ))}
        </div>

        <div className="space-y-3 text-sm">
          <div>
            <div className="text-xs font-semibold text-ink-muted mb-1">Как я понял описание</div>
            <ul className="list-disc pl-5 text-ink-soft space-y-0.5">
              {(result.summary ?? []).map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
            {(result.assumptions ?? []).map((text, i) => (
              <div key={i} className="mt-1 text-xs text-warning bg-warning-soft rounded px-2 py-1">
                {text}
              </div>
            ))}
          </div>

          {result.metrics && (
            <div>
              <div className="text-xs font-semibold text-ink-muted mb-1">Результат</div>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-ink-soft">
                <dt className="text-ink-subtle">Габариты</dt>
                <dd>{result.metrics.size_mm.join(" × ")} мм</dd>
                <dt className="text-ink-subtle">Объём</dt>
                <dd>{result.metrics.volume_cm3} см³</dd>
                <dt className="text-ink-subtle">Масса</dt>
                <dd>
                  {result.metrics.mass_g} г ({result.metrics.material})
                </dd>
              </dl>
            </div>
          )}

          <div>
            <div className="text-xs font-semibold text-ink-muted mb-1">Проверка по расчёту</div>
            <ul className="space-y-0.5">
              {(result.checks ?? []).map((c) => (
                <li key={c.name} className={c.ok ? "text-success" : "text-danger"}>
                  {c.ok ? "✔" : "✖"} {c.name}: {c.detail}
                </li>
              ))}
            </ul>
            {!allOk && (
              <div className="mt-1 text-xs text-danger">
                Модель не сошлась с расчётом — не используйте её без проверки.
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-2 pt-1">
            {stepUrl && (
              <a
                href={stepUrl}
                download="part.step"
                className="btn-soft rounded-lg px-3 py-1.5 text-xs font-medium"
              >
                ⬇ STEP
              </a>
            )}
            {scriptUrl && (
              <a
                href={scriptUrl}
                download="part.script.json"
                className="rounded-lg border border-line-strong bg-surface px-3 py-1.5 text-xs text-ink-muted hover:border-primary"
              >
                ⬇ Скрипт команд CAD
              </a>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function EditorTab({ status }: { status: CadStatus | null }) {
  if (status && !status.editor) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-ink-subtle px-6 text-center">
        Редактор не установлен. Выполните scripts/fetch_solidcraft.sh и перезапустите сервер.
      </div>
    );
  }
  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="px-6 py-2 text-xs text-ink-subtle bg-surface border-b border-line">
        Редактор SolidCraft работает прямо в браузере. Скачанный на вкладке «Описание → деталь» файл STEP можно
        открыть в нём и доработать вручную.
      </div>
      <iframe
        title="SolidCraft"
        src="/cad-editor/index.html"
        className="flex-1 w-full border-0 bg-surface"
        allow="fullscreen"
      />
    </div>
  );
}

