import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { CadHistoryItem, CadStatus, RouteResult } from "../api/types";
import { base64ToBlob, HistoryPanel, TabButton, useBlobUrl } from "./shared";

interface Props {
  onBackToChat: () => void;
}

const EXAMPLES = [
  "Труба Ø219×8: 12 м, затем налево на 90° и 8 м, затем направо на 45° и 10 м, и ещё 14 м",
  "Трасса стальной трубы 114x6: 250 м, поворот направо на 60°, 120 м, поворот налево на 60°, 300 м",
  "Труба 273x8, участки 40, 25 и 30 м, на каждом стыке поворот на 90 градусов вправо",
];

type Tab = "build" | "editor";

export function DrawingPage({ onBackToChat }: Props) {
  const [status, setStatus] = useState<CadStatus | null>(null);
  const [tab, setTab] = useState<Tab>("build");
  const [description, setDescription] = useState("");
  const [result, setResult] = useState<RouteResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<CadHistoryItem[]>([]);
  const [openingId, setOpeningId] = useState<number | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.get<{ items: CadHistoryItem[] }>("/api/cad/history?kind=route");
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
      .catch((err) => setError(err instanceof ApiError ? err.message : "Не удалось проверить CADCraft"));
  }, []);

  const unavailable = status !== null && status.drawing?.available === false;

  const build = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.post<RouteResult>("/api/cad/route", { description }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось построить чертёж");
    } finally {
      setBusy(false);
      loadHistory();
    }
  };

  // Открыть запись истории: чертёж собирается заново по сохранённой схеме, без модели.
  const openFromHistory = async (item: CadHistoryItem) => {
    setDescription(item.description);
    setError(null);
    setResult(null);
    setOpeningId(item.id);
    try {
      setResult(await api.post<RouteResult>(`/api/cad/history/${item.id}/rebuild`));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось собрать чертёж заново");
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
    <div className="flex flex-col h-screen bg-surface-muted">
      <header className="border-b border-line bg-surface px-6 py-3 flex items-center gap-3">
        <button onClick={onBackToChat} className="text-sm text-ink-muted hover:text-ink">
          ← Назад в чат
        </button>
        <h1 className="font-display text-base font-bold uppercase tracking-wide text-ink">Чертёж трассы</h1>
        <div className="ml-auto flex items-center gap-2">
          <TabButton active={tab === "build"} onClick={() => setTab("build")}>
            Описание → чертёж
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
              {unavailable && (
                <div className="rounded-xl border border-warning-line bg-warning-soft p-4 text-sm text-warning">
                  CADCraft не установлен. Выполните в папке проекта{" "}
                  <code className="font-mono">scripts/fetch_cadcraft.sh</code> и перезапустите сервер.
                </div>
              )}

              <section className="rounded-xl border border-line bg-surface p-4 space-y-3">
                <label className="text-xs font-semibold text-ink-muted" htmlFor="route-description">
                  Опишите трассу трубопровода: диаметр трубы, длины участков (м) и повороты между ними
                </label>
                <textarea
                  id="route-description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                  maxLength={1000}
                  placeholder="Например: труба Ø219×8, участок 12 м, затем налево на 90° и 8 м, затем направо на 45° и 10 м"
                  className="w-full rounded-lg border border-line-strong px-3 py-2 text-sm focus:outline-none focus:border-primary"
                />
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={build}
                    disabled={busy || !description.trim() || unavailable}
                    className="btn-primary rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
                  >
                    {busy ? "Чертим…" : "Построить чертёж"}
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
                  <div className="text-xs text-info mt-1">Дополните описание выше и нажмите «Построить чертёж».</div>
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
                Модель только заполняет схему трассы, чертёж строит программа; длины участков из готового файла
                сверяются с вашим описанием. Чертёж —{" "}
                <a className="underline" href="https://github.com/storytold/cadcraft" target="_blank" rel="noreferrer">
                  CADCraft
                </a>
                {status?.drawing?.version ? ` ${status.drawing.version}` : ""} (MIT/Apache-2.0). Это{" "}
                <strong>эскиз плана трассы</strong>: без привязки к местности, отметок и пересечений; рамка и штамп
                упрощённые, соответствие ГОСТ и СПДС не заявляется.
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

function ResultView({ result }: { result: RouteResult }) {
  const pdfUrl = useBlobUrl(
    () => (result.pdf_base64 ? base64ToBlob(result.pdf_base64, "application/pdf") : null),
    [result.pdf_base64],
  );
  const dxfUrl = useBlobUrl(
    () => (result.dxf_base64 ? base64ToBlob(result.dxf_base64, "application/dxf") : null),
    [result.dxf_base64],
  );
  const dwgUrl = useBlobUrl(
    () => (result.dwg_base64 ? base64ToBlob(result.dwg_base64, "application/acad") : null),
    [result.dwg_base64],
  );
  const allOk = (result.checks ?? []).every((c) => c.ok);

  return (
    <section className="rounded-xl border border-line bg-surface p-4 space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-ink">{result.title}</h2>
        {result.attempts > 1 && (
          <span className="text-xs rounded-full bg-surface-sunken px-2 py-0.5 text-ink-subtle">
            схему пришлось исправить: попыток {result.attempts}
          </span>
        )}
      </div>

      {result.images?.preview && (
        <img
          src={`data:image/png;base64,${result.images.preview}`}
          alt="Предварительный просмотр чертежа"
          className="w-full rounded-lg border border-line"
        />
      )}

      <div className="grid md:grid-cols-2 gap-4 text-sm">
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

        <div className="space-y-3">
          {result.metrics && (
            <div>
              <div className="text-xs font-semibold text-ink-muted mb-1">Результат</div>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-ink-soft">
                <dt className="text-ink-subtle">Длина трассы</dt>
                <dd>{result.metrics.total_m} м</dd>
                <dt className="text-ink-subtle">Масштаб</dt>
                <dd>{result.metrics.scale}</dd>
                <dt className="text-ink-subtle">Участков</dt>
                <dd>{result.metrics.segments}</dd>
              </dl>
            </div>
          )}
          <div>
            <div className="text-xs font-semibold text-ink-muted mb-1">Проверка чертежа</div>
            <ul className="space-y-0.5">
              {(result.checks ?? []).map((c) => (
                <li key={c.name} className={c.ok ? "text-success" : "text-danger"}>
                  {c.ok ? "✔" : "✖"} {c.name}: {c.detail}
                </li>
              ))}
            </ul>
            {!allOk && (
              <div className="mt-1 text-xs text-danger">
                Чертёж не сошёлся с описанием — не используйте его без проверки.
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-2 pt-1">
            {pdfUrl && (
              <a
                href={pdfUrl}
                download="plan-trassy.pdf"
                className="btn-soft rounded-lg px-3 py-1.5 text-xs font-medium"
              >
                ⬇ PDF
              </a>
            )}
            {dwgUrl && (
              <a
                href={dwgUrl}
                download="plan-trassy.dwg"
                title="Формат AutoCAD, версия R2000: открывается в КОМПАС, nanoCAD, AutoCAD"
                className="rounded-lg border border-line-strong bg-surface px-3 py-1.5 text-xs text-ink-muted hover:border-primary"
              >
                ⬇ DWG (AutoCAD, КОМПАС, nanoCAD)
              </a>
            )}
            {dxfUrl && (
              <a
                href={dxfUrl}
                download="plan-trassy.dxf"
                title="Открытый обменный формат: откроет любая CAD-программа"
                className="rounded-lg border border-line-strong bg-surface px-3 py-1.5 text-xs text-ink-muted hover:border-primary"
              >
                ⬇ DXF
              </a>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function EditorTab({ status }: { status: CadStatus | null }) {
  if (status && status.drawing && !status.drawing.editor) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-ink-subtle px-6 text-center">
        Редактор не установлен. Выполните scripts/fetch_cadcraft.sh и перезапустите сервер.
      </div>
    );
  }
  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="px-6 py-2 text-xs text-ink-subtle bg-surface border-b border-line">
        Редактор чертежей CADCraft работает прямо в браузере. Скачанный на вкладке «Описание → чертёж» файл DXF можно
        открыть в нём (команда Open) и доработать вручную.
      </div>
      <iframe
        title="CADCraft"
        src="/cadcraft-editor/index.html"
        className="flex-1 w-full border-0 bg-surface"
        allow="fullscreen"
      />
    </div>
  );
}
