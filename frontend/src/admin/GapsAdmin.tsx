import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import { getToken } from "../auth/store";
import type { CadDemand, CadDemandTopic, KnowledgeGap, KnowledgeGapsReport } from "../api/types";

interface Props {
  onBackToChat: () => void;
}

const PERIODS = [
  { days: 7, label: "7 дней" },
  { days: 30, label: "30 дней" },
  { days: 90, label: "90 дней" },
];

export function GapsAdmin({ onBackToChat }: Props) {
  const [days, setDays] = useState(30);
  const [report, setReport] = useState<KnowledgeGapsReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  const load = useCallback(async (period: number) => {
    setLoading(true);
    setError(null);
    try {
      setReport(await api.get<KnowledgeGapsReport>(`/api/admin/knowledge-gaps?days=${period}`));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось построить отчёт");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(days).catch(() => undefined);
  }, [days, load]);

  // Excel скачивается через fetch: ссылке нужен заголовок авторизации.
  const exportXlsx = async () => {
    setExporting(true);
    setError(null);
    try {
      const response = await fetch(`/api/admin/knowledge-gaps/export?days=${days}`, {
        headers: { Authorization: `Bearer ${getToken() ?? ""}` },
      });
      if (!response.ok) throw new Error(response.statusText);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "knowledge_gaps.xlsx";
      link.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("Не удалось выгрузить Excel");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-surface-muted">
      <header className="border-b border-line bg-surface px-6 py-3 flex items-center gap-3">
        <button onClick={onBackToChat} className="text-sm text-ink-muted hover:text-ink">
          ← Назад в чат
        </button>
        <h1 className="font-display text-base font-bold uppercase tracking-wide text-ink">Пробелы в знаниях</h1>
        <div className="ml-auto flex items-center gap-2">
          {PERIODS.map((p) => (
            <button
              key={p.days}
              onClick={() => setDays(p.days)}
              className={
                "text-xs rounded-full border px-3 py-1 transition " +
                (days === p.days
                  ? "bg-primary text-primary-fg border-primary"
                  : "bg-surface text-ink-muted border-line-strong hover:border-primary")
              }
            >
              {p.label}
            </button>
          ))}
          <button
            onClick={exportXlsx}
            disabled={exporting || loading}
            className="btn-soft text-xs rounded-lg px-3 py-1 font-medium disabled:opacity-50"
          >
            {exporting ? "Готовлю…" : "⬇ Excel"}
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
          {error && (
            <div className="text-sm text-danger bg-danger-soft border border-danger-line rounded-lg px-3 py-2">
              {error}
              <button onClick={() => setError(null)} className="ml-2 underline">
                закрыть
              </button>
            </div>
          )}

          {loading && !report ? (
            <p className="text-sm text-ink-subtle">Строю отчёт…</p>
          ) : (
            report && <Report report={report} />
          )}
        </div>
      </div>
    </div>
  );
}

function Report({ report }: { report: KnowledgeGapsReport }) {
  const s = report.summary;
  if (s.questions === 0) {
    return (
      <div className="rounded-xl border border-line bg-surface p-8 text-center text-sm text-ink-subtle">
        За выбранный период вопросов к базе знаний ещё не было. Отчёт появится, когда сотрудники
        начнут пользоваться ассистентом.
      </div>
    );
  }

  const usefulness = s.usefulness === null ? "—" : `${Math.round(s.usefulness * 100)}%`;
  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card
          title="Вопросов к базе"
          value={String(s.questions)}
          hint={`задавали сотрудников: ${s.people}`}
        />
        <Card
          title="Без ответа"
          value={String(s.refused + s.clarifications)}
          hint={`отказов: ${s.refused} · просьб уточнить: ${s.clarifications} · ответов: ${s.answered}`}
          tone={s.refused + s.clarifications > 0 ? "warn" : undefined}
        />
        <Card
          title="Оценили «полезно»"
          value={usefulness}
          hint={s.rated > 0 ? `оценок: ${s.rated} (👍 ${s.useful} / 👎 ${s.not_useful})` : "оценок пока нет"}
        />
        <Card
          title="Есть неподтверждённое"
          value={String(s.warned)}
          hint="ответов, где часть утверждений не найдена в документах"
          tone={s.warned > 0 ? "warn" : undefined}
        />
      </div>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-ink-soft">Чего не хватает в базе</h2>
        {report.gaps.length === 0 ? (
          <div className="rounded-xl border border-line bg-surface p-4 text-sm text-ink-subtle">
            Проблемных тем нет: нет отказов, неразрешённых просьб уточнить и ответов с оценкой «не полезно».
          </div>
        ) : (
          <ul className="space-y-3">
            {report.gaps.map((gap, i) => (
              <GapCard key={i} gap={gap} />
            ))}
          </ul>
        )}
        <p className="text-xs text-ink-faint">
          Похожие вопросы объединены{" "}
          {report.grouping === "embeddings" ? "по смыслу (эмбеддинги)" : "по общим словам"}.
          Считаются разные сотрудники, а не повторы одного вопроса.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-ink-soft">Документы</h2>
        <div className="rounded-xl border border-line bg-surface p-4 space-y-4">
          <div>
            <div className="text-xs font-semibold text-ink-muted mb-1">
              Ответы по этим документам чаще оценивают как «не полезно»
            </div>
            {report.documents.weak.length === 0 ? (
              <p className="text-xs text-ink-subtle">Таких документов нет.</p>
            ) : (
              <ul className="space-y-1 text-sm">
                {report.documents.weak.map((d) => (
                  <li key={d.relative_path} className="flex flex-wrap gap-x-3">
                    <span className="font-medium text-ink">{d.file_name}</span>
                    <span className="text-xs text-ink-subtle">
                      в ответах: {d.used} · 👍 {d.useful} · 👎 {d.not_useful}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <div className="text-xs font-semibold text-ink-muted mb-1">
              Не использовались в ответах за период
            </div>
            {!report.documents.unused_checked ? (
              <p className="text-xs text-ink-subtle">
                Мало вопросов, чтобы судить об этом: нужно не меньше 20.
              </p>
            ) : report.documents.unused.length === 0 ? (
              <p className="text-xs text-ink-subtle">Каждый документ хотя бы раз участвовал в ответе.</p>
            ) : (
              <details>
                <summary className="cursor-pointer text-xs text-primary-strong">
                  показать ({report.documents.unused.length})
                </summary>
                <ul className="mt-1 list-disc pl-5 text-sm text-ink-soft space-y-0.5">
                  {report.documents.unused.map((d) => (
                    <li key={d.relative_path} title={d.relative_path}>
                      {d.file_name}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </div>
      </section>

      {report.cad && <CadDemandSection cad={report.cad} />}
    </>
  );
}

// Конструктор деталей: что просили построить, но это пока не поддерживается.
function CadDemandSection({ cad }: { cad: CadDemand }) {
  return (
    <section className="space-y-3">
      <h2 className="text-sm font-semibold text-ink-soft">
        Конструктор деталей: что просили построить, но это пока не поддерживается
      </h2>
      {cad.requests === 0 ? (
        <div className="rounded-xl border border-line bg-surface p-4 text-sm text-ink-subtle">
          За выбранный период запросов к конструктору не было.
        </div>
      ) : (
        <>
          <p className="text-xs text-ink-subtle">
            Запросов к конструктору: {cad.requests} · построено: {cad.built} · «пока не поддерживается»:{" "}
            {cad.unsupported} · нужно уточнить: {cad.clarify} · не получилось: {cad.errors}
          </p>
          {cad.topics.length === 0 ? (
            <div className="rounded-xl border border-line bg-surface p-4 text-sm text-ink-subtle">
              Запросов, которые конструктор не умеет, не было.
            </div>
          ) : (
            <ul className="space-y-3">
              {cad.topics.map((topic, i) => (
                <CadTopicCard key={i} topic={topic} />
              ))}
            </ul>
          )}
          <p className="text-xs text-ink-faint">
            Темы объединены {cad.grouping === "embeddings" ? "по смыслу (эмбеддинги)" : "по общим словам"}.
            Это подсказка, что добавить в конструктор следующим: чем больше разных сотрудников просили одно и то
            же, тем важнее.
          </p>
        </>
      )}
    </section>
  );
}

function CadTopicCard({ topic }: { topic: CadDemandTopic }) {
  const refused = topic.kind === "refused";
  return (
    <li className="rounded-xl border border-line bg-surface p-4">
      <div className="flex flex-wrap items-start gap-2">
        <span
          className={
            "text-xs rounded-full px-2.5 py-0.5 font-medium " +
            (refused ? "bg-warning-soft text-warning" : "bg-surface-sunken text-ink-soft")
          }
        >
          {refused ? "Отказ: конструктор не умеет" : "Деталь построена без этого"}
        </span>
        <span className="text-xs text-ink-subtle pt-0.5">
          {topic.people} {pluralPeople(topic.people)} · запросов: {topic.requests}
          {topic.divisions.length > 0 ? ` · ${topic.divisions.join(", ")}` : ""}
        </span>
      </div>
      <div className="mt-2 text-sm font-medium text-ink">{topic.topic}</div>
      {topic.examples.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-primary-strong">
            как просили ({topic.examples.length})
          </summary>
          <ul className="mt-1 list-disc pl-5 text-xs text-ink-muted space-y-0.5">
            {topic.examples.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}

function Card({
  title,
  value,
  hint,
  tone,
}: {
  title: string;
  value: string;
  hint: string;
  tone?: "warn";
}) {
  return (
    <div
      className={
        "rounded-xl border bg-surface p-4 " + (tone === "warn" ? "border-warning-line" : "border-line")
      }
    >
      <div className="text-xs text-ink-subtle">{title}</div>
      <div className="text-2xl font-semibold text-ink mt-1">{value}</div>
      <div className="text-xs text-ink-faint mt-1">{hint}</div>
    </div>
  );
}

function GapCard({ gap }: { gap: KnowledgeGap }) {
  const missing = gap.problem === "no_document";
  return (
    <li className="rounded-xl border border-line bg-surface p-4">
      <div className="flex flex-wrap items-start gap-2">
        <span
          className={
            "text-xs rounded-full px-2.5 py-0.5 font-medium " +
            (missing ? "bg-danger-soft text-danger" : "bg-warning-soft text-warning")
          }
        >
          {gap.problem_label}
        </span>
        <span className="text-xs text-ink-subtle pt-0.5">
          {gap.people} {pluralPeople(gap.people)} · обращений: {gap.requests}
          {gap.not_useful > 0 ? ` · 👎 ${gap.not_useful}` : ""}
          {gap.divisions.length > 0 ? ` · ${gap.divisions.join(", ")}` : ""}
        </span>
      </div>
      <div className="mt-2 text-sm font-medium text-ink">{gap.topic}</div>

      {gap.examples.length > 1 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-primary-strong">
            как спрашивали ({gap.examples.length})
          </summary>
          <ul className="mt-1 list-disc pl-5 text-xs text-ink-muted space-y-0.5">
            {gap.examples.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </details>
      )}

      {gap.sources.length > 0 && (
        <div className="mt-2 text-xs text-ink-subtle">
          {missing ? "Ближайшие документы в базе: " : "Ответ строился по: "}
          {gap.sources.map((s) => s.relative_path.split("/").pop()).join(", ")}
        </div>
      )}
      <div className="mt-2 text-xs text-ink-faint">
        {missing
          ? "Что делать: добавьте документ или раздел, отвечающий на этот вопрос."
          : "Что делать: проверьте, полно и актуально ли написан документ."}
      </div>
    </li>
  );
}

function pluralPeople(n: number): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return "сотрудник";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "сотрудника";
  return "сотрудников";
}
