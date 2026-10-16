import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "../api/client";
import type { AskAttachment, ChatMessage, ChatSource, Grounding } from "../api/types";

interface Props {
  messages: ChatMessage[];
  loading: boolean;
  // Кнопка предложенного действия: отправить тот же файл с этой просьбой.
  onSuggest: (message: string, files: File[]) => void;
}

export function MessageList({ messages, loading, onSuggest }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-ink-faint text-sm px-6">
        <div className="text-center max-w-md">
          <h2 className="font-display text-xl font-bold text-ink mb-2">Чем могу помочь?</h2>
          <p>
            Задайте вопрос по корпоративным процессам: ЦУС, ЕКТП, лаборатория ИИ, обучение и
            медосмотр.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} busy={loading} onSuggest={onSuggest} />
        ))}
        {loading && (
          <div className="flex gap-3">
            <Avatar role="assistant" />
            <div className="flex-1 text-sm text-ink-faint pt-1.5">Печатает…</div>
          </div>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function MessageBubble({
  msg,
  busy,
  onSuggest,
}: {
  msg: ChatMessage;
  busy: boolean;
  onSuggest: (message: string, files: File[]) => void;
}) {
  const isUser = msg.role === "user";
  return (
    <div className="flex gap-3">
      <Avatar role={msg.role} />
      <div className="flex-1 min-w-0">
        <div className="text-xs font-semibold text-ink-subtle mb-1">
          {isUser ? "Вы" : "Ассистент"}
        </div>
        <div className="prose-msg text-sm text-ink whitespace-pre-wrap break-words">
          {isUser ? msg.content : <ReactMarkdown>{msg.content}</ReactMarkdown>}
        </div>
        {msg.grounding && msg.grounding.total > 0 && <GroundingBlock grounding={msg.grounding} />}
        {msg.sources.length > 0 && <SourcesList sources={msg.sources} />}
        {msg.attachment && <AttachmentDownload attachment={msg.attachment} />}
        {msg.suggestions && msg.suggestions.length > 0 && msg.files && msg.files.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {msg.suggestions.map((s) => (
              <button
                key={s.label}
                type="button"
                disabled={busy}
                onClick={() => onSuggest(s.message, msg.files ?? [])}
                className="btn-soft rounded-full px-3 py-1 text-xs font-medium disabled:opacity-50"
              >
                {s.label}
              </button>
            ))}
          </div>
        )}
        {!isUser && msg.id > 0 && <RatingButtons messageId={msg.id} initial={msg.rating ?? null} />}
      </div>
    </div>
  );
}

// Оценка ответа: «полезно / не полезно». Повторное нажатие снимает оценку.
// Оценки попадают в аналитику пробелов в знаниях (кабинет владельца базы).
function RatingButtons({ messageId, initial }: { messageId: number; initial: number | null }) {
  const [rating, setRating] = useState<number | null>(initial);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  const choose = async (value: 1 | -1) => {
    const next = rating === value ? null : value;
    const previous = rating;
    setRating(next);
    setBusy(true);
    setFailed(false);
    try {
      await api.put(`/api/messages/${messageId}/rating`, { rating: next });
    } catch {
      setRating(previous);
      setFailed(true);
    } finally {
      setBusy(false);
    }
  };

  const base = "rounded-md border px-2 py-0.5 text-xs transition disabled:opacity-50 ";
  const idle = "border-line bg-surface text-ink-subtle hover:border-line-strong";
  return (
    <div className="mt-2 flex items-center gap-2 text-xs text-ink-faint">
      <span>Ответ был полезен?</span>
      <button
        type="button"
        disabled={busy}
        aria-pressed={rating === 1}
        onClick={() => choose(1)}
        className={base + (rating === 1 ? "border-success-line bg-success-soft text-success" : idle)}
      >
        👍 Полезно
      </button>
      <button
        type="button"
        disabled={busy}
        aria-pressed={rating === -1}
        onClick={() => choose(-1)}
        className={base + (rating === -1 ? "border-warning-line bg-warning-soft text-warning" : idle)}
      >
        👎 Не полезно
      </button>
      {failed && <span className="text-danger">Не удалось сохранить оценку</span>}
    </div>
  );
}

function AttachmentDownload({ attachment }: { attachment: AskAttachment }) {
  // Скачиваем через Blob/object-URL, а не data:-URI: для бинарных файлов
  // (.xlsx) data-URI ненадёжен — байты могут повреждаться, и файл потом не
  // открывается. Blob отдаёт точные байты.
  const href = useMemo(() => {
    const binary = atob(attachment.content_base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: attachment.mime });
    return URL.createObjectURL(blob);
  }, [attachment.content_base64, attachment.mime]);

  // Освобождаем object-URL при размонтировании, чтобы не текла память.
  useEffect(() => () => URL.revokeObjectURL(href), [href]);

  return (
    <div className="mt-3">
      <a
        href={href}
        download={attachment.filename}
        className="btn-soft inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium"
      >
        ⬇️ Скачать {attachment.filename}
      </a>
    </div>
  );
}

function Avatar({ role }: { role: "user" | "assistant" }) {
  const isUser = role === "user";
  return (
    <div
      className={`w-8 h-8 shrink-0 rounded-full flex items-center justify-center text-xs font-semibold ${
        isUser ? "avatar-user" : "avatar-assistant"
      }`}
    >
      {isUser ? "Я" : "ИИ"}
    </div>
  );
}

// Проверка ответа по документам: сколько утверждений подтверждено, цитаты, спорные места.
function GroundingBlock({ grounding }: { grounding: Grounding }) {
  const alarms = grounding.unsupported.length;
  const confirmed = grounding.total - alarms;
  return (
    <div className="mt-3 space-y-2">
      <div
        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
          alarms === 0 ? "bg-success-soft text-success" : "bg-warning-soft text-warning"
        }`}
        title="Сверяются слова и числа ответа с найденными фрагментами документов. Выдуманные числа и детали проверка ловит; перепутанный смысл теми же словами — нет, поэтому важные факты сверяйте по цитатам."
      >
        {alarms === 0 ? "✔" : "⚠"} Найдено в документах: {confirmed} из {grounding.total} утверждений
      </div>

      {alarms > 0 && (
        <div className="rounded-lg border border-warning-line bg-warning-soft p-3 text-xs text-warning">
          <div className="font-semibold mb-1">Не найдено в документах — проверьте:</div>
          <ul className="list-disc pl-4 space-y-0.5">
            {grounding.unsupported.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </div>
      )}

      {grounding.citations.length > 0 && (
        <details className="rounded-lg border border-line bg-surface p-3 text-xs">
          <summary className="cursor-pointer font-semibold text-ink-muted">
            На что опирается ответ: цитаты ({grounding.citations.length})
          </summary>
          <ul className="mt-2 space-y-2.5">
            {grounding.citations.map((c, i) => (
              <li key={i}>
                <blockquote className="border-l-2 border-primary/50 pl-2.5 text-ink-soft italic">
                  «{c.quote}»
                </blockquote>
                <div className="mt-0.5 pl-2.5 text-ink-subtle">
                  {c.file_name}
                  {c.page != null ? `, стр. ${c.page}` : ""}
                  {c.section ? ` · ${c.section}` : ""}
                </div>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function SourcesList({ sources }: { sources: ChatSource[] }) {
  return (
    <div className="mt-3 rounded-lg border border-line bg-surface-muted p-3">
      <div className="text-xs font-semibold text-ink-muted mb-1.5">Источники</div>
      <ul className="space-y-1">
        {sources.map((src) => (
          <li key={src.relative_path} className="text-xs flex flex-wrap items-center gap-2">
            <span className="font-medium text-ink-soft">{src.file_name}</span>
            <a
              href={src.url}
              target="_blank"
              rel="noreferrer"
              className="text-primary-strong hover:underline"
            >
              открыть
            </a>
            <a href={src.download_url} className="text-primary-strong hover:underline">
              скачать
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
