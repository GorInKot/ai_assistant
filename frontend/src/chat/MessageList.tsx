import { useEffect, useMemo, useRef } from "react";
import ReactMarkdown from "react-markdown";
import type { AskAttachment, ChatMessage, ChatSource, Grounding } from "../api/types";

interface Props {
  messages: ChatMessage[];
  loading: boolean;
}

export function MessageList({ messages, loading }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400 text-sm px-6">
        <div className="text-center max-w-md">
          <h2 className="text-lg font-semibold text-slate-700 mb-2">Чем могу помочь?</h2>
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
          <MessageBubble key={msg.id} msg={msg} />
        ))}
        {loading && (
          <div className="flex gap-3">
            <Avatar role="assistant" />
            <div className="flex-1 text-sm text-slate-400 pt-1.5">Печатает…</div>
          </div>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className="flex gap-3">
      <Avatar role={msg.role} />
      <div className="flex-1 min-w-0">
        <div className="text-xs font-semibold text-slate-500 mb-1">
          {isUser ? "Вы" : "Ассистент"}
        </div>
        <div className="prose-msg text-sm text-slate-800 whitespace-pre-wrap break-words">
          {isUser ? msg.content : <ReactMarkdown>{msg.content}</ReactMarkdown>}
        </div>
        {msg.grounding && msg.grounding.total > 0 && <GroundingBlock grounding={msg.grounding} />}
        {msg.sources.length > 0 && <SourcesList sources={msg.sources} />}
        {msg.attachment && <AttachmentDownload attachment={msg.attachment} />}
      </div>
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
        className="inline-flex items-center gap-2 rounded-lg border border-accent/40 bg-accent/5 px-3 py-2 text-sm font-medium text-accent hover:bg-accent/10"
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
        isUser ? "bg-slate-200 text-slate-700" : "bg-accent text-white"
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
          alarms === 0 ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"
        }`}
        title="Сверяются слова и числа ответа с найденными фрагментами документов. Выдуманные числа и детали проверка ловит; перепутанный смысл теми же словами — нет, поэтому важные факты сверяйте по цитатам."
      >
        {alarms === 0 ? "✔" : "⚠"} Найдено в документах: {confirmed} из {grounding.total} утверждений
      </div>

      {alarms > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900">
          <div className="font-semibold mb-1">Не найдено в документах — проверьте:</div>
          <ul className="list-disc pl-4 space-y-0.5">
            {grounding.unsupported.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </div>
      )}

      {grounding.citations.length > 0 && (
        <details className="rounded-lg border border-slate-200 bg-white p-3 text-xs">
          <summary className="cursor-pointer font-semibold text-slate-600">
            На что опирается ответ: цитаты ({grounding.citations.length})
          </summary>
          <ul className="mt-2 space-y-2.5">
            {grounding.citations.map((c, i) => (
              <li key={i}>
                <blockquote className="border-l-2 border-accent/50 pl-2.5 text-slate-700 italic">
                  «{c.quote}»
                </blockquote>
                <div className="mt-0.5 pl-2.5 text-slate-500">
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
    <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="text-xs font-semibold text-slate-600 mb-1.5">Источники</div>
      <ul className="space-y-1">
        {sources.map((src) => (
          <li key={src.relative_path} className="text-xs flex flex-wrap items-center gap-2">
            <span className="font-medium text-slate-700">{src.file_name}</span>
            <a
              href={src.url}
              target="_blank"
              rel="noreferrer"
              className="text-accent hover:underline"
            >
              открыть
            </a>
            <a href={src.download_url} className="text-accent hover:underline">
              скачать
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
