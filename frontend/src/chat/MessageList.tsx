import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage, ChatSource } from "../api/types";

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
        {msg.sources.length > 0 && <SourcesList sources={msg.sources} />}
      </div>
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
