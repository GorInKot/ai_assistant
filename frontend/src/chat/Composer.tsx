import { useRef, useState, KeyboardEvent } from "react";

interface Props {
  onSend: (text: string) => void;
  disabled: boolean;
}

export function Composer({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-slate-200 bg-white">
      <div className="max-w-3xl mx-auto px-6 py-4">
        <div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white shadow-sm focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/30 px-3 py-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = Math.min(el.scrollHeight, 220) + "px";
            }}
            onKeyDown={handleKey}
            placeholder="Задайте вопрос ассистенту…"
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm text-slate-800 placeholder-slate-400 focus:outline-none py-1.5"
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            className="rounded-lg bg-accent text-white px-3 py-1.5 text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent-dark"
            title="Отправить (Enter)"
          >
            →
          </button>
        </div>
        <p className="mt-2 text-xs text-slate-400 text-center">
          Enter — отправить. Shift+Enter — новая строка.
        </p>
      </div>
    </div>
  );
}
