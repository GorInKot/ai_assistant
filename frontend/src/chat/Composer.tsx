import { useRef, useState, KeyboardEvent } from "react";

interface Props {
  onSend: (text: string, files: File[]) => void;
  disabled: boolean;
}

const ACCEPT = ".xlsx,.docx,.doc,.pdf";
const MAX_FILES = 20;

export function Composer({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    const text = value.trim();
    if (disabled) return;
    if (!text && files.length === 0) return;
    onSend(text, files);
    setValue("");
    setFiles([]);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const next = [...files];
    for (const f of Array.from(incoming)) {
      if (next.length >= MAX_FILES) break;
      if (!next.some((x) => x.name === f.name && x.size === f.size)) next.push(f);
    }
    setFiles(next);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeFile = (idx: number) => setFiles(files.filter((_, i) => i !== idx));

  const canSend = !disabled && (value.trim().length > 0 || files.length > 0);

  return (
    <div className="border-t border-line bg-surface">
      <div className="max-w-3xl mx-auto px-6 py-4">
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {files.map((f, i) => (
              <span
                key={`${f.name}-${i}`}
                className="inline-flex items-center gap-1.5 rounded-lg bg-surface-sunken border border-line px-2 py-1 text-xs text-ink-soft"
              >
                <span>📄 {f.name}</span>
                <button
                  type="button"
                  onClick={() => removeFile(i)}
                  className="text-ink-faint hover:text-danger"
                  title="Убрать"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        <div className="flex items-end gap-2 rounded-2xl border border-line-strong bg-surface shadow-sm focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/30 px-3 py-2">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="hidden"
            onChange={(e) => addFiles(e.target.files)}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="rounded-lg px-2 py-1.5 text-lg text-ink-subtle hover:bg-surface-sunken disabled:opacity-40"
            title="Прикрепить файл (.xlsx, .docx, .doc)"
          >
            📎
          </button>
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
            placeholder="Спросите ассистента или прикрепите файл…"
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm text-ink placeholder-ink-subtle focus:outline-none py-1.5"
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            className="btn-primary rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed"
            title="Отправить (Enter)"
          >
            →
          </button>
        </div>
        <p className="mt-2 text-xs text-ink-faint text-center">
          Enter — отправить. Прикрепите .xlsx/.docx/.doc: «сделай выжимку», «сравни», «объедини».
        </p>
      </div>
    </div>
  );
}
