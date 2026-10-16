import { useEffect, useState } from "react";
import { CadPage } from "./CadPage";
import { DrawingPage } from "./DrawingPage";

// Раздел «Конструктор и чертежи» (/tools): отдельно от чата, своя плитка в портале.
// Вкладка — в адресе (#drawing), чтобы ссылка вела сразу на нужную.

type Section = "cad" | "drawing";

// В портале раздел — своя плитка (окно внутри портала): ссылка «к ассистенту» там лишняя.
// Открытый из чата — отдельная страница, и ссылка ведёт обратно в чат.
const insidePortal = window.parent !== window;

function sectionFromHash(): Section {
  return window.location.hash === "#drawing" ? "drawing" : "cad";
}

export function ToolsPage() {
  const [section, setSection] = useState<Section>(sectionFromHash);

  useEffect(() => {
    const onHash = () => setSection(sectionFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const open = (next: Section) => {
    setSection(next);
    history.replaceState(null, "", next === "drawing" ? "#drawing" : window.location.pathname);
  };

  return (
    <div className="flex flex-col h-screen">
      <nav className="bg-inverse text-inverse-fg border-b border-inverse-line px-6 flex items-center gap-6">
        {!insidePortal && (
          <a href="/" className="text-sm text-inverse-muted hover:text-inverse-fg py-3">
            ← К ассистенту
          </a>
        )}
        <span className="font-display text-base font-bold uppercase tracking-wide py-3">Конструктор и чертежи</span>
        <div className="flex items-stretch gap-1 self-stretch" role="tablist">
          <SectionTab active={section === "cad"} onClick={() => open("cad")}>
            🧩 Конструктор деталей
          </SectionTab>
          <SectionTab active={section === "drawing"} onClick={() => open("drawing")}>
            📐 Чертёж трассы
          </SectionTab>
        </div>
      </nav>
      {/* Обе вкладки остаются смонтированными: введённое описание и результат не теряются при переключении. */}
      <div className={section === "cad" ? "flex-1 min-h-0" : "hidden"}>
        <CadPage />
      </div>
      <div className={section === "drawing" ? "flex-1 min-h-0" : "hidden"}>
        <DrawingPage />
      </div>
    </div>
  );
}

function SectionTab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={
        "px-3 text-sm border-b-[3px] transition " +
        (active ? "border-primary text-inverse-fg font-medium" : "border-transparent text-inverse-muted hover:text-inverse-fg")
      }
    >
      {children}
    </button>
  );
}
