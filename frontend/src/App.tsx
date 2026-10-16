import { useEffect, useState } from "react";
import { AuthPage } from "./auth/AuthPage";
import { ChatPage } from "./chat/ChatPage";
import { ToolsPage } from "./cad/ToolsPage";
import { getToken, subscribeToken } from "./auth/store";
import { WidgetPage } from "./widget/WidgetPage";

// Разделы со своим адресом (сервер отдаёт для них тот же index.html, см. routers/system.py).
const TOOLS_PATH = "/tools";
const WIDGET_PATH = "/widget";

export function App() {
  const [token, setToken] = useState<string | null>(getToken());

  useEffect(() => {
    return subscribeToken(setToken);
  }, []);

  if (!token) return <AuthPage />;
  const path = window.location.pathname;
  if (path === WIDGET_PATH) return <WidgetPage />;
  return path === TOOLS_PATH ? <ToolsPage /> : <ChatPage />;
}
