import { useEffect, useState } from "react";
import { AuthPage } from "./auth/AuthPage";
import { ChatPage } from "./chat/ChatPage";
import { ToolsPage } from "./cad/ToolsPage";
import { getToken, subscribeToken } from "./auth/store";

// Разделы со своим адресом (сервер отдаёт для них тот же index.html, см. routers/system.py).
const TOOLS_PATH = "/tools";

export function App() {
  const [token, setToken] = useState<string | null>(getToken());

  useEffect(() => {
    return subscribeToken(setToken);
  }, []);

  if (!token) return <AuthPage />;
  return window.location.pathname === TOOLS_PATH ? <ToolsPage /> : <ChatPage />;
}
