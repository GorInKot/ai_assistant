import { useEffect, useState } from "react";
import { AuthPage } from "./auth/AuthPage";
import { ChatPage } from "./chat/ChatPage";
import { getToken, subscribeToken } from "./auth/store";

export function App() {
  const [token, setToken] = useState<string | null>(getToken());

  useEffect(() => {
    return subscribeToken(setToken);
  }, []);

  return token ? <ChatPage /> : <AuthPage />;
}
