import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { CurrentUser } from "../api/types";

export function useCurrentUser() {
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .get<CurrentUser>("/api/user/profile")
      .then((data) => {
        if (!cancelled) setUser(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message ?? "Не удалось получить профиль");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const isAdmin = user?.roles.includes("admin") ?? false;
  const isManager = isAdmin || (user?.roles.includes("manager") ?? false);

  return { user, loading, error, isAdmin, isManager };
}
