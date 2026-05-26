import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: запускаем FastAPI на 8000, фронт на 5173.
// Все /api/* запросы прокидываем на FastAPI.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
