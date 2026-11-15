import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Two pages: the landing page at "/" (plain HTML + /landing assets) and the app at "/lab/".
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // paths are relative to the frontend folder
      input: { main: "index.html", lab: "lab/index.html" },
    },
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000", "/ws": { target: "ws://127.0.0.1:8000", ws: true } },
  },
});
