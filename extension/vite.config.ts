import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Two entry points: the popup (React UI) and the MV3 background service worker.
// public/manifest.json is copied verbatim into dist/.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "chrome120",
    rollupOptions: {
      input: {
        popup: "popup.html",
        background: "src/background.ts",
      },
      output: {
        entryFileNames: (chunk) => (chunk.name === "background" ? "background.js" : "assets/[name]-[hash].js"),
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
  },
});
