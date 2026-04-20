/// <reference types="vitest" />
import { defineConfig } from "vite";

// Single source of truth for the base path. The benchmark page lives in
// benchmark/ and is the sole web entry — the project root has no index.html.
// GitHub Pages serves this repo at https://<user>.github.io/Tempo-js/
// (case-preserving path from the repo name), so `base` must match that
// exact path or all asset requests 404.
//
// `test.root = '.'` is required because the Vite root is benchmark/, which
// would otherwise scope Vitest's file discovery to benchmark/ and miss the
// project-level tests/ directory entirely.
export default defineConfig({
  root: "benchmark",
  base: "/Tempo-js/",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
  },
  test: {
    root: ".",
    include: ["tests/**/*.test.js"],
  },
});
