/// <reference types="vitest" />
import { defineConfig } from "vite";

// Single source of truth for the base path. The benchmark page lives in
// benchmark/ and is the sole web entry — the project root has no index.html.
// For GitHub Pages deploys under https://<user>.github.io/tempo/, base must
// match the repo path so built assets resolve correctly.
//
// `test.root = '.'` is required because the Vite root is benchmark/, which
// would otherwise scope Vitest's file discovery to benchmark/ and miss the
// project-level tests/ directory entirely.
export default defineConfig({
  root: "benchmark",
  base: "/tempo/",
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
