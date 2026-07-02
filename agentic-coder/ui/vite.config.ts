import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Minimal Node global declaration so this config typechecks without pulling in
// @types/node (it runs in Node at build time, where `process` exists).
declare const process: { env: Record<string, string | undefined> }

// The production build is served by the FastAPI backend from `ui/dist/` at `/`,
// so the app always talks to the API with same-origin relative URLs (`/events`,
// `/project/state`, …). In dev (`npm run dev`) this proxy forwards those same
// paths to the running backend, so the exact same relative URLs work unchanged.
const API = process.env.VITE_API_TARGET || 'http://localhost:8765'
const proxy = (path: string) => ({ [path]: { target: API, changeOrigin: true, ws: false } })

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      ...proxy('/events'),
      ...proxy('/start'),
      ...proxy('/resume'),
      ...proxy('/pause'),
      ...proxy('/cancel'),
      ...proxy('/status'),
      ...proxy('/healthz'),
      ...proxy('/file'),
      ...proxy('/project'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
})
