import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During development the FastAPI backend runs on :8000 (uvicorn api.main:app).
// Proxying /api avoids CORS issues and keeps the fetch URLs relative.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': process.env.VITE_API_URL || 'http://localhost:8000',
    },
  },
})
