import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Bind to all interfaces
    port: 3000,
    strictPort: false, // Allow port fallback if 3000 is busy
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})