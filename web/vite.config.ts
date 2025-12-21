import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],

  // Tauri expects a fixed port
  server: {
    port: 5173,
    strictPort: true,
  },

  // Prevent Vite from obscuring Rust errors
  clearScreen: false,

  // Tauri env variables
  envPrefix: ['VITE_', 'TAURI_'],

  build: {
    // Tauri uses Chromium, which supports ES2021
    target: process.env.TAURI_PLATFORM === 'windows' ? 'chrome105' : 'safari15',

    // Don't minify for debug builds
    minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,

    // Source maps for debug builds
    sourcemap: !!process.env.TAURI_DEBUG,

    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          editor: ['@monaco-editor/react'],
          charts: ['recharts'],
          state: ['zustand'],
        },
      },
    },
  },
});
