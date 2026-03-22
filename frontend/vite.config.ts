/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Use pre-built browser bundle to avoid Node.js builtin imports
      'plotly.js': 'plotly.js-dist-min',
    },
  },
  optimizeDeps: {
    // Force Vite to pre-bundle Plotly (5MB) instead of transforming on every request
    include: ['plotly.js-dist-min', 'react-plotly.js/factory'],
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['src/__tests__/**/*.test.ts'],
  },
})
