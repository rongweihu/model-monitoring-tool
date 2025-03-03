import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  assetsInclude: ['**/*.jpg', '**/*.png', '**/*.svg'],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    open: true
  }
})
