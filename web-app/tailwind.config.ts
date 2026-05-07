import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0d6b4e',
          dark: '#095c41',
          light: '#e8f5f0',
        },
        'risk-high': '#ef4444',
        'risk-med': '#f97316',
        'risk-low': '#22c55e',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
    },
  },
  plugins: [],
} satisfies Config
