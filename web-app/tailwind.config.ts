import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0d6b4e',
          dark: '#095c41',
          mid: '#1a8a63',
          light: '#e8f5f0',
          50: '#f0faf5',
        },
        surface: '#ffffff',
        bg: '#f8fafb',
        border: '#e2e8f0',
        ink: {
          base: '#0f172a',
          secondary: '#475569',
          muted: '#64748b',   // slate-500: 4.5:1 on bg (#f8fafb) — WCAG AA
        },
        risk: {
          high: '#b91c1c',   // red-700: 5.6:1 on risk-high/10 bg — WCAG AA
          med:  '#92400e',   // amber-800: 6.3:1 on risk-med/10 bg — WCAG AA
          low:  '#15803d',   // green-700: 4.6:1 on risk-low/10 bg — WCAG AA
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Figtree', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
} satisfies Config
