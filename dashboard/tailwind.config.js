/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'panel': 'rgba(15, 15, 25, 0.85)',
        'panel-border': 'rgba(50, 50, 80, 0.3)',
        'bg-dark': '#0a0a0f',
        'bullish': '#3b82f6',
        'bearish': '#ef4444',
        'warning': '#f59e0b',
        'esoteric': '#8b5cf6',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
