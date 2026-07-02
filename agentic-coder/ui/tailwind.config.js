/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // VSCode dark palette
        vs: {
          bg: '#1e1e1e',
          panel: '#252526',
          panel2: '#2d2d30',
          inset: '#1b1b1c',
          border: '#3c3c3c',
          'border-light': '#454545',
          accent: '#0e639c',
          'accent-hover': '#1177bb',
          text: '#cccccc',
          'text-dim': '#9d9d9d',
          muted: '#858585',
          green: '#4ec9b0',
          blue: '#569cd6',
          yellow: '#dcdcaa',
          orange: '#ce9178',
          red: '#f48771',
          purple: '#c586c0',
          tab: '#2d2d2d',
          'tab-active': '#1e1e1e',
        },
      },
      fontFamily: {
        mono: [
          '"JetBrains Mono"',
          '"Fira Code"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'monospace',
        ],
        sans: ['"Segoe UI"', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        'flash-green': {
          '0%': { backgroundColor: 'rgba(78, 201, 176, 0.35)' },
          '100%': { backgroundColor: 'transparent' },
        },
        'blink': {
          '0%, 49%': { opacity: '1' },
          '50%, 100%': { opacity: '0' },
        },
      },
      animation: {
        'flash-green': 'flash-green 1.4s ease-out',
        'blink': 'blink 1s step-end infinite',
      },
    },
  },
  plugins: [],
}
