/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{rs,html}"],
  theme: {
    extend: {
      colors: {
        vect: {
          bg: "#07071a",
          surface: "#0d0d24",
          elevated: "#12122e",
          border: "#1e1e4a",
          violet: "#7c3aed",
          "violet-hover": "#6d28d9",
          "violet-light": "#a78bfa",
          cyan: "#06b6d4",
          "cyan-hover": "#0891b2",
          "cyan-light": "#67e8f9",
          text: "#e2e8f0",
          muted: "#94a3b8",
          subtle: "#475569",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "fade-in": "fadeIn 0.35s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
        "pulse-glow": "pulseGlow 2s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(124,58,237,0)" },
          "50%": { boxShadow: "0 0 16px 4px rgba(124,58,237,0.35)" },
        },
      },
      boxShadow: {
        "glow-violet": "0 0 24px rgba(124,58,237,0.4)",
        "glow-cyan": "0 0 24px rgba(6,182,212,0.35)",
      },
    },
  },
  plugins: [],
};
