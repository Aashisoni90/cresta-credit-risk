import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "var(--ink-950)",
          900: "var(--ink-900)",
          850: "var(--ink-850)",
          800: "var(--ink-800)",
          750: "var(--ink-750)",
          700: "var(--ink-700)",
          600: "var(--ink-600)",
          500: "var(--ink-500)",
          400: "var(--ink-400)",
          300: "var(--ink-300)",
          100: "var(--ink-100)",
          50: "var(--ink-50)",
        },
        amber: {
          DEFAULT: "var(--amber)",
          dim: "var(--amber-dim)",
          glow: "var(--amber-glow)",
        },
        model: {
          a: "var(--model-a)",
          b: "var(--model-b)",
        },
        risk: {
          low: "var(--risk-low)",
          moderate: "var(--risk-moderate)",
          high: "var(--risk-high)",
          critical: "var(--risk-critical)",
        },
        pos: "var(--positive)",
        neg: "var(--negative)",
        grid: "var(--grid)",
      },
      fontFamily: {
        display: ["var(--font-display)", "Space Grotesk", "system-ui", "sans-serif"],
        body: ["var(--font-body)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "IBM Plex Mono", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.04em" }],
      },
      boxShadow: {
        panel: "var(--shadow-panel)",
        lamp: "0 0 0 1px var(--amber-dim), 0 0 24px var(--amber-glow)",
      },
      borderRadius: {
        card: "10px",
      },
      keyframes: {
        "fill-up": {
          from: { transform: "scaleX(0)" },
          to: { transform: "scaleX(1)" },
        },
        "fade-rise": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "sever": {
          "0%": { strokeDashoffset: "0", opacity: "1" },
          "45%": { strokeDashoffset: "0", opacity: "1" },
          "55%": { strokeDashoffset: "-14", opacity: "0.35" },
          "100%": { strokeDashoffset: "-14", opacity: "0.35" },
        },
      },
      animation: {
        "fill-up": "fill-up 0.5s cubic-bezier(0.22, 1, 0.36, 1) both",
        "fade-rise": "fade-rise 0.4s ease-out both",
      },
    },
  },
  plugins: [],
};
export default config;
