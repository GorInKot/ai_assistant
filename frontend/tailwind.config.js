/** @type {import('tailwindcss').Config} */

// Цвета — смысловые роли из src/styles/tokens.css; сами значения в коде интерфейса не пишутся.
const role = (name) => `rgb(var(--color-${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: role("canvas"),
        surface: { DEFAULT: role("surface"), muted: role("surface-muted"), sunken: role("surface-sunken") },
        line: { DEFAULT: role("line"), strong: role("line-strong") },
        ink: {
          DEFAULT: role("ink"),
          soft: role("ink-soft"),
          muted: role("ink-muted"),
          subtle: role("ink-subtle"),
          faint: role("ink-faint"),
        },
        primary: {
          DEFAULT: role("primary"),
          hover: role("primary-hover"),
          fg: role("primary-fg"),
          soft: role("primary-soft"),
          strong: role("primary-strong"),
        },
        inverse: {
          DEFAULT: role("inverse"),
          raised: role("inverse-raised"),
          hover: role("inverse-hover"),
          line: role("inverse-line"),
          fg: role("inverse-fg"),
          muted: role("inverse-muted"),
        },
        success: { DEFAULT: role("success"), soft: role("success-soft"), line: role("success-line") },
        warning: { DEFAULT: role("warning"), soft: role("warning-soft"), line: role("warning-line") },
        danger: { DEFAULT: role("danger"), soft: role("danger-soft"), line: role("danger-line") },
        info: { DEFAULT: role("info"), soft: role("info-soft"), line: role("info-line") },
      },
      fontFamily: {
        sans: "var(--font-body)",
        display: "var(--font-display)",
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        DEFAULT: "var(--radius-md)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
        xl: "var(--radius-xl)",
        "2xl": "var(--radius-2xl)",
      },
      boxShadow: {
        pop: "var(--shadow-pop)",
      },
    },
  },
  plugins: [],
};
