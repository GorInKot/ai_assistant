/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: "#0e7a5f",
          soft: "#e7f5f0",
          dark: "#0a5a45",
        },
      },
    },
  },
  plugins: [],
};
