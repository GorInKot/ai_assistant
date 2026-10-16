/*
 * Кнопка ИИ-ассистента для страниц портала: круглая кнопка в правом нижнем углу открывает
 * панель справа с компактным ассистентом (/widget). Без сторонних библиотек.
 *
 * Подключение — одна строка на странице портала (для вошедших пользователей):
 *   <script src="http://<ассистент>/widget.js" data-panel="/assistant/panel/"
 *           data-full="/assistant/" data-ai-assistant-widget defer></script>
 * data-panel — страница портала, которая выдаёт одноразовый пропуск и открывает /widget;
 * data-full — полная версия ассистента на портале («Открыть полностью»).
 *
 * Сообщения от панели принимаются только от адреса ассистента (откуда загружен этот скрипт)
 * и только из её окна.
 */
(function () {
  "use strict";
  // Внутри фреймов (дашборды, окно ассистента) — без кнопки; повторное подключение — не дублируем.
  if (window.top !== window || window.__aiAssistantWidget) return;
  var script = document.currentScript;
  if (!script || !script.src) return;
  window.__aiAssistantWidget = true;

  var assistantOrigin = new URL(script.src).origin;
  var panelUrl = script.getAttribute("data-panel") || "/assistant/panel/";
  var fullUrl = script.getAttribute("data-full") || "/assistant/";
  var STATE_KEY = "ai_assistant_panel_open";

  function readOpen() {
    try {
      return sessionStorage.getItem(STATE_KEY) === "1";
    } catch (e) {
      return false;
    }
  }

  function writeOpen(open) {
    try {
      sessionStorage.setItem(STATE_KEY, open ? "1" : "0");
    } catch (e) {
      /* состояние не переживёт переход по странице — панель всё равно работает */
    }
  }

  // Цвета — из темы портала (static/theme.css), если страница их задаёт.
  function themeColor(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  var CSS = [
    ":host { all: initial; }",
    ".btn { position: fixed; right: 24px; bottom: 24px; width: 56px; height: 56px; border-radius: 50%;",
    "  border: 0; cursor: pointer; display: flex; align-items: center; justify-content: center;",
    "  background: var(--gold); color: var(--black); box-shadow: 0 6px 20px rgba(0,0,0,.28);",
    "  transition: transform .15s ease, box-shadow .15s ease; z-index: 2; }",
    ".btn:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(0,0,0,.32); }",
    ".btn:focus-visible { outline: 3px solid var(--black); outline-offset: 3px; }",
    ".btn svg { width: 28px; height: 28px; }",
    ".btn[hidden] { display: none; }",
    ".panel { position: fixed; top: 0; right: 0; height: 100%; width: 420px; max-width: 100vw;",
    "  background: #fff; box-shadow: -8px 0 28px rgba(0,0,0,.22); transform: translateX(100%);",
    "  visibility: hidden; transition: transform .2s ease, visibility 0s linear .2s; z-index: 3; }",
    ".panel.open { transform: none; visibility: visible; transition: transform .2s ease; }",
    ".panel.instant { transition: none; }",
    ".panel iframe { display: block; width: 100%; height: 100%; border: 0; }",
    ".loading { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;",
    "  font: 14px Arial, sans-serif; color: #6b7280; z-index: -1; }",
    "@media (max-width: 640px) { .panel { width: 100vw; box-shadow: none; } }",
  ].join("\n");

  var ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"' +
    ' stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20.5l1.4-5A8 8 0 1 1 21 12z"/>' +
    '<path d="M8.5 11h.01M12 11h.01M15.5 11h.01"/></svg>';

  function start() {
    var host = document.createElement("div");
    host.setAttribute("data-ai-assistant", "");
    host.style.cssText = "position: fixed; z-index: 2147483000; top: 0; left: 0; width: 0; height: 0;";
    var root = host.attachShadow({ mode: "closed" });

    var style = document.createElement("style");
    style.textContent = CSS;
    root.appendChild(style);
    host.style.setProperty("--gold", themeColor("--rn-gold", "#e8a419"));
    host.style.setProperty("--black", themeColor("--rn-black", "#0b0d10"));

    var button = document.createElement("button");
    button.className = "btn";
    button.type = "button";
    button.title = "ИИ-ассистент";
    button.setAttribute("aria-label", "Открыть ИИ-ассистент");
    button.setAttribute("aria-expanded", "false");
    button.innerHTML = ICON;
    root.appendChild(button);

    var panel = document.createElement("div");
    panel.className = "panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "ИИ-ассистент");
    var loading = document.createElement("div");
    loading.className = "loading";
    loading.textContent = "Загрузка ассистента…";
    panel.appendChild(loading);
    root.appendChild(panel);

    var frame = null;

    function open(instant) {
      if (!frame) {
        // Окно создаётся при первом открытии: страница портала выдаёт пропуск и ведёт на /widget.
        frame = document.createElement("iframe");
        frame.title = "ИИ-ассистент";
        frame.src = panelUrl;
        frame.setAttribute("allow", "clipboard-write");
        panel.appendChild(frame);
      }
      panel.classList.toggle("instant", !!instant);
      panel.classList.add("open");
      button.hidden = true;
      button.setAttribute("aria-expanded", "true");
      writeOpen(true);
    }

    function close() {
      panel.classList.remove("instant");
      panel.classList.remove("open");
      button.hidden = false;
      button.setAttribute("aria-expanded", "false");
      writeOpen(false);
      button.focus();
    }

    button.addEventListener("click", function () {
      open(false);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && panel.classList.contains("open")) close();
    });

    window.addEventListener("message", function (event) {
      if (event.origin !== assistantOrigin || !frame || event.source !== frame.contentWindow) return;
      var data = event.data;
      if (!data || data.source !== "ai-assistant") return;
      if (data.type === "close") {
        close();
      } else if (data.type === "open-full") {
        var id = Number(data.conversation);
        var url = fullUrl + (Number.isInteger(id) && id > 0 ? "?conversation=" + id : "");
        window.open(url, "_blank", "noopener");
      }
    });

    document.body.appendChild(host);
    if (readOpen()) open(true);
  }

  if (document.body) start();
  else document.addEventListener("DOMContentLoaded", start);
})();
