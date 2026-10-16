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
 * и только из её окна; ответ (содержимое страницы перед вопросом) уходит только на этот адрес.
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

  var STYLE = [
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


  // ---------- Что открыто на странице (по запросу панели, перед вопросом) ----------
  // Собирается только видимое; пароли, скрытые и служебные поля — никогда. Объём ограничен.
  var LIMITS = { fields: 40, tables: 3, rows: 8, cells: 12, text: 2500, errors: 6 };
  var SECRET = /парол|password|passwd|token|токен|csrf|secret|секрет|ключ|api[-_ ]?key|otp|pin/i;
  // подписи полей и таблицы собираются отдельно — в тексте страницы их не повторяем
  var SKIP_TEXT = "script,style,noscript,template,svg,table,[role=grid],[role=table],select,textarea,option,label,nav,[data-ai-assistant],[aria-hidden=true]";

  function squash(text, limit) {
    text = (text || "").replace(/\s+/g, " ").trim();
    return limit && text.length > limit ? text.slice(0, limit - 1) + "…" : text;
  }

  function visible(el) {
    if (!el || !(el instanceof Element)) return false;
    if (el.closest("[data-ai-assistant]")) return false;
    if (typeof el.checkVisibility === "function") return el.checkVisibility({ visibilityProperty: true });
    var style = getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden" && (el.offsetParent !== null || style.position === "fixed");
  }

  function fieldLabel(el) {
    var text = "";
    if (el.id) {
      var byFor = document.querySelector('label[for="' + window.CSS.escape(el.id) + '"]');
      if (byFor) text = byFor.textContent;
    }
    if (!text && el.closest("label")) {
      var clone = el.closest("label").cloneNode(true);
      clone.querySelectorAll("input,select,textarea").forEach(function (n) { n.remove(); });
      text = clone.textContent;
    }
    text = text || el.getAttribute("aria-label") || el.getAttribute("placeholder") || el.getAttribute("title") || "";
    return squash(text, 100);
  }

  function fieldValue(el) {
    if (el.tagName === "SELECT") {
      return squash(Array.prototype.map.call(el.selectedOptions || [], function (o) { return o.textContent; }).join(", "), 200);
    }
    if (el.type === "checkbox" || el.type === "radio") return el.checked ? "да" : "нет";
    if (el.type === "file") return el.files && el.files.length ? el.files.length + " файл(ов)" : "";
    return squash(el.value, 200);
  }

  function collectFields() {
    var out = [];
    var seenRadio = {};
    document.querySelectorAll("input,select,textarea").forEach(function (el) {
      if (out.length >= LIMITS.fields || !visible(el)) return;
      var type = (el.getAttribute("type") || "").toLowerCase();
      if (type === "password" || type === "hidden" || type === "submit" || type === "button" || type === "reset" || type === "image") return;
      if (el.autocomplete && /password|one-time-code|cc-/.test(el.autocomplete)) return;
      var name = el.getAttribute("name") || "";
      var label = (fieldLabel(el) || squash(name, 60)).replace(/\s*Filter Input$/i, " (фильтр колонки)");
      if (!label || SECRET.test(label) || SECRET.test(name)) return;
      var value = fieldValue(el);
      // пустые фильтры колонок таблицы-сетки — только шум
      if (!value && el.closest("[role=grid],[role=treegrid]")) return;
      if (type === "radio") {
        if (!el.checked || seenRadio[name]) return;
        seenRadio[name] = true;
      }
      out.push({ label: label, value: value });
    });
    return out;
  }

  function cellsOf(row, selector) {
    return Array.prototype.slice.call(row.querySelectorAll(selector), 0, LIMITS.cells).map(function (c) { return squash(c.textContent, 80); });
  }

  function hasText(cells) {
    return cells.some(function (c) { return c; });
  }

  function collectTables() {
    var out = [];
    // обычные таблицы
    document.querySelectorAll("table").forEach(function (table) {
      if (out.length >= LIMITS.tables || !visible(table) || table.closest("[role=grid]")) return;
      var headRow = table.querySelector("thead tr") || table.querySelector("tr");
      var headers = headRow ? cellsOf(headRow, "th").filter(Boolean) : [];
      var bodyRows = Array.prototype.filter.call(table.querySelectorAll("tbody tr, tr"), function (tr) {
        return tr !== headRow && tr.querySelector("td") && visible(tr);
      });
      var rows = bodyRows.map(function (tr) { return cellsOf(tr, "td,th"); }).filter(hasText).slice(0, LIMITS.rows);
      if (!headers.length && !rows.length) return;
      var caption = table.querySelector("caption");
      out.push({ caption: caption ? squash(caption.textContent, 100) : "", headers: headers, rows: rows, total_rows: bodyRows.length });
    });
    // таблицы-сетки (например, AG Grid в СИЗ)
    document.querySelectorAll("[role=grid],[role=treegrid]").forEach(function (grid) {
      if (out.length >= LIMITS.tables || !visible(grid)) return;
      var headers = cellsOf(grid, "[role=columnheader]").filter(Boolean);
      var bodyRows = Array.prototype.filter.call(grid.querySelectorAll("[role=row]"), function (row) {
        return row.querySelector("[role=gridcell]") && visible(row);
      });
      var rows = bodyRows.map(function (row) { return cellsOf(row, "[role=gridcell]"); }).filter(hasText);
      // aria-rowcount считает и строки заголовка (у AG Grid — заголовок и строка фильтров)
      var headerRows = Array.prototype.filter.call(grid.querySelectorAll("[role=row]"), function (row) {
        return row.querySelector("[role=columnheader]") || !row.querySelector("[role=gridcell]");
      }).length;
      var count = Number(grid.getAttribute("aria-rowcount"));
      out.push({ caption: squash(grid.getAttribute("aria-label") || "", 100), headers: headers, rows: rows.slice(0, LIMITS.rows),
                 total_rows: count > 0 ? Math.max(rows.length, count - headerRows) : rows.length });
    });
    return out;
  }

  function collectSelected() {
    var row = document.querySelector("tr[aria-selected=true], [role=row][aria-selected=true], tr.selected, tr.is-selected, .ag-row-selected");
    return row && visible(row) ? squash(row.textContent, 500) : "";
  }

  function collectTab() {
    var tab = document.querySelector("[role=tab][aria-selected=true], [aria-current=page], .tab.active, .tabs .active, .nav-link.active");
    return tab && visible(tab) ? squash(tab.textContent, 100) : "";
  }

  function collectErrors() {
    var out = [];
    document.querySelectorAll("[role=alert], .errorlist, .error, .alert-danger, .alert-error, .messages .error, .invalid-feedback").forEach(function (el) {
      if (out.length >= LIMITS.errors || !visible(el)) return;
      var text = squash(el.textContent, 300);
      if (text && out.indexOf(text) === -1) out.push(text);
    });
    return out;
  }

  function collectText() {
    var root = document.querySelector("main, [role=main]") || document.body;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    var lines = [];
    var last = null;
    var total = 0;
    var cache = new Map();
    for (var node = walker.nextNode(); node && total < LIMITS.text; node = walker.nextNode()) {
      var parent = node.parentElement;
      var text = squash(node.nodeValue);
      if (!text || !parent || parent.closest(SKIP_TEXT)) continue;
      if (!cache.has(parent)) cache.set(parent, visible(parent));
      if (!cache.get(parent)) continue;
      var block = parent.closest("h1,h2,h3,h4,h5,h6,p,li,dt,dd,label,button,a,section,article,div") || parent;
      if (block === last) lines[lines.length - 1] += " " + text;
      else lines.push(text);
      last = block;
      total += text.length + 1;
    }
    return lines.join("\n").slice(0, LIMITS.text);
  }

  function collectPage() {
    var heading = document.querySelector("main h1, h1") || document.querySelector("main h2, h2");
    return {
      path: location.pathname,
      title: squash(document.title, 200),
      heading: heading && visible(heading) ? squash(heading.textContent, 200) : "",
      tab: collectTab(),
      fields: collectFields(),
      selected: collectSelected(),
      tables: collectTables(),
      errors: collectErrors(),
      text: collectText(),
    };
  }

  function start() {
    var host = document.createElement("div");
    host.setAttribute("data-ai-assistant", "");
    host.style.cssText = "position: fixed; z-index: 2147483000; top: 0; left: 0; width: 0; height: 0;";
    var root = host.attachShadow({ mode: "closed" });

    var style = document.createElement("style");
    style.textContent = STYLE;
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
      } else if (data.type === "page-context-request") {
        var context = null;
        try {
          context = collectPage();
        } catch (e) {
          context = null; // страница устроена необычно — вопрос уйдёт без её содержимого
        }
        frame.contentWindow.postMessage(
          { source: "ai-assistant-host", type: "page-context", id: data.id, context: context },
          assistantOrigin
        );
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
