// Связь компактной панели (/widget) со страницей портала, в которую она встроена.
// Пока — только сообщения без данных («закрыть панель»), поэтому адрес получателя не важен.
// Обмен данными (содержимое страницы) пойдёт с проверкой адресов обеих сторон.

export const embedded = window.parent !== window;

export function notifyHost(type: "close"): void {
  if (!embedded) return;
  window.parent.postMessage({ source: "ai-assistant", type }, "*");
}
