#!/usr/bin/env bash
# Скачивает CADCraft (открытый 2D-редактор чертежей в духе AutoCAD, MIT/Apache-2.0) в tools/cadcraft/:
#   cadcraft-cli — построение чертежей командами и экспорт в DXF/DWG/PDF/SVG/PNG (macOS, универсальный бинарник);
#   web/         — статическая веб-сборка редактора (вкладка «Редактор чертежей»).
# Версия и контрольные суммы зафиксированы: проект молодой (v0.x), формат команд может меняться.
# Бинарники не хранятся в репозитории, их скачивает этот скрипт.
set -euo pipefail

VERSION="0.6.2"
CLI_ZIP="cadcraft-cli-${VERSION}-macos-universal.zip"
WEB_ZIP="cadcraft-web-${VERSION}.zip"
CLI_SHA256="2ffe9b0a0169810e3177e4f1119807e04fb4ea96efe5e3e1599cea52b25c87a1"
WEB_SHA256="9d27c10b3c9101df894e44867bba46c644f23f7b4e05bd2c743762fb49509e7b"
BASE_URL="https://github.com/storytold/cadcraft/releases/download/v${VERSION}"

DEST="$(cd "$(dirname "$0")/.." && pwd)/tools/cadcraft"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Скрипт рассчитан на macOS (CLI из релиза — универсальный бинарник для macOS)." >&2
  exit 1
fi

fetch() { # имя архива, ожидаемая сумма
  curl -fsSL --retry 3 --max-time 600 -o "$TMP/$1" "$BASE_URL/$1"
  echo "$2  $TMP/$1" | shasum -a 256 -c - >/dev/null || { echo "Контрольная сумма $1 не совпала" >&2; exit 1; }
}

fetch "$CLI_ZIP" "$CLI_SHA256"
fetch "$WEB_ZIP" "$WEB_SHA256"

rm -rf "$DEST"
mkdir -p "$DEST"
unzip -q "$TMP/$CLI_ZIP" -d "$TMP/cli"
unzip -q "$TMP/$WEB_ZIP" -d "$TMP/web"
SRC_CLI="$TMP/cli/cadcraft-cli-${VERSION}-macos-universal"
cp "$SRC_CLI/cadcraft-cli" "$DEST/cadcraft-cli"
cp "$SRC_CLI"/LICENSE-* "$DEST/"
cp -R "$TMP/web/"* "$DEST/web-src" 2>/dev/null || cp -R "$TMP/web" "$DEST/web-src"
# веб-сборка лежит внутри каталога с именем релиза — выносим содержимое в web/
WEB_DIR="$(find "$DEST/web-src" -maxdepth 2 -name index.html -print -quit | xargs -I{} dirname {})"
mv "$WEB_DIR" "$DEST/web"
rm -rf "$DEST/web-src"
chmod +x "$DEST/cadcraft-cli"
echo "$VERSION" > "$DEST/VERSION"

"$DEST/cadcraft-cli" --version
echo "Готово: $DEST"
