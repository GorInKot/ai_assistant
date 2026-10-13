#!/usr/bin/env bash
# Скачивает SolidCraft (открытый параметрический CAD, MIT/Apache-2.0) в tools/solidcraft/:
#   solidcraft-cli — построение деталей из командной строки (macOS, универсальный бинарник);
#   web/           — статическая веб-сборка редактора для вкладки «Конструктор».
# Версия и контрольные суммы зафиксированы: проект молодой, формат команд меняется между релизами.
# Бинарники не хранятся в репозитории (≈100 МБ), поэтому их скачивают этим скриптом.
set -euo pipefail

VERSION="0.8.0"
CLI_ZIP="solidcraft-cli-${VERSION}-macos-universal.zip"
WEB_ZIP="solidcraft-web-${VERSION}.zip"
CLI_SHA256="10a8a9cc336421fc668dc3b1d4daa666bc76b504171131ac33f3d758ac169ef3"
WEB_SHA256="531c4b0651484add3289fae3a8f19235443b713a353abb2be84b89f4a0e74862"
BASE_URL="https://github.com/bherbruck/solidcraft/releases/download/v${VERSION}"

DEST="$(cd "$(dirname "$0")/.." && pwd)/tools/solidcraft"
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
cp "$TMP/cli/solidcraft-cli-${VERSION}-macos-universal/solidcraft-cli" "$DEST/solidcraft-cli"
cp "$TMP/cli/solidcraft-cli-${VERSION}-macos-universal/LICENSE-"* "$TMP/cli/solidcraft-cli-${VERSION}-macos-universal/NOTICE" "$DEST/"
cp -R "$TMP/web/solidcraft-web-${VERSION}" "$DEST/web"
chmod +x "$DEST/solidcraft-cli"
echo "$VERSION" > "$DEST/VERSION"

"$DEST/solidcraft-cli" --version
echo "Готово: $DEST"
