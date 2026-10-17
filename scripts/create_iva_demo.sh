#!/bin/sh
# Синтетическая запись для показа конвейера; не выдавать за запись реального совещания IVA.
set -eu
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR="$PROJECT_DIR/samples"
mkdir -p "$OUTPUT_DIR"
say -v Milena -f "$PROJECT_DIR/demo/iva_meeting_script.txt" -o "$OUTPUT_DIR/iva_demo.aiff"
ffmpeg -nostdin -y -f lavfi -i 'color=c=0x17354a:s=1280x720:r=1' \
  -i "$OUTPUT_DIR/iva_demo.aiff" -shortest -c:v libx264 -pix_fmt yuv420p \
  -c:a aac -b:a 128k "$OUTPUT_DIR/iva_demo.mp4"
printf 'Готово: %s\n' "$OUTPUT_DIR/iva_demo.mp4"
