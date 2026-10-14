#!/usr/bin/env python3
"""Эскиз плана трассы трубопровода из командной строки (проба; см. app/services/route_drawing.py).

Участки задаются по порядку: «ДЛИНА» или «ДЛИНА@ПОВОРОТ». Длина в метрах, поворот перед участком
в градусах относительно прежнего направления: положительный — влево, отрицательный — вправо.
У первого участка поворот не нужен. Результат — файлы PDF, DXF, DWG (R2000) и PNG в выбранной папке.

Примеры:
  python scripts/draw_route.py 12 8@90 10@-45 14@-45
  python scripts/draw_route.py --diameter 114 --wall 6 --title "Трасса Т-1" 250 120@-60 300@60 --open
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import route_drawing as rd  # noqa: E402


def parse_segment(text: str, index: int) -> rd.Segment:
    length, _, turn = text.replace(",", ".").partition("@")
    try:
        return rd.Segment(length_m=float(length), turn_deg=float(turn) if turn else 0.0)
    except ValueError:
        raise SystemExit(f"Участок {index}: не понимаю «{text}». Нужно «ДЛИНА» или «ДЛИНА@ПОВОРОТ», например 12 или 8@-90.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Эскиз плана трассы трубопровода", formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=__doc__.split("Примеры:")[1].strip() and "Примеры:" + __doc__.split("Примеры:")[1])
    parser.add_argument("segments", nargs="+", help="участки: ДЛИНА или ДЛИНА@ПОВОРОТ (метры, градусы)")
    parser.add_argument("--title", default="План трассы трубопровода", help="название в основной надписи")
    parser.add_argument("--diameter", type=float, default=219, help="наружный диаметр трубы, мм (по умолчанию 219)")
    parser.add_argument("--wall", type=float, default=8, help="толщина стенки, мм (по умолчанию 8)")
    parser.add_argument("--material", default="сталь", help="материал трубы (по умолчанию «сталь»)")
    parser.add_argument("--out", default=str(ROOT / "tools" / "cadcraft" / "demo"), help="папка для результата")
    parser.add_argument("--name", default="moya_trassa", help="имя файлов без расширения")
    parser.add_argument("--open", action="store_true", help="открыть PDF после построения (macOS)")
    args = parser.parse_args()

    spec = rd.RouteSpec(title=args.title, diameter_mm=args.diameter, wall_mm=args.wall, material=args.material,
                        segments=[parse_segment(text, i) for i, text in enumerate(args.segments, start=1)])
    try:
        files = rd.draw_route(spec, ROOT / "tools" / "cadcraft")
    except rd.CadcraftUnavailable as exc:
        print(f"{exc}\nУстановка: ./scripts/fetch_cadcraft.sh", file=sys.stderr)
        return 2
    except rd.CadcraftError as exc:
        print(f"Не получилось: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for extension, data in (("pdf", files.pdf), ("dxf", files.dxf), ("dwg", files.dwg), ("png", files.png)):
        (out / f"{args.name}.{extension}").write_bytes(data)

    print(f"Масштаб 1:{files.info['scale']:g}, длина трассы {files.info['total_m']:.2f} м, участков {files.info['segments']}")
    for check in files.checks:
        print(f"  {'✔' if check['ok'] else '✖'} {check['name']}: {check['detail']}")
    print(f"Файлы: {out / (args.name + '.pdf')}  (.dxf, .dwg, .png рядом)")
    if args.open:
        subprocess.run(["open", str(out / f"{args.name}.pdf")], check=False)
    return 0 if all(c["ok"] for c in files.checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
