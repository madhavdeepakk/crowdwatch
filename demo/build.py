"""Build the live demo page.

    python demo/build.py

Reads ``demo/page.html`` and the recordings in ``docs/demo-data`` (made by
``crowdwatch record-demo``) and writes ``docs/index.html``, the page GitHub
Pages serves. The default recording and the typeface are embedded so the page
shows something the instant it loads; the other recordings are fetched when
chosen.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "crowdwatch" / "server" / "static" / "fonts" / "overpass-latin-wght-normal.woff2"


def parts() -> tuple[str, str]:
    src = (ROOT / "demo" / "page.html").read_text(encoding="utf-8")
    head = src.split("<!--HEAD-->")[1].split("<!--/HEAD-->")[0].strip()
    body = src.split("<!--BODY-->")[1].split("<!--/BODY-->")[0].strip()
    font = "data:font/woff2;base64," + base64.b64encode(FONT.read_bytes()).decode("ascii")
    data = (ROOT / "docs" / "demo-data" / "surge.json").read_text(encoding="utf-8").replace("</", "<\\/")
    return head.replace("__FONT__", font), body.replace("__DATA__", data)


def full_page() -> str:
    head, body = parts()
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<meta name=\"description\" content=\"Live demo of CrowdWatch: real-time overcrowding detection "
            "from CCTV video, replaying recorded runs of the real system.\">\n"
            f"{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def fragment() -> str:
    """The same page without the document wrapper, for hosts that add their own."""
    head, body = parts()
    return f"{head}\n{body}\n"


if __name__ == "__main__":
    out = ROOT / "docs" / "index.html"
    out.write_text(full_page(), encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("", encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(fragment(), encoding="utf-8")
        print(sys.argv[1])
