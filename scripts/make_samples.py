"""Generate deterministic, openly redistributable fictional PDF evaluation fixtures."""

import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "samples" / "pdfs"
    output.mkdir(parents=True, exist_ok=True)
    for document in json.loads((ROOT / "samples" / "corpus.json").read_text(encoding="utf-8")):
        canvas = Canvas(str(output / document["filename"]), pagesize=(595, 842), invariant=1)
        canvas.setTitle(document["title"])
        canvas.setAuthor("Folio assignment sample corpus")
        for number, page in enumerate(document["pages"], 1):
            canvas.setFillColor(HexColor("#21664f"))
            canvas.setFont("Helvetica-Bold", 11)
            canvas.drawString(56, 785, "NORTHSTAR / SAMPLE DOCUMENT")
            canvas.setStrokeColor(HexColor("#dde3d5"))
            canvas.line(56, 763, 539, 763)
            canvas.setFillColor(HexColor("#243c35"))
            canvas.setFont("Times-Roman", 26)
            y = 713
            for line in simpleSplit(page["heading"], "Times-Roman", 26, 470):
                canvas.drawString(56, y, line)
                y -= 33
            canvas.setFont("Helvetica", 10)
            canvas.setFillColor(HexColor("#78847c"))
            canvas.drawString(56, y - 7, document["title"])
            style = ParagraphStyle(
                "body",
                fontName="Helvetica",
                fontSize=12,
                leading=21,
                textColor=HexColor("#37493e"),
                alignment=TA_LEFT,
            )
            paragraph = Paragraph(page["text"], style)
            _, height = paragraph.wrap(475, 560)
            paragraph.drawOn(canvas, 56, y - 40 - height)
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(HexColor("#78847c"))
            canvas.drawString(
                56, 55, "FICTIONAL DATA / For reproducible testing, not organizational guidance"
            )
            canvas.drawRightString(539, 55, f"{number} / {len(document['pages'])}")
            canvas.showPage()
        canvas.save()
        print(output / document["filename"])


if __name__ == "__main__":
    main()
