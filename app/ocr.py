"""Page-preserving OCR fallback."""

import io
import subprocess

from app.errors import DocumentError
from app.observability import failure


def read_scanned_page(pdf_bytes: bytes, page_index: int, command: str = "tesseract") -> str:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise DocumentError("OCR requires the dependency pypdfium2.") from exc
    try:
        document = pdfium.PdfDocument(pdf_bytes)
        try:
            page = document[page_index]
            try:
                bitmap = page.render(scale=2)
                image = bitmap.to_pil()
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                bitmap.close()
            finally:
                page.close()
        finally:
            document.close()
        result = subprocess.run(
            [command, "stdin", "stdout", "-l", "eng", "--psm", "3"],
            input=buffer.getvalue(),
            capture_output=True,
            timeout=45,
            check=False,
        )
        if result.returncode:
            raise DocumentError("Tesseract could not read this page. Check the English language pack.")
        return result.stdout.decode("utf-8").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        failure("ocr.engine_failed", exc)
        raise DocumentError(
            "OCR failed: configure TESSERACT_CMD and install the English language pack."
        ) from exc
    except (OSError, pdfium.PdfiumError) as exc:
        failure("ocr.render_failed", exc)
        raise DocumentError("The scanned PDF page could not be rendered or read.") from exc
